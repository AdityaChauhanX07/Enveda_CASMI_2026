"""Kaggle job (GPU; CPU for a smoke test): S2 spectrum-candidate scorer with PubChem isomer negatives.

Model, features and the training recipe are Saner Turhaner's public kit (sanpier/casmi-spectrum-model-v1: model.py,
train.py, config.json, read at run time; architecture unchanged, so the result drops into any pipeline that loads the
kit's model with the kit's config.json). The change is the contrastive denominator: next to the kit's 16 hard
negatives (training molecules of the same formula or nearest mass) each query also sees its formula's PubChem pool
(up to 32 PubChem compounds within 5 ppm: the 16 best documented and 16 random ones, from casmi26-s2-pools-*), i.e.
the isomers it will meet in real candidate lists, most of which have no reference spectra.
Stages:
  holdout stages (one per STAGES entry, same seed, `train` split of casmi26-s2-prep only), each scored every
  VAL_EVERY epochs on
    - novelty holdout, PubChem pool: rank of the true structure among its formula's PubChem pool (no training
      structure within Tanimoto 0.6 of these molecules: the hidden answers' regime);
    - novelty holdout, kit diagnostic: the kit's own check (true structure + 255 known structures of the same formula or
      nearest mass);
    - panel: C1's real shortlists of the 250 panel molecules (answer's library spectra hidden), scored from their
      Enveda NP-example spectra: the model alone and fused with C1's score (z + gamma x z(model));
  refit: the chosen negatives on every structure (all splits) -> final_model.pt (+ config_used.json), the only file of
  that name in the output.
Output (kernel output, s2_model/): final_model.pt, config_used.json, config.json / model.py / features.py /
inference.py (the kit's, unchanged), holdout_<stage>_best.pt, s2_report.json.

usage: python tools/s2/build_train.py OUT_DIR [--smoke] [--gpu] [--stages=A:kit:40,B:kit+pc:40] [--refit=auto:64]
       [--hours=1.75]
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

TEMPLATE = r'''# S2 spectrum-candidate scorer: PubChem isomer negatives (kit architecture), holdout comparison then refit
import glob, json, os, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.nn import functional as F
T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
SMOKE, STAGES, REFIT, HOURS, VAL_EVERY, PMAX, GAMMAS = __SMOKE__, __STAGES__, __REFIT__, __HOURS__, 2, 32, (0.25, 0.5, 1.0, 2.0)
DEADLINE = T0 + HOURS * 3600
kit = Path(sorted(glob.glob("/kaggle/input/**/casmi_spectrum_model_v1/train.py", recursive=True), key=len)[0]).parent
sys.path.insert(0, str(kit))
from model import SpectrumCandidateModel
import train as kt
cache = Path(sorted(glob.glob("/kaggle/input/**/s2_cache/molecules.parquet", recursive=True), key=len)[0]).parent
cfg = json.loads((kit / "config.json").read_text())
cfg.update(work=str(cache), require_gpu=False)
if SMOKE:
    cfg.update(steps_per_epoch=12, max_validation_molecules=60)
OUT = Path("/kaggle/working/s2_model"); OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(4, os.cpu_count() or 1))
log("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", "| torch", torch.__version__)
data = kt.Data(cfg)
data.x = np.array(data.x); data.meta = np.array(data.meta); data.fp = np.load(cache / "fingerprints.npy")
mol, spec = data.mol, data.spec
N_MOL, B, H, T, W, BITS = len(mol), cfg["batch_size"], cfg["hard_negatives"], cfg["temperature"], cfg["fp_score_weight"], cfg["fp_bits"]
log("molecules", N_MOL, "| spectra", len(spec), "| split", mol.split.value_counts().to_dict())

# PubChem pools: formula -> up to PMAX entries (unique canonical keys), entry -> training molecule (or -1)
parts = [np.load(f) for f in sorted(glob.glob("/kaggle/input/**/pools_*.npz", recursive=True))]
pkey = np.concatenate([p["key"] for p in parts]); pfp = np.concatenate([p["fp"] for p in parts])
pkind = np.concatenate([p["kind"] for p in parts]); prange, base = {}, 0
for p in parts:
    for f, a, b in zip(p["formula"], p["off"][:-1], p["off"][1:]):
        prange[str(f)] = (base + int(a), base + int(b))
    base += len(p["key"])
key_to_mol = pd.Series(np.arange(N_MOL), index=mol.key.to_numpy())
emol = key_to_mol.reindex(pd.Index(pkey.astype(str))).fillna(-1).astype(np.int64).to_numpy()
pc_idx = np.full((N_MOL, PMAX), -1, np.int64)
for f, rows in mol.groupby("formula").indices.items():
    if f not in prange:
        continue
    a, b = prange[f]; ent = np.arange(a, b)
    _, first = np.unique(pkey[ent], return_index=True); ent = ent[np.sort(first)][:PMAX]
    pc_idx[rows, :len(ent)] = ent
has_pool = (pc_idx >= 0).any(1)
log("PubChem pools: shards", len(parts), "| formulas", len(prange), "| entries", len(pkey), "| entries that are training molecules",
    int((emol >= 0).sum()), "| molecules with a pool", int(has_pool.sum()), "| mean pool", float((pc_idx >= 0).sum(1)[has_pool].mean()))
PFP = torch.from_numpy(pfp).to(device)
SHIFTS = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)
def unpack(b):                                  # packed (..., 256) uint8 -> (..., 2048) float, numpy bit order
    return ((b.unsqueeze(-1) >> SHIFTS) & 1).flatten(-2).float()
def unpack_np(b):
    return torch.from_numpy(np.unpackbits(b, axis=-1).astype(np.float32)).to(device)

# evaluation sets
val_ids = data.val[:cfg["max_validation_molecules"]]
kit_cands = kt.candidate_sets(data, cfg)
val_pc = {}
for i in val_ids:
    ent = pc_idx[i][pc_idx[i] >= 0]
    ent = ent[emol[ent] != i]
    if len(ent):
        val_pc[int(i)] = ent
pc_ = pd.read_parquet(cache / "panel_cands.parquet"); pcfp = np.load(cache / "panel_cand_fp.npy")
lib = spec.original_molecule_id.astype(str).to_numpy()
panel_rows = {}
for m in pc_.m.unique():
    i = key_to_mol.get(m)
    if i is None:
        continue
    r = data.by_mol[int(i)]; r = r[lib[r] == "enveda-np-examples"]
    if len(r):
        panel_rows[m] = r
panel_lists = {m: g.index.to_numpy() for m, g in pc_[pc_.ok].groupby("m") if m in panel_rows}
log("evaluation: novelty holdout", len(val_ids), "| with a PubChem pool", len(val_pc), "| mean pool", float(np.mean([len(v) for v in val_pc.values()])),
    "| panel molecules", len(panel_lists))

@torch.inference_mode()
def queries(model, rows_list):
    model.eval()
    allr = np.concatenate(rows_list); seg = torch.from_numpy(np.repeat(np.arange(len(rows_list)), [len(r) for r in rows_list])).to(device)
    Z, P = [], []
    for a in range(0, len(allr), 512):
        z, p = model.encode_spectra(*data.batch(allr[a:a + 512], device)); Z.append(z.float()); P.append(p.float())
    Z, P = torch.cat(Z), torch.cat(P)
    n = torch.bincount(seg, minlength=len(rows_list)).float()[:, None]
    zs = torch.zeros(len(rows_list), Z.shape[1], device=device).index_add_(0, seg, Z) / n
    ps = torch.zeros(len(rows_list), P.shape[1], device=device).index_add_(0, seg, P) / n
    return zs, ps

@torch.inference_mode()
def score(model, fp, z, p):                     # the kit's inference score (inference.Scorer)
    return ((model.encode_molecules(fp) @ z) / T + W * (fp @ p) / BITS).cpu().numpy()

def rr_keys(order_keys, ans, cut=25):
    seen, k = set(), 0
    for x in order_keys:
        if x in seen:
            continue
        seen.add(x); k += 1
        if x == ans:
            return 1.0 / k if k <= cut else 0.0, k
    return 0.0, None

def zz(v):
    return (v - v.mean()) / (v.std() + 1e-9)

@torch.inference_mode()
def evaluate(model):
    res = {}
    ids = list(val_ids)
    zs, ps = queries(model, [data.by_mol[int(i)] for i in ids])
    kit_rr, pc_rr, pc_top1, pcdoc_rr = [], [], [], []
    for n, i in enumerate(ids):
        cand = kit_cands.get(int(i))
        if cand is not None:
            s = score(model, torch.as_tensor(np.asarray(data.fp[cand]), device=device, dtype=torch.float32), zs[n], ps[n])
            r = 1 + int(np.sum(s[1:] >= s[0])); kit_rr.append(1 / r if r <= 25 else 0.0)
        ent = val_pc.get(int(i))
        if ent is not None:
            fp = torch.cat([torch.as_tensor(np.asarray(data.fp[[i]]), device=device, dtype=torch.float32), unpack(PFP[torch.from_numpy(ent).to(device)])])
            s = score(model, fp, zs[n], ps[n])
            r = 1 + int(np.sum(s[1:] >= s[0])); pc_rr.append(1 / r); pc_top1.append(float(r == 1))
            d = np.r_[True, pkind[ent] == 0]; sd = s[d]
            r = 1 + int(np.sum(sd[1:] >= sd[0])); pcdoc_rr.append(1 / r)
    res.update(kit_mrr25=float(np.mean(kit_rr)), pc_mrr=float(np.mean(pc_rr)), pc_top1=float(np.mean(pc_top1)),
               pc_doc_mrr=float(np.mean(pcdoc_rr)), n_pc=len(pc_rr))
    ms = list(panel_lists)
    zs, ps = queries(model, [panel_rows[m] for m in ms])
    acc = {"base": [], "model": [], **{f"g{g:g}": [] for g in GAMMAS}}
    for n, m in enumerate(ms):
        idx = panel_lists[m]; keys = pc_.key.to_numpy()[idx]; b = pc_.base.to_numpy()[idx]
        s = score(model, unpack_np(pcfp[idx]), zs[n], ps[n])
        for name, v in (("base", b), ("model", s), *((f"g{g:g}", zz(b) + g * zz(s)) for g in GAMMAS)):
            acc[name].append(rr_keys(keys[np.argsort(-v, kind="stable")], m)[0])
    res.update({f"panel_{k}": float(np.mean(v)) for k, v in acc.items()}); res["n_panel"] = len(ms)
    return res

def train_stage(name, neg, epochs, refit=False):
    seed = cfg["seed"] + (1 if refit else 0)
    kt.seed_all(seed); rng = np.random.default_rng(seed)
    ids = np.arange(N_MOL) if refit else data.train
    allowed = np.zeros(N_MOL, bool); allowed[ids] = True
    use_kit, use_pc = "kit" in neg, "pc" in neg
    if refit not in POOLS:                        # the holdout stages share one split: build its pools once
        tp = time.time(); POOLS[refit] = kt.hard_pools(mol, ids, cfg["negative_pool"]); log("kit pools", len(POOLS[refit]), f"{time.time() - tp:.0f}s")
    pools = POOLS[refit]
    model = SpectrumCandidateModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    use_amp = device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    counts = np.zeros(BITS)
    for a in range(0, len(ids), 4096):
        counts += np.asarray(data.fp[ids[a:a + 4096]]).sum(0)
    pos_weight = torch.tensor(np.clip((len(ids) - counts) / (counts + 1), 1, 20), device=device, dtype=torch.float32)
    hist, best, best_ep, bad, stopped = [], -1.0, 0, 0, False
    cap = DEADLINE - (0 if refit else REFIT_RESERVE)
    for ep in range(1, epochs + 1):
        model.train(); losses, t = [], time.time(); n_pc = 0
        for step in range(cfg["steps_per_epoch"]):
            if time.time() > cap:
                stopped = True; break
            bi = rng.choice(ids, min(B, len(ids)), replace=False)
            sr = np.array([rng.choice(data.by_mol[int(i)]) for i in bi])
            fp = torch.tensor(np.asarray(data.fp[bi]), device=device, dtype=torch.float32)
            parts_l, masks = [], []
            if use_kit:
                negi = np.stack([rng.choice(pools[int(i)], H, replace=len(pools[int(i)]) < H) for i in bi])
                dup = np.isin(negi, bi)
                for r in range(len(negi)):
                    seen = set()
                    for j, k in enumerate(negi[r]):
                        if int(k) in seen:
                            dup[r, j] = True
                        seen.add(int(k))
                negfp = torch.tensor(np.asarray(data.fp[negi.reshape(-1)]), device=device, dtype=torch.float32)
            if use_pc:
                pidx = pc_idx[bi]; pm = np.where(pidx >= 0, emol[np.maximum(pidx, 0)], -2)
                pmask = (pidx < 0) | np.isin(pm, bi) | ((pm >= 0) & ~allowed[np.maximum(pm, 0)])
                if use_kit:
                    pmask |= (pm[:, :, None] == negi[:, None, :]).any(-1)
                n_pc += int((~pmask).sum())
                pcf = unpack(PFP[torch.from_numpy(np.maximum(pidx, 0)).to(device)])
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                z, pred = model.encode_spectra(*data.batch(sr, device, augment=True))
                m = model.encode_molecules(fp)
                logits = [z @ m.T / T]
                if use_kit:
                    nm = model.encode_molecules(negfp).reshape(len(bi), H, -1)
                    logits.append(torch.einsum("bd,bkd->bk", z, nm).div(T).masked_fill(torch.from_numpy(dup).to(device), float("-inf")))
                if use_pc:
                    pe = model.encode_molecules(pcf.reshape(-1, BITS)).reshape(len(bi), PMAX, -1)
                    logits.append(torch.einsum("bd,bkd->bk", z, pe).div(T).masked_fill(torch.from_numpy(pmask).to(device), float("-inf")))
                retrieval = F.cross_entropy(torch.cat(logits, 1).float(), torch.arange(len(bi), device=device))
                auxiliary = F.binary_cross_entropy_with_logits(pred.float(), fp, pos_weight=pos_weight)
                loss = retrieval + cfg["aux_weight"] * auxiliary
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
        if not losses:
            break
        ck = dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, epoch=ep, stage=name, negatives=neg)
        rec = dict(epoch=ep, loss=float(np.mean(losses)), steps=len(losses), seconds=round(time.time() - t, 1),
                   pc_negatives_per_query=n_pc / max(1, len(losses) * B), partial=stopped)
        if refit:
            kt.atomic_save(ck, OUT / "final_model.pt")
        elif ep % VAL_EVERY == 0 or ep == epochs or stopped:
            te = time.time(); rec.update(evaluate(model)); rec["eval_seconds"] = round(time.time() - te, 1)
            if rec["pc_mrr"] > best:
                best, best_ep, bad = rec["pc_mrr"], ep, 0
                kt.atomic_save(ck, OUT / f"holdout_{name}_best.pt")
            else:
                bad += 1
        hist.append(rec); log(name, "REFIT" if refit else "HOLDOUT", json.dumps(rec))
        if stopped or (not refit and bad >= PATIENCE):
            break
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dict(name=name, negatives=neg, epochs=epochs, best_pc_mrr=best, best_epoch=best_ep, history=hist, stopped=stopped)

PATIENCE = 3                                     # evaluations without a better novelty-holdout PubChem MRR
POOLS = {}
REFIT_RESERVE = REFIT["reserve_hours"] * 3600 if REFIT else 0
report = dict(stages=[], cfg=cfg)
for st in STAGES:
    report["stages"].append(train_stage(st["name"], st["neg"], st["epochs"]))
    (OUT / "s2_report.json").write_text(json.dumps(report, indent=1, default=str))
if REFIT:
    done = [s for s in report["stages"] if s["best_epoch"]]
    neg = REFIT["neg"]
    if neg == "auto":
        neg = max(done, key=lambda s: (s["best_pc_mrr"], -("pc" in s["negatives"])))["negatives"]
    ref = next(s for s in done if s["negatives"] == neg) if done else None
    ep = REFIT["epochs"]
    if ref is not None and ref["best_epoch"] < 0.75 * ref["epochs"]:
        ep = min(ep, ref["best_epoch"])
    if SMOKE:
        ep = 1
    log("refit on every structure:", neg, ep, "epochs")
    report["refit"] = train_stage("refit", neg, ep, refit=True)
    cfg_used = dict(cfg, s2=dict(negatives=neg, refit_epochs=ep, pubchem_pool=PMAX, stages=[s["name"] for s in report["stages"]]))
    (OUT / "config_used.json").write_text(json.dumps(cfg_used, indent=1))
for f in ("config.json", "model.py", "features.py", "inference.py"):
    shutil.copy2(kit / f, OUT / f)
report["seconds"] = round(time.time() - T0)
(OUT / "s2_report.json").write_text(json.dumps(report, indent=1, default=str))
log("done", json.dumps({s["name"]: (s["best_pc_mrr"], s["best_epoch"]) for s in report["stages"]}))
'''


def main() -> None:
    out = Path(sys.argv[1])
    args = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[2:] if a.startswith("--"))
    smoke, gpu = "smoke" in args, "gpu" in args
    stages = []
    for item in args.get("stages", "A:kit:40,B:kit+pc:40").split(","):
        name, neg, ep = item.split(":")
        assert neg in ("kit", "kit+pc", "pc"), neg
        stages.append(dict(name=name, neg=neg, epochs=2 if smoke else int(ep)))
    refit = None
    if args.get("refit", "auto:64") != "none":
        neg, ep = args.get("refit", "auto:64").split(":")
        refit = dict(neg=neg, epochs=int(ep), reserve_hours=0.05 if smoke else 0.6)
    hours = float(args.get("hours", 1.75))
    script = (TEMPLATE.replace("__SMOKE__", str(smoke)).replace("__STAGES__", repr(stages))
              .replace("__REFIT__", repr(refit)).replace("__HOURS__", repr(hours)))
    ast.parse(script)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s2_train.py").write_text(script)
    slug = "casmi26-s2-train" + ("-smoke" if smoke else "")
    meta = {"id": f"adityachauhanx/{slug}", "title": slug.replace("-", " "), "code_file": "s2_train.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["sanpier/casmi-spectrum-model-v1"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
            "kernel_sources": ["adityachauhanx/casmi26-s2-prep"] + [f"adityachauhanx/casmi26-s2-pools-{c}" for c in range(4)],
            "model_sources": []}
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "s2_train.py", len(script), "bytes", "| stages", stages, "| refit", refit, "| hours", hours)


if __name__ == "__main__":
    main()
