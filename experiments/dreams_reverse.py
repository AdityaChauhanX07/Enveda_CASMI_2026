"""Candidate-centric (reverse) DreaMS support: do a candidate's structural analogues have spectra like the query?

The analogue overlay is query-centric: the query's 50 nearest public spectra vote for candidates that resemble their
structures (positive evidence only). The reverse view starts from each candidate: its analogues among MassSpecGym's
structures (Tanimoto >= t, Morgan radius 2; the candidate's own structure excluded unless it is the query's library
match; in the hidden view the answer's structure is excluded too), and how close each analogue's best spectrum is to the
query in DreaMS space. Two scores per candidate, averaged over the molecule's positive-mode spectra:
  rmax  = max over analogues of Tanimoto x cos^p        (like the forward support, but over every library structure)
  rmean = Tanimoto-weighted mean of cos^p over analogues (low when close analogues all look different: negative evidence)
Candidates without analogues get the list mean (neutral). Lists are re-ranked by
1/(3 + rank) / sd + lambda x zd(forward support, final rule) + rho x zd(reverse score).

usage: dreams_reverse.py MSG_DIR PANEL_EMBED_DIR NEIGHBOURS.json.gz LISTS.csv DUMP.pkl [--view=hidden] [--t=0.6]
       [--p=4] [--rhos=0,0.25,0.5,1] [--lam=1.5] [--grid=0,0.5,1,1.5 (forward lambdas; prints the lambda x rho grid)]
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dreams_local as dl  # noqa: E402


class SafeUnpickler(pickle.Unpickler):
    """Unpickle a downloaded metadata file without executing arbitrary code: numpy arrays and plain containers only."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"),
               ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
               ("collections", "OrderedDict"), ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        return super().find_class(module, name)


def zd(v, eps=0.01):
    return (v - v.mean()) / (v.std() + eps)


def main() -> None:
    msg_dir, emb_dir, nb_path, lists_path, dump_path = (Path(a) for a in sys.argv[1:6])
    args = dict(a[2:].split("=", 1) for a in sys.argv[6:] if a.startswith("--") and "=" in a)
    view, t_min, pw, lam = args.get("view", "hidden"), float(args.get("t", 0.6)), float(args.get("p", 4)), float(args.get("lam", 1.5))
    rhos = [float(x) for x in args.get("rhos", "0,0.25,0.5,1").split(",")]
    R = np.load(msg_dir / "MassSpecGym_DreaMS_embeddings.npy", mmap_mode="r")
    meta = SafeUnpickler(open(msg_dir / "MassSpecGym_DreaMS_meta.pkl", "rb")).load()   # downloaded file: numpy only
    rs = np.asarray(meta["normalized_smiles"] if "normalized_smiles" in meta else meta["smiles"], dtype=object); del meta
    uniq, inv = np.unique(rs, return_inverse=True)
    print("MassSpecGym spectra", len(rs), "| structures", len(uniq), flush=True)
    u_fp = [dl.fp(s_) for s_ in uniq]; u_ok = np.array([f is not None for f in u_fp])
    u_ik = np.array([dl.std_ik(s_) for s_ in uniq], dtype=object)
    fps_ok = [f for f in u_fp if f is not None]; idx_ok = np.flatnonzero(u_ok)
    Q = np.load(emb_dir / "panel_embeddings.npy").astype(np.float32)
    rows = pd.read_parquet(emb_dir / "panel_rows.parquet")
    by_m = {}
    for i, (m, md, ik) in enumerate(zip(rows.m, rows["mode"], rows.ik14)):
        by_m.setdefault(m, []).append((i, md, ik))
    use = {m: [i for i, md, _ in v if md == "pos"] or [i for i, _, _ in v] for m, v in by_m.items()}
    ans_ik = {m: v[0][2] for m, v in by_m.items()}
    sub = pd.read_csv(lists_path, dtype=str, keep_default_na=False)
    lists = {m: s.split(";") for m, s in zip(sub.molecule_id, sub.smiles)}
    hits = pickle.load(open(dump_path, "rb"))["hits"]
    selfok = {m: {dl.std_ik(s_) for c, s_ in hits.get(m, {}).values() if c >= 0.8} for m in lists}
    heads = {}
    for m, cand in lists.items():
        ks = {s_ for c, s_ in hits.get(m, {}).values() if c >= 0.95}
        if ks:
            heads[m] = next((i for i, c in enumerate(cand) if c not in ks), len(cand))
    # best cosine of each MassSpecGym structure's spectra, for every query spectrum used (one pass over the references)
    ms = [m for m in lists if m in use and len(lists[m]) > 1]
    qi = sorted({i for m in ms for i in use[m]}); qpos = {i: k for k, i in enumerate(qi)}
    Qn = Q[qi] / np.maximum(np.linalg.norm(Q[qi], axis=1, keepdims=True), 1e-9)
    order_r = np.argsort(inv, kind="stable"); bounds = np.flatnonzero(np.r_[True, np.diff(inv[order_r]) != 0])
    best = np.empty((len(qi), len(uniq)), np.float32)
    sims = np.empty((len(qi), len(rs)), np.float32)
    for a in range(0, len(rs), 50000):
        blk = np.asarray(R[order_r[a:a + 50000]], np.float32)
        sims[:, a:a + 50000] = Qn @ (blk / np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-9)).T
    best[:] = np.maximum.reduceat(sims, bounds, axis=1); del sims
    struct_of_block = inv[order_r][bounds]                 # structure index of each reduced column
    cos_by_struct = np.empty_like(best); cos_by_struct[:, struct_of_block] = best; del best
    cos_by_struct = np.clip(cos_by_struct, 0.0, 1.0)
    print("query x structure cosines", cos_by_struct.shape, flush=True)
    rmax, rmean = {}, {}
    for n, m in enumerate(ms):
        cos_struct = cos_by_struct[[qpos[i] for i in use[m]]].mean(0)
        a_max, a_mean = [], []
        for s_ in lists[m]:
            f = dl.fp(s_)
            if f is None:
                a_max.append(np.nan); a_mean.append(np.nan); continue
            tani = np.array(DataStructs.BulkTanimotoSimilarity(f, fps_ok), np.float32)
            sel = idx_ok[tani >= t_min]; tv = tani[tani >= t_min]
            cik = dl.std_ik(s_)
            keep = np.array([(u_ik[j] != cik or cik in selfok.get(m, ())) and not (view == "hidden" and u_ik[j] == ans_ik[m])
                             for j in sel], bool) if len(sel) else np.zeros(0, bool)
            sel, tv = sel[keep], tv[keep]
            if not len(sel):
                a_max.append(np.nan); a_mean.append(np.nan); continue
            cv = cos_struct[sel] ** pw
            a_max.append(float((tv * cv).max())); a_mean.append(float((tv * cv).sum() / tv.sum()))
        for store, vals in ((rmax, a_max), (rmean, a_mean)):
            v = np.array(vals, np.float64)
            store[m] = np.where(np.isfinite(v), v, np.nanmean(v) if np.isfinite(v).any() else 0.0)
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(ms)} molecules", flush=True)
    nbd = json.loads(gzip.decompress(nb_path.read_bytes()))
    dl.MODES = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "dreams" / "panel_modes.json").read_text())
    sup = dl.supports(nbd, lists, view=view, noself=True, selfok=selfok)
    keys = dl._key_cache([s_ for c in lists.values() for s_ in c], dl.metric_key)

    def rr(m, lst):
        k = [keys[s_] for s_ in lst[:25]]
        return 1.0 / (k.index(m) + 1) if m in k else 0.0

    def order(m, score):
        cand = lists[m]; new = [cand[i] for i in np.argsort(-score, kind="stable")]; h = heads.get(m, 0)
        return cand[:h] + [c for c in new if c not in set(cand[:h])] if h else new
    ms = [m for m in ms if m in sup]
    print(f"{lists_path.name} | n {len(ms)} | t {t_min} p {pw:g} | list order {np.mean([rr(m, lists[m]) for m in ms]):.4f} | "
          f"rmax alone {np.mean([rr(m, order(m, rmax[m])) for m in ms]):.4f} | rmean alone {np.mean([rr(m, order(m, rmean[m])) for m in ms]):.4f}")
    if args.get("grid"):                                # joint forward lambda x reverse rho grid (rmax)
        for lam_g in [float(x) for x in args.get("grid").split(",")]:
            res = []
            for rho in rhos:
                v = []
                for m in ms:
                    b = 1.0 / (3.0 + np.arange(1, len(lists[m]) + 1))
                    v.append(rr(m, order(m, b / b.std() + lam_g * zd(sup[m]) + rho * zd(rmax[m]))))
                res.append(f"{rho:g}: {np.mean(v):.4f}")
            print(f"  grid lambda {lam_g:<4g} + rho x rmax | " + " | ".join(res), flush=True)
        return
    for name, rev in (("rmax", rmax), ("rmean", rmean)):
        for with_fwd in (False, True):
            res = []
            for rho in rhos:
                v = []
                for m in ms:
                    b = 1.0 / (3.0 + np.arange(1, len(lists[m]) + 1))
                    v.append(rr(m, order(m, b / b.std() + (lam * zd(sup[m]) if with_fwd else 0.0) + rho * zd(rev[m]))))
                res.append(f"{rho:g}: {np.mean(v):.4f}")
            print(f"  {name} {'+ forward overlay' if with_fwd else 'on list order    '} | " + " | ".join(res), flush=True)


if __name__ == "__main__":
    main()
