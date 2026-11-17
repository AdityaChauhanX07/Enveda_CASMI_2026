"""Kaggle CPU job: a fingerprint head on frozen DreaMS embeddings (spectrum embedding -> Morgan fingerprint bits).

The DreaMS analogue overlay uses only the structures of a query's nearest public spectra. A head trained to predict the
query's own fingerprint from its DreaMS embedding generalises beyond nearest neighbours and scores every candidate by
how well its bits fit the prediction (Bernoulli log-likelihood, as fingerprint-based retrieval does).

Training pairs (embedding, Morgan radius-2 2,048-bit fingerprint of the spectrum's structure), all precomputed public
embeddings: MassSpecGym (hengck23 dataset), the train-library references (casmi26-dreams-train-refs-*), and a sample of
Enveda-180 (timsTOF, the test's instrument). Structures of the panel answers (enveda-np-examples) are excluded by
standard InChIKey first block, so the panel stays a fair check; 2 % of the remaining structures are held out to monitor
over-fitting. MLP 1024 -> 1536 -> 2048, binary cross-entropy, AdamW, cosine schedule, CPU.

Output (kernel output files): fphead.pt (state dict + config), panel_logits.npy (float16, one row per panel spectrum
in casmi26-dreams-panel-embed's order) and fphead_report.json.

usage: python tools/build_dreams_fphead.py OUT_DIR [E180_SAMPLE] [EPOCHS]
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

TEMPLATE = r'''# Fingerprint head on frozen DreaMS embeddings (CPU)
import glob, json, pickle, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
_whl = [w for w in glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True) if _tag in w and "2026.3.3" in w]
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", _whl[0]], check=True)
import torch, torch.nn as nn
torch.set_num_threads(4); torch.manual_seed(0)
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
T0 = time.time(); E180_N, EPOCHS, LIMIT_S = __E180__, __EPOCHS__, 10.5 * 3600
rng = np.random.default_rng(0)
D = Path(sorted(glob.glob("/kaggle/input/**/MassSpecGym_DreaMS_embeddings.npy", recursive=True), key=len)[0]).parent
P = Path(sorted(glob.glob("/kaggle/input/**/panel_rows.parquet", recursive=True), key=len)[0]).parent
panel_rows = pd.read_parquet(P / "panel_rows.parquet"); hide = set(panel_rows.ik14)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def ik14(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m)[:14] if m is not None else ""
def fpbits(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None: return None
    a = np.zeros(2048, np.uint8); a[list(gen.GetFingerprint(m).GetOnBits())] = 1
    return np.packbits(a)
sources = []                                   # (embedding array, row indices, smiles per row)
meta = pickle.load(open(D / "MassSpecGym_DreaMS_meta.pkl", "rb"))
msg_s = np.asarray(meta["normalized_smiles"] if "normalized_smiles" in meta else meta["smiles"], dtype=object); del meta
sources.append(("MassSpecGym", np.load(D / "MassSpecGym_DreaMS_embeddings.npy", mmap_mode="r"), np.arange(len(msg_s)), msg_s))
for f in sorted(glob.glob("/kaggle/input/**/train_refs_*_embeddings.npy", recursive=True)):
    mt = pd.read_parquet(f.replace("_embeddings.npy", "_meta.parquet"))
    sources.append((Path(f).stem, np.load(f, mmap_mode="r"), np.arange(len(mt)), mt.normalized_smiles.to_numpy(dtype=object)))
idx = pd.read_parquet(D / "Enveda180_DreaMS_index.parquet", columns=["normalized_smiles"])
pick = np.sort(rng.choice(len(idx), size=min(E180_N, len(idx)), replace=False))
sources.append(("Enveda180", np.load(D / "Enveda180_DreaMS_embeddings.npy", mmap_mode="r"), pick, idx.normalized_smiles.to_numpy(dtype=object)[pick]))
# structure fingerprints and exclusion of the panel answers
smiles_all = pd.unique(np.concatenate([s for *_, s in sources]))
print("unique structures", len(smiles_all), f"{time.time() - T0:.0f}s", flush=True)
fp_of, n_hidden = {}, 0
for s in smiles_all:
    k = ik14(s)
    if k in hide: n_hidden += 1; continue
    b = fpbits(s)
    if b is not None: fp_of[s] = (b, k)
print("structures with fingerprints", len(fp_of), "| panel answers excluded", n_hidden, f"{time.time() - T0:.0f}s", flush=True)
keys = sorted({k for _, k in fp_of.values()}); val_keys = set(rng.choice(keys, size=len(keys) // 50, replace=False))
X, Y, V = [], [], []
for name, emb, rows, smis in sources:
    keep = [i for i, s in enumerate(smis) if s in fp_of]
    for a in range(0, len(keep), 100000):
        part = keep[a:a + 100000]
        X.append(np.asarray(emb[rows[part]], np.float16))
        Y.append(np.stack([fp_of[smis[i]][0] for i in part])); V.append(np.array([fp_of[smis[i]][1] in val_keys for i in part]))
    print(name, "pairs", len(keep), f"{time.time() - T0:.0f}s", flush=True)
X, Y, V = np.concatenate(X), np.concatenate(Y), np.concatenate(V)
X = X.astype(np.float32); X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
tr, va = np.flatnonzero(~V), np.flatnonzero(V)
print("train", len(tr), "| validation", len(va), f"{time.time() - T0:.0f}s", flush=True)
model = nn.Sequential(nn.Linear(1024, 1536), nn.GELU(), nn.Dropout(0.1), nn.Linear(1536, 2048))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
B = 1024; steps = EPOCHS * (len(tr) // B); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=max(steps, 1), pct_start=0.05)
lossf = nn.BCEWithLogitsLoss()
def batch(ix):
    return torch.from_numpy(X[ix]), torch.from_numpy(np.unpackbits(Y[ix], axis=1).astype(np.float32))
def evaluate():
    model.eval(); tot = n = 0; ranks = []
    with torch.no_grad():
        for a in range(0, min(len(va), 50000), 4096):
            xb, yb = batch(va[a:a + 4096]); lo = model(xb)
            tot += float(lossf(lo, yb)) * len(xb); n += len(xb)
            # retrieval among the batch: does the true fingerprint score best under the predicted Bernoulli?
            lp, ln = torch.nn.functional.logsigmoid(lo), torch.nn.functional.logsigmoid(-lo)
            S = lp @ yb.T + ln @ (1 - yb).T
            ranks.append((S > S.diag()[:, None]).sum(1).numpy())
    model.train(); r = np.concatenate(ranks)
    return tot / max(n, 1), float(np.mean(1.0 / (r + 1)))
step, report = 0, dict(epochs=[])
for ep in range(EPOCHS):
    perm = rng.permutation(tr)
    for a in range(0, len(perm) - B + 1, B):
        xb, yb = batch(perm[a:a + B]); loss = lossf(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        if time.time() - T0 > LIMIT_S: break
    vl, vmrr = evaluate()
    report["epochs"].append(dict(epoch=ep + 1, val_bce=round(vl, 5), val_batch_mrr=round(vmrr, 4), seconds=round(time.time() - T0)))
    print("epoch", ep + 1, report["epochs"][-1], flush=True)
    if time.time() - T0 > LIMIT_S: break
torch.save(dict(state=model.state_dict(), config=dict(arch="1024-1536-2048 GELU", radius=2, bits=2048)), "/kaggle/working/fphead.pt")
Q = np.load(P / "panel_embeddings.npy").astype(np.float32); Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
model.eval()
with torch.no_grad():
    L = model(torch.from_numpy(Q)).numpy()
np.save("/kaggle/working/panel_logits.npy", L.astype(np.float16))
report.update(train_pairs=int(len(tr)), val_pairs=int(len(va)), structures=len(fp_of), panel_answers_excluded=n_hidden, seconds=round(time.time() - T0))
Path("/kaggle/working/fphead_report.json").write_text(json.dumps(report, indent=1))
print("done", json.dumps(report), flush=True)
'''


def main() -> None:
    out = Path(sys.argv[1])
    e180 = int(sys.argv[2]) if len(sys.argv) > 2 else 500000
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    script = TEMPLATE.replace("__E180__", str(e180)).replace("__EPOCHS__", str(epochs))
    ast.parse(script)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dreams_fphead.py").write_text(script)
    meta = {"id": "adityachauhanx/casmi26-dreams-fphead", "title": "casmi26 dreams fphead", "code_file": "dreams_fphead.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["hengck23/hengck23-dreams-enveda-casmi26", "metric/rdkit-2026-3-3-wheel"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "model_sources": [],
            "kernel_sources": ["adityachauhanx/casmi26-dreams-panel-embed"] + [f"adityachauhanx/casmi26-dreams-train-refs-{c}" for c in range(3)]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "dreams_fphead.py", len(script), "bytes")


if __name__ == "__main__":
    main()
