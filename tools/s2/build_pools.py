"""Kaggle CPU job: PubChem isomer pools for the S2 scorer's hard negatives and its novelty validation, one shard.

For every molecular formula among train.parquet's structures (neutral formulas only; the PubChem index holds neutral
compounds), the PubChem compounds within 5 ppm of its monoisotopic mass (our index, casmi26-pubchem-index-build):
the K_DOC best documented (log1p substance + log1p PubMed records, the isomers that reach the top of real candidate
lists) and K_RND drawn uniformly from the rest, fingerprinted with the training kit's structure() (stereo removed,
canonical tautomer, Morgan radius 2, 2,048 bits; read from sanpier/casmi-spectrum-model-v1 at run time) so they
match the training molecules' fingerprints exactly. Formulas are split into NSHARD shards by hash, so several CPU
sessions share the work; within a shard the most common formulas go first, under an 11 h time limit.
Output (kernel output): pools_<c>.npz with formula (str), off (formula -> entry range), key (S14, canonical-tautomer
InChIKey first block), fp (packed 2,048 bits), doc (float16), kind (0 documented, 1 random), row (index row),
same_formula (bool); pools_<c>_report.json.

usage: python tools/s2/build_pools.py OUT_DIR SHARD NSHARD [K_DOC K_RND]
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

TEMPLATE = r'''# S2 PubChem isomer pools (Kaggle CPU), shard __C__ of __N__
import glob, hashlib, json, subprocess, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
C, N, K_DOC, K_RND, PPM, SEED, WORKERS, LIMIT_S = __C__, __N__, __K_DOC__, __K_RND__, 5.0, 20261119, 4, 11.0 * 3600
_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
_whl = [w for w in glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True) if _tag in w and "2026.3.3" in w]
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", _whl[0]], check=True)
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
kit = Path(sorted(glob.glob("/kaggle/input/**/casmi_spectrum_model_v1/features.py", recursive=True), key=len)[0]).parent
sys.path.insert(0, str(kit))
import features
comp = next(Path(t).parent for t in glob.glob("/kaggle/input/**/train.parquet", recursive=True) if "enveda" in t.lower())
INDEX = Path(sorted(glob.glob("/kaggle/input/**/pubchem_index/pc_mass.npy", recursive=True), key=len)[0]).parent
pc_mass = np.load(INDEX / "pc_mass.npy", mmap_mode="r"); pc_off = np.load(INDEX / "pc_off.npy", mmap_mode="r")
pc_smi = np.load(INDEX / "pc_smiles.npy", mmap_mode="r")

def _formula(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return rdMolDescriptors.CalcMolFormula(m), rdMolDescriptors.CalcExactMolWt(m)

# 1. formulas of the training structures (tautomers and stereoisomers share a formula and a mass)
smi = pd.unique(pq.read_table(comp / "train.parquet", columns=["normalized_smiles"]).column(0).to_pandas().astype(str)).tolist()
with mp.get_context("fork").Pool(WORKERS) as pool:
    res = pool.map(_formula, smi, chunksize=2000)
fm = pd.DataFrame([r for r in res if r is not None], columns=["formula", "mass"])
fm = fm[~fm.formula.str.contains(r"[+\-]", regex=True)]
tab = fm.groupby("formula").agg(mass=("mass", "median"), n=("mass", "size")).reset_index()
log("training SMILES", len(smi), "| neutral formulas", len(tab))
tab = tab[[int(hashlib.md5(f.encode()).hexdigest(), 16) % N == C for f in tab.formula]]
tab = tab.sort_values(["n", "formula"], ascending=[False, True]).reset_index(drop=True)
log("this shard", len(tab), "formulas covering", int(tab.n.sum()), "structures")

# 2. rows per formula: best documented, then uniform from the rest
lsid = np.load(INDEX / "pc_lsid.npy", mmap_mode="r"); lpmid = np.load(INDEX / "pc_lpmid.npy", mmap_mode="r")
rng = np.random.default_rng(SEED + C)
tasks, n_rows = [], 0
for i, (f, mass) in enumerate(zip(tab.formula, tab.mass)):
    tol = mass * PPM * 1e-6
    lo, hi = int(np.searchsorted(pc_mass, mass - tol, "left")), int(np.searchsorted(pc_mass, mass + tol, "right"))
    if hi <= lo:
        tasks.append((i, f, [], [])); continue
    d = np.asarray(lsid[lo:hi], np.float32) + np.asarray(lpmid[lo:hi], np.float32)
    o = np.lexsort((rng.random(hi - lo), -d))
    top = o[:K_DOC]; rest = o[K_DOC:]
    rnd = rng.choice(rest, size=min(K_RND, len(rest)), replace=False) if len(rest) else np.zeros(0, int)
    rows = [lo + int(x) for x in top] + [lo + int(x) for x in rnd]
    kinds = [0] * len(top) + [1] * len(rnd)
    tasks.append((i, f, rows, kinds)); n_rows += len(rows)
log("PubChem rows to fingerprint", n_rows)

def _pool(task):
    i, f, rows, kinds = task
    out = []
    for r, k in zip(rows, kinds):
        s = bytes(pc_smi[pc_off[r]:pc_off[r + 1]]).decode()
        try:
            key, _, cf, _, fp = features.structure(s, 2048)
        except Exception:
            continue
        out.append((key, np.packbits(np.asarray(fp, np.uint8)), r, k, cf == f))
    return i, out

# 3. fingerprints, most common formulas first, under the time limit
done = {}
with mp.get_context("fork").Pool(WORKERS) as pool:
    it = pool.imap(_pool, tasks, chunksize=8)
    for n in range(len(tasks)):
        if time.time() - T0 > LIMIT_S:
            log("time limit: stopped after", n, "formulas"); pool.terminate(); break
        i, out = next(it)
        done[i] = out
        if (n + 1) % 2000 == 0:
            log(f"  formulas {n + 1}/{len(tasks)}")
ids = sorted(done)
off = np.zeros(len(ids) + 1, np.int64)
key, fp, row, kind, same = [], [], [], [], []
for n, i in enumerate(ids):
    for k, b, r, t, sf in done[i]:
        key.append(k); fp.append(b); row.append(r); kind.append(t); same.append(sf)
    off[n + 1] = len(key)
d_all = np.asarray(lsid[np.asarray(row, np.int64)], np.float32) + np.asarray(lpmid[np.asarray(row, np.int64)], np.float32) if row else np.zeros(0, np.float32)
np.savez(f"/kaggle/working/pools_{C}.npz", formula=np.array([tasks[i][1] for i in ids]), off=off,
         key=np.array(key, dtype="S14"), fp=np.stack(fp) if fp else np.zeros((0, 256), np.uint8),
         doc=d_all.astype(np.float16), kind=np.array(kind, np.uint8), row=np.array(row, np.int64),
         same_formula=np.array(same, bool))
rep = dict(shard=C, nshard=N, formulas=len(tab), formulas_done=len(ids), entries=len(key),
           entries_per_formula=float(len(key) / max(1, len(ids))), same_formula_share=float(np.mean(same)) if same else None,
           structures_covered=int(tab.n.iloc[:len(ids)].sum()), seconds=round(time.time() - T0))
Path(f"/kaggle/working/pools_{C}_report.json").write_text(json.dumps(rep, indent=1))
log("done", json.dumps(rep))
'''


def main() -> None:
    out, c, n = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    k_doc = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    k_rnd = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    script = (TEMPLATE.replace("__C__", str(c)).replace("__N__", str(n))
              .replace("__K_DOC__", str(k_doc)).replace("__K_RND__", str(k_rnd)))
    ast.parse(script)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s2_pools.py").write_text(script)
    meta = {"id": f"adityachauhanx/casmi26-s2-pools-{c}", "title": f"casmi26 s2 pools {c}", "code_file": "s2_pools.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["sanpier/casmi-spectrum-model-v1", "metric/rdkit-2026-3-3-wheel"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
            "kernel_sources": ["adityachauhanx/casmi26-pubchem-index-build"], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "s2_pools.py", len(script), "bytes")


if __name__ == "__main__":
    main()
