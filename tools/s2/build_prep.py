"""Kaggle CPU job: training cache for the S2 spectrum-candidate scorer, with a novelty holdout and the panel's lists.

Builds on Saner Turhaner's public training kit (sanpier/casmi-spectrum-model-v1: prepare.py and features.py, read
from the dataset at run time, with the `normalized_smiles` column alias the kit needs for train.parquet). Changes:
  - structures of the ~285k training SMILES are computed in 4 processes first (the kit computes them one by one);
  - each spectrum keeps its source library (spectra.parquet `original_molecule_id`) and train.parquet row number
    (`original_spectrum_id`);
  - splits (molecules.parquet `split`), replacing the kit's random 10 % holdout:
      panel       structures with Enveda NP-example spectra (the 250 panel answers); held out of training so the
                  panel stays a fair check on its real candidate lists
      validation  VAL_N structures drawn from the public libraries (not structures measured only in Enveda-180)
      excluded    every other structure with Morgan Tanimoto >= 0.6 to a validation structure, so validation
                  structures have no close relative in training - like the hidden answers, whose nearest public
                  analogues are mostly far (the panel's are close, which is why the panel overrates analogue ideas)
      train       the rest
    All four are used by a final refit; only `train` by the holdout stage;
  - the panel's C1 shortlists (dataset casmi26-s2-panel: C1's PubChem lists of the 250 panel molecules with the
    answer's own library spectra hidden, ~240 candidates each, with C1's score) are fingerprinted with the kit's
    structure(), for scoring a model on real candidate lists.
Output (kernel output, in s2_cache/): the kit cache (peaks.bin, metadata.bin, fingerprints.npy, molecules.parquet,
spectra.parquet), panel_cands.parquet + panel_cand_fp.npy (packed bits), prep_report.json.

usage: python tools/s2/build_prep.py OUT_DIR [VAL_N]
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

TEMPLATE = r'''# S2 scorer data preparation (Kaggle CPU): kit cache, novelty holdout, panel candidate fingerprints
import glob, gzip, json, shutil, subprocess, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
VAL_N, NOVEL_T, SEED, WORKERS = __VAL_N__, 0.6, 20261119, 4
_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
_whl = [w for w in glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True) if _tag in w and "2026.3.3" in w]
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", _whl[0]], check=True)
from rdkit import Chem, DataStructs, RDLogger, rdBase
RDLogger.DisableLog("rdApp.*")
log("rdkit", rdBase.rdkitVersion)
kit = Path(sorted(glob.glob("/kaggle/input/**/casmi_spectrum_model_v1/prepare.py", recursive=True), key=len)[0]).parent
stage = Path("/kaggle/working/kit_src"); shutil.copytree(kit, stage, dirs_exist_ok=True)
src = (stage / "prepare.py").read_text()
fixed = src.replace("'smiles':['smiles','canonical_smiles']", "'smiles':['smiles','canonical_smiles','normalized_smiles']")
assert fixed != src, "kit alias line not found"
(stage / "prepare.py").write_text(fixed)
sys.path.insert(0, str(stage))
import features, prepare
comp = next(Path(t).parent for t in glob.glob("/kaggle/input/**/train.parquet", recursive=True) if "enveda" in t.lower())
work = Path("/kaggle/working/s2_cache"); work.mkdir(parents=True, exist_ok=True)

def _structure(s):
    try:
        k, c, f, m, fp = features.structure(s, 2048)
        return s, (k, c, f, float(m), np.packbits(np.asarray(fp, np.uint8)))
    except Exception:
        return s, None

def structures(smiles, label):
    out, t = {}, time.time()
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for n, (s, v) in enumerate(pool.imap_unordered(_structure, smiles, chunksize=64)):
            out[s] = v
            if (n + 1) % 50000 == 0:
                log(f"  {label}: {n + 1}/{len(smiles)} structures ({time.time() - t:.0f}s)")
    return out

# 1. every training SMILES' structure (key, canonical SMILES, formula, mass, fingerprint), in parallel
smi = pq.read_table(comp / "train.parquet", columns=["normalized_smiles"]).column(0).to_pandas()
uniq = pd.unique(smi.astype(str)).tolist(); del smi
log("unique training SMILES", len(uniq))
CACHE = structures(uniq, "training SMILES")
log("valid structures", sum(v is not None for v in CACHE.values()))

def _cached(s, nbits=2048):
    v = CACHE.get(str(s))
    if v is None:
        raise ValueError("Invalid structure")
    k, c, f, m, fpp = v
    return k, c, f, m, np.unpackbits(fpp)[:nbits]
prepare.structure = _cached
_orig_batches = prepare.batches
def _batches(path):                 # library and row number of each spectrum ride along in the kit's id columns
    n0 = 0
    for f in _orig_batches(path):
        f = f.copy(); f["molecule_id"] = f["ingest_lib"].astype(str)
        f["spectrum_id"] = np.arange(n0, n0 + len(f)).astype(str); n0 += len(f)
        yield f
prepare.batches = _batches
cfg = json.loads((kit / "config.json").read_text())
cfg.update(train_path=str(comp / "train.parquet"), work=str(work), external_paths=[])
audit = prepare.prepare(cfg)
del CACHE
mol = pd.read_parquet(work / "molecules.parquet"); spec = pd.read_parquet(work / "spectra.parquet")
log("prepared: molecules", len(mol), "| spectra", len(spec))

# 2. splits
spec["lib"] = spec.original_molecule_id.astype(str)
pairs = spec[["mol_idx", "lib"]].drop_duplicates()
libs = pairs.groupby("mol_idx").lib.apply(lambda x: ",".join(sorted(x))).reindex(range(len(mol))).fillna("")
panel = libs.str.contains("enveda-np-examples", regex=False).to_numpy()
e180_only = (libs == "enveda-180").to_numpy()
log("panel structures", int(panel.sum()), "| Enveda-180-only structures", int(e180_only.sum()))
fpa = np.load(work / "fingerprints.npy")
def to_bv(row):
    v = DataStructs.ExplicitBitVect(2048); v.SetBitsFromList(np.flatnonzero(row).tolist()); return v
bvs = [to_bv(fpa[i]) for i in range(len(mol))]
log("bit vectors", len(bvs))
rng = np.random.default_rng(SEED)
pool_ids = np.flatnonzero(~panel & ~e180_only)
val = np.sort(rng.choice(pool_ids, size=min(VAL_N, len(pool_ids)), replace=False))
excluded = np.zeros(len(mol), bool)
for n, i in enumerate(val):
    excluded |= np.asarray(DataStructs.BulkTanimotoSimilarity(bvs[int(i)], bvs)) >= NOVEL_T
    if (n + 1) % 500 == 0:
        log(f"  novelty exclusion {n + 1}/{len(val)}: excluded so far {int(excluded.sum())}")
split = np.full(len(mol), "train", dtype=object)
split[excluded] = "excluded"; split[panel] = "panel"; split[val] = "validation"
val_bvs = [bvs[int(i)] for i in val]
panel_max = np.zeros(len(val))
for p in np.flatnonzero(panel):
    panel_max = np.maximum(panel_max, np.asarray(DataStructs.BulkTanimotoSimilarity(bvs[int(p)], val_bvs)))
train_bvs = [bvs[int(i)] for i in np.flatnonzero(split == "train")]
train_max = np.array([max(DataStructs.BulkTanimotoSimilarity(bvs[int(i)], train_bvs)) for i in val[:200]])
mol["split"] = split; mol["libs"] = libs.to_numpy()
mol["val_panel_max_t"] = np.nan; mol.loc[val, "val_panel_max_t"] = panel_max
mol.to_parquet(work / "molecules.parquet", index=False)
spec["split"] = spec.mol_idx.map(dict(enumerate(split)))
spec.drop(columns=["lib"]).to_parquet(work / "spectra.parquet", index=False)
counts = mol.split.value_counts().to_dict()
log("split", counts, "| validation: max Tanimoto to train (first 200) max", round(float(train_max.max()), 3),
    "| to the panel: median", round(float(np.median(panel_max)), 3))
del bvs, train_bvs, val_bvs

# 3. the panel's C1 shortlists, fingerprinted
P = json.loads(gzip.decompress(Path(sorted(glob.glob("/kaggle/input/**/panel_lists.json.gz", recursive=True), key=len)[0]).read_bytes()))
lists = P["c2"]["lists"]
cand = sorted({s for v in lists.values() for s, _ in v})
st = structures(cand, "panel candidates")
rows, fps = [], []
for m, v in lists.items():
    for j, (s, sc) in enumerate(v):
        r = st.get(s)
        rows.append((m, j, s, float(sc), r[0] if r else "", r is not None))
        fps.append(r[4] if r else np.zeros(256, np.uint8))
pc = pd.DataFrame(rows, columns=["m", "j", "smiles", "base", "key", "ok"])
pc.to_parquet(work / "panel_cands.parquet", index=False); np.save(work / "panel_cand_fp.npy", np.stack(fps))
keys_panel = set(mol.key[panel])
in_cache = sum(m in keys_panel for m in lists)
ans_in_list = sum(m in set(pc.key[pc.m == m]) for m in lists)
log("panel lists", len(lists), "| candidates", len(pc), "| fingerprinted", int(pc.ok.sum()),
    "| answers among cache panel structures", in_cache, "| answer in its list", ans_in_list)
rep = dict(molecules=len(mol), spectra=len(spec), split=counts, enveda180_only=int(e180_only.sum()),
           validation_panel_max_t=dict(median=float(np.median(panel_max)), q90=float(np.quantile(panel_max, 0.9))),
           validation_train_max_t_first200=float(train_max.max()), panel_lists=len(lists), panel_candidates=len(pc),
           panel_answers_in_cache=in_cache, panel_answer_in_list=ans_in_list, kit_audit_rejected=audit.get("rejected"),
           seconds=round(time.time() - T0))
(work / "prep_report.json").write_text(json.dumps(rep, indent=1))
log("done", json.dumps(rep))
'''


def main() -> None:
    out = Path(sys.argv[1])
    val_n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    script = TEMPLATE.replace("__VAL_N__", str(val_n))
    ast.parse(script)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s2_prep.py").write_text(script)
    meta = {"id": "adityachauhanx/casmi26-s2-prep", "title": "casmi26 s2 prep", "code_file": "s2_prep.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": ["sanpier/casmi-spectrum-model-v1", "metric/rdkit-2026-3-3-wheel", "adityachauhanx/casmi26-s2-panel"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "s2_prep.py", len(script), "bytes")


if __name__ == "__main__":
    main()
