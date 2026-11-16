"""Kaggle CPU check of the fork's spec2mol fusion step (tools/build_fork_variant.py --s2m) before a GPU variant uses it.

The step runs only inside the fork's scoring rerun (fast commits skip the pipeline), so it is exercised here on its own:
the panel's np-examples spectra stand in for effective_query_spectra.parquet, C1's class-2 lists (from the spec2mol panel
job's payload) for the candidates, and a random neural_score for the fork's blend. The check runs the exact cell code and
fusion block that build_fork_variant inserts, and reports scored candidates and MRR before / after fusion.

usage: python tools/build_s2m_fork_check.py OUT_DIR
"""
from __future__ import annotations

import ast
import base64
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_fork_variant as bfv  # noqa: E402

SCRATCH_PANEL = None   # path to the built spec2mol panel script (its payload holds C1's lists); given on the command line

TEMPLATE = r'''# Check of the fork's spec2mol fusion step on the panel (CPU)
import base64, glob, json, subprocess, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd
_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
_whl = [w for w in glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True) if _tag in w and "2026.3.3" in w]
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", _whl[0]], check=True)
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
LISTS = json.loads(zlib.decompress(base64.b64decode(__LISTS__)).decode())
ROOT = Path("/kaggle/working")
te = rdMolStandardize.TautomerEnumerator()
def key(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return ""
    Chem.RemoveStereochemistry(m)
    return Chem.MolToInchiKey(te.Canonicalize(m))[:14]
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
comp = next(Path(t).parent for t in glob.glob("/kaggle/input/**/train.parquet", recursive=True) if "enveda" in t.lower())
pf = pq.ParquetFile(comp / "train.parquet"); parts = []
for g in range(pf.num_row_groups):
    msk = pc.equal(pf.read_row_group(g, columns=["ingest_lib"])["ingest_lib"], "enveda-np-examples")
    idx = np.flatnonzero(np.asarray(msk.to_numpy(zero_copy_only=False), dtype=bool))
    if len(idx): parts.append(pf.read_row_group(g).take(pa.array(idx)))
q = pa.concat_tables(parts).to_pandas()
q["molecule_id"] = [key(s) for s in q.normalized_smiles]
q.to_parquet(ROOT / "effective_query_spectra.parquet", index=False)
rng = np.random.default_rng(0)
rows = []
for m, cand in LISTS.items():
    for r, (k, s) in enumerate(cand, 1):
        rows.append(dict(molecule_id=m, smiles=s, key=k, baseline_rank=r, neural_score=float(rng.normal()), neural_ok=True))
f = pd.DataFrame(rows); f.drop(columns=["key"]).to_parquet(ROOT / "neural_candidate_features.parquet", index=False)
answer = {(r.molecule_id, r.smiles) for r in f.itertuples() if r.key == r.molecule_id}
def mrr(frame):
    tot = 0.0
    for m, g in frame.groupby("molecule_id", sort=False):
        order = g.sort_values("neural_score", ascending=False).smiles.tolist()
        hit = [i for i, s in enumerate(order) if (m, s) in answer]
        tot += 1.0 / (hit[0] + 1) if hit else 0.0
    return tot / frame.molecule_id.nunique()
before = pd.read_parquet(ROOT / "neural_candidate_features.parquet")
print("molecules", before.molecule_id.nunique(), "candidates", len(before), "| MRR with a random neural score", round(mrr(before), 4), flush=True)
__S2M_CELL__
_pd = pd
def _z(_s):
    _g = _s.groupby(level=0)
    return (_s - _g.transform('mean')) / _g.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)
if True:
    if True:
__S2M_FUSE__
after = pd.read_parquet(ROOT / "neural_candidate_features.parquet")
print("MRR after the spectra-model fusion", round(mrr(after), 4), "| rows unchanged:", len(after) == len(before), flush=True)
'''


def main() -> None:
    out, panel_script = Path(sys.argv[1]), Path(sys.argv[2])
    src = panel_script.read_text()
    lists_lit = next(n.value.args[0].args[0].value for n in ast.walk(ast.parse(src))
                     if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LISTS")
    lists = json.loads(zlib.decompress(base64.b64decode(lists_lit)).decode())["c2"]
    payload = base64.b64encode(zlib.compress(json.dumps(lists).encode(), 9)).decode()
    script = (TEMPLATE.replace("__LISTS__", repr(payload))
              .replace("__S2M_CELL__", bfv.S2M_CELL.replace("__GAMMA__", "0.5"))
              .replace("__S2M_FUSE__", bfv.S2M_FUSE))
    ast.parse(script)
    out.mkdir(parents=True, exist_ok=True)
    (out / "s2m_fork_check.py").write_text(script)
    meta = {"id": "adityachauhanx/casmi26-s2m-fork-check", "title": "casmi26 s2m fork check", "code_file": "s2m_fork_check.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets",
                                                          "dmitriigluzdov/casmi26-fold-safe-fpnet", "metric/rdkit-2026-3-3-wheel"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "s2m_fork_check.py", len(script), "bytes;", len(lists), "molecules")


if __name__ == "__main__":
    main()
