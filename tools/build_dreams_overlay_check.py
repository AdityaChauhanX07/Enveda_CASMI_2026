"""Kaggle CPU check of casmi/dreams_overlay.py on the panel before a notebook uses it.

The overlay runs exactly as in a notebook (apply()) on the panel's np-examples spectra, re-ranking the documentation-free
C1e final lists (real-pipeline dump), with the answer's own structure hidden from the MassSpecGym neighbours. It should
reproduce the experiment (tools/build_dreams_panel.py: 0.588 -> ~0.614 at lam 1).

usage: python tools/build_dreams_overlay_check.py OUT_DIR FINAL_LISTS.csv [LAMBDA]
"""
from __future__ import annotations

import ast
import base64
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = r'''# Check of the DreaMS analogue overlay on the panel (CPU)
import base64, glob, json, subprocess, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd
_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
_whl = [w for w in glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True) if _tag in w and "2026.3.3" in w]
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", _whl[0]], check=True)
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
unpack = lambda t: zlib.decompress(base64.b64decode(t)).decode()
W = Path("/kaggle/working"); (W / "dreams_overlay.py").write_text(unpack(__SRC__)); sys.path.insert(0, str(W))
import dreams_overlay as do
(W / "lists.csv").write_text(unpack(__LISTS__))
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
hide = {m: {Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14]} for m, s in zip(q.molecule_id, q.normalized_smiles)}
q.to_parquet(W / "panel_test.parquet", index=False)
dd = Path(sorted(glob.glob("/kaggle/input/**/MassSpecGym_DreaMS_embeddings.npy", recursive=True), key=len)[0]).parent
rep = do.apply(W / "panel_test.parquet", W / "lists.csv", dd, W / "lists_dreams.csv", W / "report.json", dict(lam=__LAM__), hide=hide)
print("REPORT", json.dumps(rep), flush=True)
def mrr(path):
    s = pd.read_csv(path, dtype=str, keep_default_na=False); tot = 0.0
    for m, smi in zip(s.molecule_id, s.smiles):
        ks = [key(x) for x in smi.split(";")]
        tot += 1.0 / (ks.index(m) + 1) if m in ks[:25] else 0.0
    return tot / len(s)
print("MRR before", round(mrr(W / "lists.csv"), 4), "| after the overlay", round(mrr(W / "lists_dreams.csv"), 4), flush=True)
'''


def main() -> None:
    out, lists_csv = Path(sys.argv[1]), Path(sys.argv[2])
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    pack = lambda t: base64.b64encode(zlib.compress(t.encode(), 9)).decode()
    script = (TEMPLATE.replace("__SRC__", repr(pack((ROOT / "casmi" / "dreams_overlay.py").read_text())))
              .replace("__LISTS__", repr(pack(lists_csv.read_text()))).replace("__LAM__", repr(lam)))
    ast.parse(script)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dreams_overlay_check.py").write_text(script)
    meta = {"id": "adityachauhanx/casmi26-dreams-overlay-check", "title": "casmi26 dreams overlay check",
            "code_file": "dreams_overlay_check.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["hengck23/hengck23-dreams-enveda-casmi26", "metric/rdkit-2026-3-3-wheel"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "dreams_overlay_check.py", len(script), "bytes")


if __name__ == "__main__":
    main()
