"""Kaggle CPU job: DreaMS embeddings of the panel spectra (enveda-np-examples), saved for offline checks.

Used by experiments that need the query embeddings themselves rather than their MassSpecGym neighbours, e.g. an
in-test network built from DreaMS similarity between test molecules (experiments/dreams_network.py).

Output (kernel output files): panel_embeddings.npy (float16, L2-normalised, one row per spectrum in train.parquet's
row-group order, the order the other panel jobs use) and panel_rows.parquet (molecule key = the answer's metric key,
standard InChIKey first block, ion mode, adduct).

usage: python tools/build_dreams_panel_embed.py OUT_DIR
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

TEMPLATE = r'''# DreaMS embeddings of the panel spectra (CPU)
import glob, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
_whl = [w for w in glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True) if _tag in w and "2026.3.3" in w]
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", _whl[0]], check=True)
import torch
torch.set_num_threads(4)
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
T0 = time.time()
D = Path(sorted(glob.glob("/kaggle/input/**/MassSpecGym_DreaMS_embeddings.npy", recursive=True), key=len)[0]).parent
sys.path.insert(0, str(D))
import dreams_minimal_model as dm
model = dm.load_model(str(D / "embedding_model_state_dict.pt"), device="cpu")
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
x = np.stack([dm.preprocess_spectrum(np.asarray(r.ms2_mzs, np.float32), np.asarray(r.ms2_normalized_intensities, np.float32),
                                     float(r.precursor_mz)) for r in q.itertuples()])
with torch.inference_mode():
    Q = np.concatenate([np.asarray(dm.embed_preprocessed(model, x[i:i + 64])) for i in range(0, len(x), 64)])
Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
np.save("/kaggle/working/panel_embeddings.npy", Q.astype(np.float16))
pd.DataFrame({"m": [key(s) for s in q.normalized_smiles],
              "ik14": [Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for s in q.normalized_smiles],
              "mode": q.ionization_mode.astype(str).str.lower().str[:3], "adduct": q.adduct.astype(str)}).to_parquet(
    "/kaggle/working/panel_rows.parquet", index=False)
print("saved", Q.shape, f"{time.time() - T0:.0f}s", flush=True)
'''


def main() -> None:
    out = Path(sys.argv[1])
    ast.parse(TEMPLATE)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dreams_panel_embed.py").write_text(TEMPLATE)
    meta = {"id": "adityachauhanx/casmi26-dreams-panel-embed", "title": "casmi26 dreams panel embed",
            "code_file": "dreams_panel_embed.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["hengck23/hengck23-dreams-enveda-casmi26", "metric/rdkit-2026-3-3-wheel"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "dreams_panel_embed.py")


if __name__ == "__main__":
    main()
