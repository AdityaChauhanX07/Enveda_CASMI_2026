"""Kaggle CPU job: DreaMS embeddings of train.parquet's public-library spectra that MassSpecGym does not cover.

The DreaMS analogue overlay (casmi/dreams_overlay.py) finds a query's nearest public spectra in MassSpecGym (hengck23's
precomputed embeddings: ~231k positive-mode spectra of ~31k structures). train.parquet's other public libraries
(pluskal_ms2, riken, gnps, massbank, mona, spectraverse, msdial, ...) hold ~95k structures and negative-mode spectra
that MassSpecGym lacks. This job embeds one spectrum per (structure, ion mode) absent from MassSpecGym, chosen by
adduct ([M+H]+ / [M-H]- first), then the most peaks (capped at 40), then collision energy closest to 35 eV; the
selection is deterministic and split into NCHUNK interleaved chunks, one per job, so several CPU sessions share it.

Output (kernel output files, attach the kernels as sources): train_refs_<c>_embeddings.npy (float16, L2-normalised)
and train_refs_<c>_meta.parquet (normalized_smiles, inchikey14, ingest_lib, adduct, ionization_mode, train_row).

usage: python tools/build_dreams_train_refs.py OUT_DIR CHUNK NCHUNK
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

TEMPLATE = r'''# DreaMS embeddings of train.parquet public-library spectra not covered by MassSpecGym (CPU), chunk __C__ of __N__
import glob, pickle, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
_whl = [w for w in glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True) if _tag in w and "2026.3.3" in w]
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", _whl[0]], check=True)
import torch
torch.set_num_threads(4)
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
T0 = time.time(); C, N = __C__, __N__; LIMIT_S = 11.0 * 3600
D = Path(sorted(glob.glob("/kaggle/input/**/MassSpecGym_DreaMS_embeddings.npy", recursive=True), key=len)[0]).parent
sys.path.insert(0, str(D))
import dreams_minimal_model as dm
meta = pickle.load(open(D / "MassSpecGym_DreaMS_meta.pkl", "rb"))
msg_smiles = pd.unique(np.asarray(meta["normalized_smiles"] if "normalized_smiles" in meta else meta["smiles"], dtype=object))
del meta
def ik14(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToInchiKey(m)[:14] if m is not None else ""
msg_ik = {ik14(s) for s in msg_smiles}
print("MassSpecGym structures", len(msg_ik), f"{time.time() - T0:.0f}s", flush=True)
import pyarrow as pa, pyarrow.parquet as pq
comp = next(Path(t).parent for t in glob.glob("/kaggle/input/**/train.parquet", recursive=True) if "enveda" in t.lower())
cols = ["ingest_lib", "inchikey14", "normalized_smiles", "adduct", "ionization_mode", "num_peaks", "collision_energy_ev"]
t = pq.read_table(comp / "train.parquet", columns=cols).to_pandas()
t["train_row"] = np.arange(len(t))
t = t[~t.ingest_lib.isin(["enveda-180", "enveda-np-examples"])]
# standard InChIKey first block of each structure, to compare with MassSpecGym's (the train column is the same key)
t = t[t.inchikey14.astype(str).str.len() == 14]
pos = t.ionization_mode.astype(str).str.lower().str.startswith("pos")
t = t[~(pos & t.inchikey14.isin(msg_ik))]                      # MassSpecGym is positive-mode only
t["mode"] = np.where(t.ionization_mode.astype(str).str.lower().str.startswith("pos"), "pos", "neg")
t["prio"] = (~t.adduct.isin(["[M+H]+", "[M-H]-"])).astype(int)
t["peaks"] = -np.minimum(t.num_peaks.fillna(0).astype(float), 40.0)
t["ce"] = [abs(float(np.mean(v)) - 35.0) if v is not None and len(np.atleast_1d(v)) else 99.0 for v in t.collision_energy_ev]
t = t.sort_values(["inchikey14", "mode", "prio", "peaks", "ce", "train_row"]).drop_duplicates(["inchikey14", "mode"])
t = t.reset_index(drop=True)
print("selected (structure, mode) spectra", len(t), "| by mode", t["mode"].value_counts().to_dict(),
      "| by library", t.ingest_lib.value_counts().to_dict(), flush=True)
mine = t.iloc[C::N].sort_values("train_row").reset_index(drop=True)
print("this chunk", len(mine), flush=True)
# read the peaks of the chosen rows, row group by row group
pf = pq.ParquetFile(comp / "train.parquet"); want = set(mine.train_row.tolist()); peaks = {}
start = 0
for g in range(pf.num_row_groups):
    n = pf.metadata.row_group(g).num_rows
    rows = [r for r in range(start, start + n) if r in want]
    if rows:
        rg = pf.read_row_group(g, columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"])
        sub = rg.take(pa.array([r - start for r in rows])).to_pydict()
        for r, mz, it, pr in zip(rows, sub["ms2_mzs"], sub["ms2_normalized_intensities"], sub["precursor_mz"]):
            peaks[r] = (mz, it, pr)
    start += n
print("peaks read", len(peaks), f"{time.time() - T0:.0f}s", flush=True)
model = dm.load_model(str(D / "embedding_model_state_dict.pt"), device="cpu")
embs, done = [], 0
B = 64
with torch.inference_mode():
    for a in range(0, len(mine), B):
        if time.time() - T0 > LIMIT_S:
            print("time limit: stopped at", a, flush=True); break
        rr = mine.train_row.iloc[a:a + B].tolist()
        x = np.stack([dm.preprocess_spectrum(np.asarray(peaks[r][0], np.float32), np.asarray(peaks[r][1], np.float32), float(peaks[r][2])) for r in rr])
        e = np.asarray(dm.embed_preprocessed(model, x), np.float32)
        embs.append((e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)).astype(np.float16))
        done += len(rr)
        if (a // B) % 100 == 0:
            print(f"  embedded {done}/{len(mine)} {time.time() - T0:.0f}s", flush=True)
E = np.concatenate(embs) if embs else np.zeros((0, 1024), np.float16)
mine = mine.iloc[:len(E)]
np.save(f"/kaggle/working/train_refs_{C}_embeddings.npy", E)
mine[["normalized_smiles", "inchikey14", "ingest_lib", "adduct", "ionization_mode", "train_row"]].to_parquet(f"/kaggle/working/train_refs_{C}_meta.parquet", index=False)
print("saved", E.shape, f"{time.time() - T0:.0f}s", flush=True)
'''


def main() -> None:
    out, c, n = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    script = TEMPLATE.replace("__C__", str(c)).replace("__N__", str(n))
    ast.parse(script)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dreams_train_refs.py").write_text(script)
    meta = {"id": f"adityachauhanx/casmi26-dreams-train-refs-{c}", "title": f"casmi26 dreams train refs {c}",
            "code_file": "dreams_train_refs.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["hengck23/hengck23-dreams-enveda-casmi26", "metric/rdkit-2026-3-3-wheel"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "dreams_train_refs.py", len(script), "bytes")


if __name__ == "__main__":
    main()
