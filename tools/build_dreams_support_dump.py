"""Kaggle CPU job: DreaMS analogue support for every (panel molecule, candidate) pair of several final-list files.

Support depends only on the molecule's spectra and the candidate, not on the list it sits in, so one job scores the
union of the candidates of every list file given, and any re-ranking rule (weight, leader guard, gate) can then be
evaluated offline (experiments/dreams_rules.py). Same model and support as casmi/dreams_overlay.py: panel spectra
(enveda-np-examples) embedded with the public DreaMS model, MassSpecGym neighbours by cosine, support = max over the
k nearest of cosine^p x Tanimoto (Morgan radius 2, 2,048 bits), averaged over the molecule's spectra. Two views per
pair: `hidden` (the answer's own structure removed from the neighbours, as for class-2 / class-3 answers with no
public spectra) and `visible` (nothing removed, as for class-1 answers whose spectra are public).

Output (the job's output files): dreams_support.json.gz = {molecule: {smiles: [hidden, visible]}};
dreams_neighbours.json.gz = per panel spectrum, its molecule, answer key and K + 40 nearest references
[index, cosine, SMILES, InChIKey first block].

usage: python tools/build_dreams_support_dump.py OUT_DIR LISTS1.csv[,LISTS2.csv,...] [K] [P]
"""
from __future__ import annotations

import ast
import base64
import json
import sys
import zlib
from pathlib import Path

TEMPLATE = r'''# DreaMS analogue support for every (panel molecule, candidate) pair (CPU)
import base64, glob, gzip, json, pickle, subprocess, sys, time, zlib
from pathlib import Path
import numpy as np, pandas as pd
_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
_whl = [w for w in glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True) if _tag in w and "2026.3.3" in w]
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", _whl[0]], check=True)
import torch
torch.set_num_threads(4)
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")
T0 = time.time()
CANDS = json.loads(zlib.decompress(base64.b64decode(__CANDS__)).decode())     # molecule -> candidate SMILES
K, P = __K__, __P__
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
q["m"] = [key(s) for s in q.normalized_smiles]
q["ik14"] = [Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for s in q.normalized_smiles]
x = np.stack([dm.preprocess_spectrum(np.asarray(r.ms2_mzs, np.float32), np.asarray(r.ms2_normalized_intensities, np.float32),
                                     float(r.precursor_mz)) for r in q.itertuples()])
with torch.inference_mode():
    Q = np.concatenate([np.asarray(dm.embed_preprocessed(model, x[i:i + 64])) for i in range(0, len(x), 64)])
Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
print("panel spectra embedded:", Q.shape, f"{time.time() - T0:.0f}s", flush=True)
ref = np.load(D / "MassSpecGym_DreaMS_embeddings.npy", mmap_mode="r")
meta = pickle.load(open(D / "MassSpecGym_DreaMS_meta.pkl", "rb"))
ref_smiles = np.asarray(meta["normalized_smiles"] if "normalized_smiles" in meta else meta["smiles"], dtype=object)
KK = K + 40
best_s = np.full((len(Q), KK), -2.0, np.float32); best_i = np.full((len(Q), KK), -1, np.int64)
qt = torch.from_numpy(Q.astype(np.float32))
for a in range(0, ref.shape[0], 200000):
    r = torch.from_numpy(np.asarray(ref[a:a + 200000], np.float32))
    r = r / r.norm(dim=1, keepdim=True).clamp(min=1e-9)
    s = (qt @ r.T).numpy()
    top = np.argpartition(-s, KK - 1, axis=1)[:, :KK]
    allc = np.concatenate([best_s, np.take_along_axis(s, top, 1)], 1); alli = np.concatenate([best_i, top + a], 1)
    o = np.argsort(-allc, axis=1)[:, :KK]
    best_s, best_i = np.take_along_axis(allc, o, 1), np.take_along_axis(alli, o, 1)
print("neighbours done", f"{time.time() - T0:.0f}s", flush=True)
ikc = {}
def ref_ik(i):
    if i not in ikc:
        m_ = Chem.MolFromSmiles(str(ref_smiles[i])); ikc[i] = Chem.MolToInchiKey(m_)[:14] if m_ is not None else ""
    return ikc[i]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fpc = {}
def fp(s_):
    if s_ not in fpc:
        m_ = Chem.MolFromSmiles(str(s_)); fpc[s_] = gen.GetFingerprint(m_) if m_ is not None else None
    return fpc[s_]
rows_of = {}
for j, m in enumerate(q.m):
    rows_of.setdefault(m, []).append(j)
out, n_pairs = {}, 0
for m, cands in CANDS.items():
    js = rows_of.get(m, [])
    if not js: continue
    cfp = [fp(s_) for s_ in cands]
    sup = np.zeros((len(cands), 2))
    for j in js:
        allnb = [(int(i), float(c)) for i, c in zip(best_i[j], best_s[j]) if i >= 0]
        views = ([(i, c) for i, c in allnb if ref_ik(i) != q.ik14.iloc[j]][:K], allnb[:K])
        for v, nb in enumerate(views):
            nb = [(fp(ref_smiles[i]), c) for i, c in nb]
            nb = [(f, c) for f, c in nb if f is not None]
            if not nb: continue
            fs, w = [f for f, _ in nb], np.array([max(c, 0.0) ** P for _, c in nb])
            for t, f in enumerate(cfp):
                if f is not None:
                    sup[t, v] += float((w * np.array(DataStructs.BulkTanimotoSimilarity(f, fs))).max())
    sup /= len(js)
    out[m] = {s_: [round(float(a), 5), round(float(b), 5)] for s_, (a, b) in zip(cands, sup)}
    n_pairs += len(cands)
Path("/kaggle/working/dreams_support.json.gz").write_bytes(gzip.compress(json.dumps(out).encode()))
# raw neighbours too (offline checks of p, k and the aggregation): per panel spectrum, the KK nearest references
nbd = [dict(m=q.m.iloc[j], ik14=q.ik14.iloc[j],
            nb=[[int(i), round(float(c), 5), str(ref_smiles[i]), ref_ik(int(i))] for i, c in zip(best_i[j], best_s[j]) if i >= 0])
       for j in range(len(Q))]
Path("/kaggle/working/dreams_neighbours.json.gz").write_bytes(gzip.compress(json.dumps(nbd).encode()))
print("molecules", len(out), "pairs", n_pairs, f"{time.time() - T0:.0f}s", flush=True)
'''


def main() -> None:
    out = Path(sys.argv[1])
    files = [Path(f) for f in sys.argv[2].split(",")]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    p = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    import pandas as pd
    cands: dict[str, list[str]] = {}
    for f in files:                                   # molecule_id = the answer's metric key (real-pipeline panel lists)
        sub = pd.read_csv(f, dtype=str, keep_default_na=False)
        for m, smi in zip(sub.molecule_id, sub.smiles):
            seen = cands.setdefault(m, [])
            seen.extend(s_ for s_ in smi.split(";") if s_ and s_ not in seen)
    payload = base64.b64encode(zlib.compress(json.dumps(cands).encode(), 9)).decode()
    script = TEMPLATE.replace("__CANDS__", repr(payload)).replace("__K__", repr(k)).replace("__P__", repr(p))
    ast.parse(script)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dreams_support_dump.py").write_text(script)
    meta = {"id": "adityachauhanx/casmi26-dreams-support-dump", "title": "casmi26 dreams support dump",
            "code_file": "dreams_support_dump.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["hengck23/hengck23-dreams-enveda-casmi26", "metric/rdkit-2026-3-3-wheel"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "dreams_support_dump.py", len(script), "bytes;", len(cands), "molecules,",
          sum(len(v) for v in cands.values()), "pairs")


if __name__ == "__main__":
    main()
