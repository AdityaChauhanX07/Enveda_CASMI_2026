"""Kaggle CPU job: does DreaMS analogue support separate the answer from its isomers? (panel, class 2)

Each panel spectrum (enveda-np-examples) is embedded with DreaMS (hengck23/hengck23-dreams-enveda-casmi26: the public
DreaMS model, minimal inference code, and precomputed embeddings of the Enveda-180 library and MassSpecGym). Its k
nearest reference spectra (cosine) are retrieved with the answer's own structure hidden (inchikey14), and every
candidate of C1's class-2 list (25, from the spec2mol panel job's payload) gets analogue support = max over neighbours of
cosine^p x Tanimoto(candidate, neighbour structure), averaged over the molecule's spectra. Reported per reference set:
how often the answer has the highest support among its list, and C1's order re-ranked with + lambda x z(support).

usage: python tools/build_dreams_panel.py OUT_DIR (SPEC2MOL_PANEL_SCRIPT | FINAL_LISTS.csv) [Enveda180,MassSpecGym] [LAMBDAS, 9 = support only]
"""
from __future__ import annotations

import ast
import base64
import json
import sys
import zlib
from pathlib import Path

TEMPLATE = r'''# DreaMS analogue support on the panel (CPU)
import base64, glob, json, pickle, subprocess, sys, time, zlib
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
LISTS = json.loads(zlib.decompress(base64.b64decode(__LISTS__)).decode())
D = Path(sorted(glob.glob("/kaggle/input/**/Enveda180_DreaMS_index.parquet", recursive=True), key=len)[0]).parent
sys.path.insert(0, str(D))
import dreams_minimal_model as dm
model = dm.load_model(str(D / "embedding_model_state_dict.pt"), device="cpu")
te = rdMolStandardize.TautomerEnumerator()
def key(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return ""
    Chem.RemoveStereochemistry(m)
    return Chem.MolToInchiKey(te.Canonicalize(m))[:14]
# panel spectra
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
K = 50
def knn(ref, hide_ik):
    """top-K (index, cosine) per query over a memory-mapped reference, the answer's own structure hidden."""
    best_s = np.full((len(Q), K + 40), -2.0, np.float32); best_i = np.full((len(Q), K + 40), -1, np.int64)
    qt = torch.from_numpy(Q.astype(np.float32))
    for a in range(0, ref.shape[0], 200000):
        r = torch.from_numpy(np.asarray(ref[a:a + 200000], np.float32))
        r = r / r.norm(dim=1, keepdim=True).clamp(min=1e-9)
        s = (qt @ r.T).numpy()
        top = np.argpartition(-s, K + 39, axis=1)[:, :K + 40]
        cs = np.take_along_axis(s, top, axis=1)
        allc = np.concatenate([best_s, cs], 1); alli = np.concatenate([best_i, top + a], 1)
        o = np.argsort(-allc, axis=1)[:, :K + 40]
        best_s = np.take_along_axis(allc, o, 1); best_i = np.take_along_axis(alli, o, 1)
    out = []
    for j in range(len(Q)):
        keep = [(int(i), float(c)) for i, c in zip(best_i[j], best_s[j]) if i >= 0 and hide_ik[i] != q.ik14.iloc[j]]
        out.append(keep[:K])
    return out
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fpc = {}
def fp(s):
    if s not in fpc:
        m = Chem.MolFromSmiles(s); fpc[s] = gen.GetFingerprint(m) if m is not None else None
    return fpc[s]
def evaluate(name, nn, ref_smiles, p=4.0):
    by_m = {}
    for j, m in enumerate(q.m):
        by_m.setdefault(m, []).append(j)
    rows = {}
    for m, cand in LISTS.items():
        js = by_m.get(m, [])
        if not js: continue
        cfp = [fp(s) for _, s in cand]
        sup = np.zeros(len(cand))
        for j in js:
            nfp = [(fp(ref_smiles[i]), c) for i, c in nn[j]]
            nfp = [(f, c) for f, c in nfp if f is not None]
            if not nfp: continue
            fs = [f for f, _ in nfp]; w = np.array([max(c, 0.0) ** p for _, c in nfp])
            for t, f in enumerate(cfp):
                if f is None: continue
                sim = np.array(DataStructs.BulkTanimotoSimilarity(f, fs))
                sup[t] += float((w * sim).max())
        rows[m] = (sup / len(js), [k for k, _ in cand])
    best = np.mean([sup.argmax() == ks.index(m) for m, (sup, ks) in rows.items() if m in ks])
    print(f"{name}: molecules {len(rows)}; answer has the highest support in its list: {best:.3f}", flush=True)
    out = []
    for lam in __LAMBDAS__:
        rr = 0.0
        for m, (sup, ks) in rows.items():
            base = 1.0 / (3.0 + np.arange(1, len(ks) + 1))
            z = (sup - sup.mean()) / (sup.std() + 1e-9)
            score = (z if lam > 8 else base / base.std() + lam * z)
            order = [ks[i] for i in np.argsort(-score, kind="stable")]
            rr += 1.0 / (order.index(m) + 1) if m in order[:25] else 0.0
        out.append(f"{'support only' if lam > 8 else lam}: {rr / len(rows):.4f}")
    print(f"  C1 order + lambda x z(support) -> MRR | " + " | ".join(out), flush=True)
for ref_name in __REFS__:
    try:
        emb = np.load(D / f"{ref_name}_DreaMS_embeddings.npy", mmap_mode="r")
        meta = pickle.load(open(D / f"{ref_name}_DreaMS_meta.pkl", "rb"))
        smiles = np.asarray(meta["normalized_smiles"] if "normalized_smiles" in meta else meta["smiles"], dtype=object)
        if "inchikey14" in meta:
            ik = np.asarray(meta["inchikey14"], dtype=object)
        elif "inchikey" in meta:
            ik = np.asarray([str(v)[:14] for v in meta["inchikey"]], dtype=object)
        else:                                          # no key column: standard InChIKey first blocks from the SMILES
            print(ref_name, "meta keys:", list(meta.keys())[:20], flush=True)
            cache = {}
            def ik14(s_):
                if s_ not in cache:
                    m_ = Chem.MolFromSmiles(str(s_)); cache[s_] = Chem.MolToInchiKey(m_)[:14] if m_ is not None else ""
                return cache[s_]
            ik = np.asarray([ik14(s_) for s_ in smiles], dtype=object)
        print(ref_name, "reference", emb.shape, f"{time.time() - T0:.0f}s", flush=True)
        nn = knn(emb, ik)
        print(ref_name, "neighbours done", f"{time.time() - T0:.0f}s", "| mean top-1 cosine", round(float(np.mean([v[0][1] for v in nn if v])), 3), flush=True)
        evaluate(ref_name, nn, smiles)
    except Exception as e:
        print(ref_name, "failed:", repr(e)[:300], flush=True)
'''


def main() -> None:
    out, source = Path(sys.argv[1]), Path(sys.argv[2])
    if source.suffix == ".csv":                       # final lists of a real-pipeline run (molecule_id = the answer's key)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import pandas as pd
        from casmi.metric import ik14
        sub = pd.read_csv(source, dtype=str, keep_default_na=False)
        lists = {m: [[ik14(s_) or "", s_] for s_ in smi.split(";")] for m, smi in zip(sub.molecule_id, sub.smiles)}
    else:                                             # the spec2mol panel job's payload (C1's class-2 lists)
        src = source.read_text()
        lists_lit = next(n.value.args[0].args[0].value for n in ast.walk(ast.parse(src))
                         if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LISTS")
        lists = json.loads(zlib.decompress(base64.b64decode(lists_lit)).decode())["c2"]
    payload = base64.b64encode(zlib.compress(json.dumps(lists).encode(), 9)).decode()
    refs = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("Enveda180", "MassSpecGym")
    lams = tuple(float(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else (0.0, 0.1, 0.25, 0.5, 1.0, 9.0)
    script = TEMPLATE.replace("__LISTS__", repr(payload)).replace("__REFS__", repr(refs)).replace("__LAMBDAS__", repr(lams))
    ast.parse(script)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dreams_panel.py").write_text(script)
    meta = {"id": "adityachauhanx/casmi26-dreams-analog-panel", "title": "casmi26 dreams analog panel", "code_file": "dreams_panel.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["hengck23/hengck23-dreams-enveda-casmi26", "metric/rdkit-2026-3-3-wheel"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "dreams_panel.py", len(script), "bytes;", len(lists), "molecules")


if __name__ == "__main__":
    main()
