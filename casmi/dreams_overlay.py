"""DreaMS analogue support: re-rank each final list by the structures of the query's nearest public spectra.

Each test spectrum is embedded with the public DreaMS model (minimal inference code and weights in
hengck23/hengck23-dreams-enveda-casmi26) and its k nearest MassSpecGym spectra (public libraries; embeddings in the same
dataset) are retrieved by cosine. A candidate's support is the maximum over those neighbours of cosine^p x Tanimoto
(Morgan radius 2, 2,048 bits) between the candidate and the neighbour's structure, averaged over the molecule's
spectra. Each list is re-ordered by 1/(3 + rank) / sd + lam x z(support) (z within the list), the form measured on the
panel (experiments: tools/build_dreams_panel.py; documentation-free C1e lists 0.588 -> 0.614 at lam 1). Lists keep their
candidates; a failure keeps the input file. keep (molecule -> SMILES): leads that stay on top while they still head the
list, e.g. C1's strong library matches (lib_first_leads.json), whose answers are usually right already. extra_refs: a
glob of further reference embeddings (*_embeddings.npy beside *_meta.parquet with normalized_smiles), e.g. the
train-library spectra embedded by tools/build_dreams_train_refs.py; neighbours are the k nearest over all references.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = dict(lam=1.0, k=50, p=4.0, batch=64, chunk=100000, max_seconds=2400, reference="MassSpecGym", extra_refs="")


def _model(dreams_dir: Path):
    sys.path.insert(0, str(dreams_dir))
    import dreams_minimal_model as dm
    return dm, dm.load_model(str(dreams_dir / "embedding_model_state_dict.pt"), device="cpu")


def embed(test: pd.DataFrame, dreams_dir: Path, batch: int) -> np.ndarray:
    import torch
    dm, model = _model(dreams_dir)
    x = np.stack([dm.preprocess_spectrum(np.asarray(r.ms2_mzs, np.float32), np.asarray(r.ms2_normalized_intensities, np.float32),
                                         float(r.precursor_mz)) for r in test.itertuples()])
    with torch.inference_mode():
        q = np.concatenate([np.asarray(dm.embed_preprocessed(model, x[i:i + batch])) for i in range(0, len(x), batch)])
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)


def knn(q: np.ndarray, ref: np.ndarray, k: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    best_s = np.full((len(q), k), -2.0, np.float32); best_i = np.full((len(q), k), -1, np.int64)
    qt = torch.from_numpy(q.astype(np.float32))
    for a in range(0, ref.shape[0], chunk):
        r = torch.from_numpy(np.array(ref[a:a + chunk], np.float32))
        r = r / r.norm(dim=1, keepdim=True).clamp(min=1e-9)
        s = (qt @ r.T).numpy()
        kk = min(k, s.shape[1])
        top = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
        allc = np.concatenate([best_s, np.take_along_axis(s, top, 1)], 1)
        alli = np.concatenate([best_i, top + a], 1)
        o = np.argsort(-allc, axis=1)[:, :k]
        best_s, best_i = np.take_along_axis(allc, o, 1), np.take_along_axis(alli, o, 1)
    return best_i, best_s


def apply(test_path, submission_path, dreams_dir, output_path, report_path=None, p: dict | None = None,
          hide: dict | None = None, keep: dict | None = None) -> dict:
    """hide (panel checks only): molecule -> standard InChIKey first blocks whose reference spectra are skipped.
    keep: molecule -> SMILES kept in place while they form the head of the list."""
    p = {**PARAMS, **(p or {})}
    t0 = time.time()
    report = dict(molecules=0, reordered=0, rank1_changed=0, kept_leads=0, errors=0, error_samples=[])
    sub = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        RDLogger.DisableLog("rdApp.*")
        dreams_dir = Path(dreams_dir)
        test = pd.read_parquet(test_path).reset_index(drop=True)
        q = embed(test, dreams_dir, p["batch"])
        report["embed_seconds"] = round(time.time() - t0, 1)
        refs = [np.load(dreams_dir / f"{p['reference']}_DreaMS_embeddings.npy", mmap_mode="r")]
        meta = pickle.load(open(dreams_dir / f"{p['reference']}_DreaMS_meta.pkl", "rb"))
        smiles = [np.asarray(meta["normalized_smiles"] if "normalized_smiles" in meta else meta["smiles"], dtype=object)]
        del meta
        for f in (sorted(glob.glob(p["extra_refs"], recursive=True)) if p["extra_refs"] else []):
            try:                                        # extra references: *_embeddings.npy beside *_meta.parquet
                smiles.append(pd.read_parquet(f.replace("_embeddings.npy", "_meta.parquet"),
                                              columns=["normalized_smiles"]).normalized_smiles.to_numpy(dtype=object))
                refs.append(np.load(f, mmap_mode="r"))
                assert refs[-1].shape[0] == len(smiles[-1]), f
            except Exception as e:
                smiles = smiles[:len(refs)]
                report["extra_ref_error"] = f"{type(e).__name__}: {str(e)[:100]}"
        nn_i = nn_s = None; off = 0
        for r in refs:                                  # nearest k over all references, global row numbers
            i_, s_ = knn(q, r, p["k"], p["chunk"])
            i_ = np.where(i_ >= 0, i_ + off, -1)
            if nn_i is None:
                nn_i, nn_s = i_, s_
            else:
                allc, alli = np.concatenate([nn_s, s_], 1), np.concatenate([nn_i, i_], 1)
                o = np.argsort(-allc, axis=1)[:, :p["k"]]
                nn_s, nn_i = np.take_along_axis(allc, o, 1), np.take_along_axis(alli, o, 1)
            off += r.shape[0]
        ref_smiles = np.concatenate(smiles)
        report["references"] = int(off)
        report["knn_seconds"] = round(time.time() - t0, 1)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fpc = {}

        def fp(s_):
            if s_ not in fpc:
                m_ = Chem.MolFromSmiles(str(s_)); fpc[s_] = gen.GetFingerprint(m_) if m_ is not None else None
            return fpc[s_]
        rows_of = {str(m): g.index.to_numpy() for m, g in test.groupby("molecule_id", sort=False)}
        ikc = {}

        def ref_ik(i):                                  # only for panel checks with hide
            if i not in ikc:
                m_ = Chem.MolFromSmiles(str(ref_smiles[i])); ikc[i] = Chem.MolToInchiKey(m_)[:14] if m_ is not None else ""
            return ikc[i]
        out = []
        for m, smi in zip(sub.molecule_id, sub.smiles):
            cand = smi.split(";")
            try:
                if time.time() - t0 > p["max_seconds"] or len(cand) < 2 or m not in rows_of:
                    out.append(smi); continue
                cfp = [fp(s_) for s_ in cand]
                sup = np.zeros(len(cand))
                for j in rows_of[m]:
                    nb = [(fp(ref_smiles[i]), c) for i, c in zip(nn_i[j], nn_s[j])
                          if i >= 0 and not (hide and ref_ik(int(i)) in hide.get(m, ()))]
                    nb = [(f, c) for f, c in nb if f is not None]
                    if not nb:
                        continue
                    fs, w = [f for f, _ in nb], np.array([max(c, 0.0) ** p["p"] for _, c in nb])
                    for t, f in enumerate(cfp):
                        if f is not None:
                            sup[t] += float((w * np.array(DataStructs.BulkTanimotoSimilarity(f, fs))).max())
                sup /= max(len(rows_of[m]), 1)
                base = 1.0 / (3.0 + np.arange(1, len(cand) + 1))
                z = (sup - sup.mean()) / (sup.std() + 1e-9)
                order = np.argsort(-(base / base.std() + p["lam"] * z), kind="stable")
                new = [cand[i] for i in order]
                ks = set((keep or {}).get(m, ()))
                n_head = next((i for i, c in enumerate(cand) if c not in ks), len(cand)) if ks else 0
                if n_head:                              # protected leads stay on top, the rest re-ranked below them
                    new = cand[:n_head] + [c for c in new if c not in set(cand[:n_head])]
                    report["kept_leads"] += 1
                report["molecules"] += 1
                report["reordered"] += int(new != cand); report["rank1_changed"] += int(new[0] != cand[0])
                out.append(";".join(new))
            except Exception as e:                      # this molecule keeps its list
                report["errors"] += 1
                if len(report["error_samples"]) < 3:
                    report["error_samples"].append(f"{m}: {type(e).__name__}: {str(e)[:100]}")
                out.append(smi)
        res = pd.DataFrame({"molecule_id": sub.molecule_id, "smiles": out})
        assert res.molecule_id.tolist() == sub.molecule_id.tolist()
        assert all(sorted(a.split(";")) == sorted(b.split(";")) for a, b in zip(res.smiles, sub.smiles)), "candidate sets changed"
        res.to_csv(output_path, index=False)
    except Exception as e:                              # without DreaMS the lists stay as they are
        report["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        sub.to_csv(output_path, index=False)
    report["seconds"] = round(time.time() - t0, 1)
    report["error_samples"] = report["error_samples"] or None
    if report_path:
        Path(report_path).write_text(json.dumps(dict(report, params=p), indent=2))
    return report
