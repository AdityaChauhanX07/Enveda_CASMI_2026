"""Local end-to-end check of casmi/dreams_overlay.apply() on the panel (no torch needed).

The DreaMS model is replaced by the panel embeddings saved by casmi26-dreams-panel-embed (same model, same spectra,
same order), and the torch kNN by a numpy one over the downloaded MassSpecGym embeddings; everything else - own-spectrum
exclusion, library-hit exemptions, protected leads, damped z, positive-mode spectra, reverse support - runs exactly as
in a notebook. The answer's structure is hidden from the references (class-2 situation), so the result should match the
offline experiments (experiments/dreams_local.py, dreams_reverse.py).

usage: dreams_module_check.py MSG_DIR PANEL_EMBED_DIR LISTS.csv DUMP.pkl [--key=value overlay parameters]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi import dreams_overlay as do  # noqa: E402
from dreams_reverse import SafeUnpickler  # noqa: E402


def main() -> None:
    msg_dir, emb_dir, lists_path, dump_path = (Path(a) for a in sys.argv[1:5])
    params = {}
    for a in sys.argv[5:]:
        k, v = a[2:].split("=", 1)
        params[k] = float(v)
    rows = pd.read_parquet(emb_dir / "panel_rows.parquet")
    Q = np.load(emb_dir / "panel_embeddings.npy").astype(np.float32)
    Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
    work = ROOT / "artifacts" / "dreams" / "module_check"        # gitignored
    work.mkdir(parents=True, exist_ok=True)
    test = pd.DataFrame({"molecule_id": rows.m, "ionization_mode": np.where(rows["mode"] == "pos", "positive", "negative")})
    test.to_parquet(work / "panel_test.parquet", index=False)
    # model -> saved embeddings; torch kNN -> numpy
    do.embed = lambda t, d, b: Q[: len(t)]

    def knn(q, ref, k, chunk):
        best_s = np.full((len(q), k), -2.0, np.float32); best_i = np.full((len(q), k), -1, np.int64)
        for a in range(0, ref.shape[0], chunk):
            r = np.array(ref[a:a + chunk], np.float32); r /= np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
            s = q @ r.T; kk = min(k, s.shape[1]); top = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
            allc = np.concatenate([best_s, np.take_along_axis(s, top, 1)], 1); alli = np.concatenate([best_i, top + a], 1)
            o = np.argsort(-allc, axis=1)[:, :k]
            best_s, best_i = np.take_along_axis(allc, o, 1), np.take_along_axis(alli, o, 1)
        return best_i, best_s
    do.knn = knn
    sub = pd.read_csv(lists_path, dtype=str, keep_default_na=False)
    hits = pickle.load(open(dump_path, "rb"))["hits"]     # our own dump
    real_load = pickle.load
    do.pickle.load = lambda f: SafeUnpickler(f).load()    # the downloaded metadata: numpy only
    keep = {m: [s_ for c, s_ in hits.get(m, {}).values() if c >= 0.95] for m in sub.molecule_id}
    keep = {m: v for m, v in keep.items() if v}
    selfok = {m: [s_ for c, s_ in hits.get(m, {}).values() if c >= 0.8] for m in sub.molecule_id}
    hide = {m: {ik} for m, ik in zip(rows.m, rows.ik14)}
    sub.to_csv(work / "lists.csv", index=False)
    p = dict(lam=1.5, noself=1, self_lib=0.8, z_eps=0.01, pos_only=1, k=50, max_seconds=1e9)
    p.update(params)
    rep = do.apply(work / "panel_test.parquet", work / "lists.csv", msg_dir, work / "lists_out.csv", None, p,
                   hide=hide, keep=keep, selfok=selfok)
    do.pickle.load = real_load
    print("report", {k: v for k, v in rep.items() if k != "error_samples"}, rep.get("error_samples"))
    out = pd.read_csv(work / "lists_out.csv", dtype=str, keep_default_na=False)
    import dreams_local as dl
    keys = dl._key_cache([s_ for d in (sub, out) for smi in d.smiles for s_ in smi.split(";")], dl.metric_key)

    def mrr(df):                                       # molecule_id = the answer's metric key
        tot = []
        for m, smi in zip(df.molecule_id, df.smiles):
            ks = [keys[s_] for s_ in smi.split(";")[:25]]
            tot.append(1.0 / (ks.index(m) + 1) if m in ks else 0.0)
        return float(np.mean(tot))
    print(f"MRR before {mrr(sub):.4f} | after {mrr(out):.4f} | params {p}")


if __name__ == "__main__":
    main()
