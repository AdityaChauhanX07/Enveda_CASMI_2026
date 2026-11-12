"""Sparse matrix of the FPNet's 10,226 selected fingerprint bits for every panel PubChem candidate, so that any
model's logits score all 1.94 M rows in seconds (score = sum of the logits of the row's bits)."""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casmi.panel import PANEL  # noqa: E402

OUT = ROOT / "artifacts" / "panel_fp"


def _chunk(rows):
    from casmi import fpscore
    from casmi.panel import tier_smiles
    out = []
    for s in tier_smiles(rows):
        f = fpscore.fingerprint(s) if s else None
        out.append(None if f is None else np.flatnonzero(f).astype(np.uint16))
    return out


def build(workers: int = 4) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = np.unique(pd.read_parquet(PANEL / "panel_fpnet_scores.parquet", columns=["tier_row"]).tier_row.to_numpy())
    chunks = np.array_split(rows, max(1, len(rows) // 20000))
    t0, parts = time.time(), []
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap(_chunk, chunks)):
            parts.extend(res)
            if i % 10 == 0:
                print(f"{len(parts):,}/{len(rows):,} rows {time.time() - t0:.0f}s", flush=True)
    valid = np.array([p is not None for p in parts])
    lens = np.array([0 if p is None else len(p) for p in parts], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(lens)])
    indices = np.concatenate([p for p in parts if p is not None]).astype(np.uint16)
    np.save(OUT / "rows.npy", rows); np.save(OUT / "valid.npy", valid)
    np.save(OUT / "indptr.npy", indptr); np.save(OUT / "indices.npy", indices)
    print(f"done {len(rows):,} rows, {valid.mean():.4f} valid, {len(indices):,} bits, {time.time() - t0:.0f}s")


class PanelFP:
    """Score panel candidate rows with per-molecule logits."""

    def __init__(self, root: Path = OUT):
        self.rows = np.load(root / "rows.npy"); self.valid = np.load(root / "valid.npy")
        self.indptr = np.load(root / "indptr.npy"); self.indices = np.load(root / "indices.npy", mmap_mode="r")

    def score(self, tier_rows: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """Score of each tier row under one logits vector (NaN for unparsable structures)."""
        pos = np.searchsorted(self.rows, tier_rows)
        assert (self.rows[pos] == tier_rows).all()
        a, b = self.indptr[pos], self.indptr[pos + 1]
        n = b - a
        idx = np.concatenate([np.arange(x, y) for x, y in zip(a, b)]) if len(a) else np.zeros(0, np.int64)
        vals = logits.astype(np.float64)[np.asarray(self.indices[idx], dtype=np.int64)]
        seg = np.repeat(np.arange(len(pos)), n)
        out = np.bincount(seg, weights=vals, minlength=len(pos))
        return np.where(self.valid[pos], out, np.nan)


def score_frame(frame: pd.DataFrame, logits_by_molecule: dict, fp: PanelFP | None = None) -> np.ndarray:
    """Score every candidate row of `frame` (molecule, tier_row) with its molecule's logits."""
    fp = fp or PanelFP()
    out = np.full(len(frame), np.nan)
    for m, idx in frame.groupby("molecule", sort=False).indices.items():
        z = logits_by_molecule.get(m)
        if z is not None:
            out[idx] = fp.score(frame.tier_row.to_numpy()[idx], z)
    return out


if __name__ == "__main__":
    if "--check" not in sys.argv:
        build()
    from casmi.panel import load_candidates
    c = load_candidates()
    zl = np.load(PANEL / "panel_logits.npz")
    logits = {str(k): zl["fold0"][i] for i, k in enumerate(zl["keys"])}
    s = score_frame(c, logits)
    ok = np.isfinite(s) & np.isfinite(c.fold0_full.to_numpy())
    print("rows compared", int(ok.sum()), "max |diff|", float(np.abs(s[ok] - c.fold0_full.to_numpy()[ok]).max()),
          "nan mismatch", int((np.isfinite(s) != np.isfinite(c.fold0_full.to_numpy())).sum()))
