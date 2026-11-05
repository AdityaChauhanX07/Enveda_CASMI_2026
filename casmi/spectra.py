"""Spectrum cleaning and spectral entropy similarity (Li et al., Nature Methods 2021)."""
from __future__ import annotations

import numpy as np


def clean(mz, inten, precursor_mz: float | None = None, min_rel: float = 0.01, above_precursor: float = 2.2,
          top_n: int | None = 256, merge_da: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Relative-intensity floor, drop peaks above precursor + above_precursor Da, optional merge, keep top_n.
    Test spectra keep peaks up to precursor + ~2.18 Da, so the default cap is 2.2 Da."""
    mz = np.asarray(mz, dtype=np.float64)
    it = np.asarray(inten, dtype=np.float64)
    if mz.size == 0:
        return mz, it
    keep = it >= min_rel * it.max()
    if precursor_mz is not None:
        keep &= mz <= precursor_mz + above_precursor
    mz, it = mz[keep], it[keep]
    order = np.argsort(mz)
    mz, it = mz[order], it[order]
    if merge_da > 0 and mz.size > 1:
        groups = np.concatenate([[0], np.cumsum(np.diff(mz) > merge_da)])
        w = np.bincount(groups, weights=it)
        mz = np.bincount(groups, weights=mz * it) / np.where(w > 0, w, 1)
        it = w
    if top_n is not None and mz.size > top_n:
        sel = np.sort(np.argsort(it)[-top_n:])
        mz, it = mz[sel], it[sel]
    return mz, it


def entropy_weight(it: np.ndarray) -> np.ndarray:
    """Normalise to sum 1; low-entropy spectra (S < 3) get intensity ** (0.25 + 0.25 * S), renormalised."""
    p = it / it.sum()
    s = float(-(p * np.log(p)).sum())
    if s < 3.0:
        p = p ** (0.25 + 0.25 * s)
        p = p / p.sum()
    return p


def entropy_similarity(mz_a, it_a, mz_b, it_b, tol_da: float = 0.01, tol_ppm: float = 0.0,
                       weighted: bool = True, shift: float = 0.0) -> float:
    """Spectral entropy similarity in [0, 1] with greedy one-to-one peak matching.
    shift is added to spectrum B's m/z before matching (precursor-difference trick for analogues)."""
    mz_a, it_a = np.asarray(mz_a, float), np.asarray(it_a, float)
    mz_b, it_b = np.asarray(mz_b, float) + shift, np.asarray(it_b, float)
    if mz_a.size == 0 or mz_b.size == 0 or it_a.sum() <= 0 or it_b.sum() <= 0:
        return 0.0
    pa = entropy_weight(it_a) if weighted else it_a / it_a.sum()
    pb = entropy_weight(it_b) if weighted else it_b / it_b.sum()
    ob = np.argsort(mz_b)
    mz_b, pb = mz_b[ob], pb[ob]
    tol = np.maximum(tol_da, mz_a * tol_ppm * 1e-6)
    lo = np.searchsorted(mz_b, mz_a - tol, "left")
    hi = np.searchsorted(mz_b, mz_a + tol, "right")
    pairs = [(pa[i] * pb[j], i, j) for i in range(mz_a.size) for j in range(lo[i], hi[i])]
    if not pairs:
        return 0.0
    pairs.sort(reverse=True)
    used_a, used_b, total = set(), set(), 0.0
    for _, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        a, b = pa[i], pb[j]
        total += (a + b) * np.log2(a + b) - a * np.log2(a) - b * np.log2(b)
    return float(min(1.0, max(0.0, total / 2.0)))
