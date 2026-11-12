"""Molecular networking across the molecules of one test set.

Two molecules of the same extract series often differ by one modification (a sugar, a methyl, an oxygen) and
then share most fragments, either at the same m/z or shifted by the precursor difference. The modified cosine
picks up both. The graph built here is used to let confident molecules inform the candidate order of their
neighbours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .adducts import neutral_mass


@dataclass
class Spec:
    molecule: str
    adduct: str
    mode: str
    precursor: float
    mz: np.ndarray
    w: np.ndarray  # unit-norm sqrt intensities


def prepare(mz, inten, precursor, top_n: int = 60, min_rel: float = 0.01, below: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Peaks below the precursor, >= min_rel of the base peak, top_n by intensity, sqrt weights of unit norm."""
    mz = np.asarray(mz, dtype=np.float64)
    it = np.asarray(inten, dtype=np.float64)
    keep = (mz < precursor - below) & np.isfinite(it) & (it > 0)
    mz, it = mz[keep], it[keep]
    if len(it) == 0:
        return mz, it
    keep = it >= min_rel * it.max()
    mz, it = mz[keep], it[keep]
    if len(it) > top_n:
        idx = np.argpartition(-it, top_n)[:top_n]
        mz, it = mz[idx], it[idx]
    order = np.argsort(mz)
    mz, it = mz[order], np.sqrt(it[order])
    return mz, it / np.linalg.norm(it)


def modified_cosine(a: Spec, b: Spec, tol: float = 0.01) -> tuple[float, int]:
    """Greedy one-to-one matching of peaks at equal m/z or shifted by the precursor difference."""
    if len(a.mz) == 0 or len(b.mz) == 0:
        return 0.0, 0
    shift = a.precursor - b.precursor
    d0 = np.abs(a.mz[:, None] - b.mz[None, :]) <= tol
    d1 = np.abs(a.mz[:, None] - (b.mz[None, :] + shift)) <= tol if abs(shift) > tol else np.zeros_like(d0)
    ii, jj = np.nonzero(d0 | d1)
    if len(ii) == 0:
        return 0.0, 0
    prod = a.w[ii] * b.w[jj]
    order = np.argsort(-prod, kind="stable")
    used_a, used_b, total, n = set(), set(), 0.0, 0
    for k in order:
        i, j = ii[k], jj[k]
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); total += prod[k]; n += 1
    return float(total), n


def build_spectra(frame: pd.DataFrame, molecule_col: str = "molecule_id") -> list[Spec]:
    out = []
    for r in frame.itertuples(index=False):
        mz, w = prepare(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
        out.append(Spec(str(getattr(r, molecule_col)), r.adduct, r.ionization_mode, float(r.precursor_mz), mz, w))
    return out


def molecule_masses(frame: pd.DataFrame, molecule_col: str = "molecule_id") -> pd.Series:
    """Median neutral mass per molecule from all of its spectra."""
    m = [neutral_mass(p, a) for p, a in zip(frame.precursor_mz, frame.adduct)]
    return pd.Series(m, index=frame.index).groupby(frame[molecule_col].astype(str)).median()


def molecule_graph(spectra: list[Spec], min_matches: int = 4) -> pd.DataFrame:
    """Best modified cosine per molecule pair over spectra with the same adduct."""
    by = {}
    for s in spectra:
        by.setdefault(s.adduct, []).append(s)
    best = {}
    for group in by.values():
        for x in range(len(group)):
            for y in range(x + 1, len(group)):
                a, b = group[x], group[y]
                if a.molecule == b.molecule:
                    continue
                score, n = modified_cosine(a, b)
                if n < min_matches:
                    continue
                key = (a.molecule, b.molecule) if a.molecule < b.molecule else (b.molecule, a.molecule)
                if score > best.get(key, (0.0, 0))[0]:
                    best[key] = (score, n)
    rows = [(k[0], k[1], v[0], v[1]) for k, v in best.items()]
    return pd.DataFrame(rows, columns=["a", "b", "cosine", "matches"])
