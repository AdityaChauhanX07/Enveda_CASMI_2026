"""Fragment explanation (MetFrag-like, no trained parameters): how much of a spectrum a candidate's bond-cleavage
fragments explain.

Every connected piece left after cutting one acyclic bond, two acyclic bonds, or two bonds of one ring (a ring
opening) is a fragment; its ion masses
are its atoms' mass with their hydrogens, plus or minus up to `h_shift` hydrogens (rearrangements and radical sites),
plus a proton (positive mode) or minus one (negative mode). A spectrum's top `top_n` peaks below the precursor are
matched within `tol` Da; the score is the share of sqrt-intensity those peaks explain.
"""
from __future__ import annotations

import numpy as np

H = 1.00782503207
PROTON = 1.007276466812


class Fragmenter:
    def __init__(self, max_cut_pairs: int = 6000, h_shift: int = 2):
        self.max_cut_pairs, self.h_shift = max_cut_pairs, h_shift
        self.cache: dict[str, np.ndarray] = {}

    def neutral_fragments(self, smiles: str) -> np.ndarray:
        """Sorted unique masses of the neutral pieces (atoms with their implicit and explicit hydrogens)."""
        if smiles in self.cache:
            return self.cache[smiles]
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() < 2:
            self.cache[smiles] = out = np.zeros(0)
            return out
        pt = Chem.GetPeriodicTable()
        n = mol.GetNumAtoms()
        am = np.array([pt.GetMostCommonIsotopeMass(a.GetSymbol()) + a.GetTotalNumHs() * H for a in mol.GetAtoms()])
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        adj = [[] for _ in range(n)]
        for bi, (a, b) in enumerate(bonds):
            adj[a].append((b, bi)); adj[b].append((a, bi))
        masses = set()

        def pieces(cut: set) -> None:
            seen = [False] * n
            for s in range(n):
                if seen[s]:
                    continue
                stack, tot, cnt = [s], 0.0, 0
                seen[s] = True
                while stack:
                    u = stack.pop(); tot += am[u]; cnt += 1
                    for v, bi in adj[u]:
                        if not seen[v] and bi not in cut:
                            seen[v] = True; stack.append(v)
                if cnt < n:
                    masses.add(round(tot, 5))

        from itertools import combinations
        acyclic = [bi for bi, b in enumerate(mol.GetBonds()) if not b.IsInRing()]
        for i in acyclic:                        # one cut: only acyclic bonds disconnect
            pieces({i})
        pairs = list(combinations(acyclic, 2))
        if len(pairs) > self.max_cut_pairs:
            pairs = pairs[: self.max_cut_pairs]
        for i, j in pairs:                       # two acyclic cuts: the middle pieces
            pieces({i, j})
        for ring in mol.GetRingInfo().BondRings():   # two cuts in one ring: ring openings
            for i, j in combinations(ring, 2):
                pieces({i, j})
        out = np.array(sorted(masses))
        self.cache[smiles] = out
        return out

    def ion_masses(self, smiles: str, mode: str) -> np.ndarray:
        frag = self.neutral_fragments(smiles)
        if not len(frag):
            return frag
        sign = 1.0 if str(mode).lower().startswith("p") else -1.0
        shifts = np.arange(-self.h_shift, self.h_shift + 1) * H
        return np.unique((frag[:, None] + shifts[None, :]).ravel() + sign * PROTON)


def peaks(mzs, intensities, precursor_mz: float, top_n: int = 30, margin: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """The spectrum's top_n peaks below precursor - margin, as (m/z, sqrt intensity)."""
    mz = np.asarray(mzs, float); it = np.asarray(intensities, float)
    keep = (mz < float(precursor_mz) - margin) & (it > 0)
    mz, it = mz[keep], it[keep]
    if len(mz) > top_n:
        idx = np.argsort(-it)[:top_n]
        mz, it = mz[idx], it[idx]
    return mz, np.sqrt(it)


def explained(ion: np.ndarray, mz: np.ndarray, w: np.ndarray, tol: float = 0.01) -> float:
    """Share of the peaks' weight within tol of an ion mass."""
    if not len(ion) or not len(mz) or w.sum() <= 0:
        return 0.0
    pos = np.searchsorted(ion, mz)
    lo, hi = np.clip(pos - 1, 0, len(ion) - 1), np.clip(pos, 0, len(ion) - 1)
    d = np.minimum(np.abs(ion[lo] - mz), np.abs(ion[hi] - mz))
    return float(w[d <= tol].sum() / w.sum())
