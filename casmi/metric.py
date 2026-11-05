"""Exact replica of the official CASMI 2026 metric (Kaggle notebook metric/casmi-mean-reciprocal-rank).

Key facts (verified from the official source):
- Solution key canonicalized with RDKit 2026.03.3. Tautomer output changes across RDKit releases,
  so always build candidate keys with exactly that version.
- Stereochemistry is removed BEFORE tautomer canonicalization (a stereocentre can pin a double bond).
- Rank = position in the semicolon list. Duplicates and unparseable SMILES STILL consume a rank
  position, so a submission must be deduplicated on this key before writing.
- Every molecule needs >= 1 guess and <= 25 guesses; NaN or duplicate molecule_id raise errors.
- The solution cell may hold several comma-separated acceptable SMILES (indistinguishable isomers).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
EXPECTED_RDKIT_VERSION = "2026.03.3"
MAX_RANK = 25
_TAUT = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=4_000_000)
def ik14(smiles: str) -> str | None:
    """Scorer key: stereo stripped, tautomer-canonical InChIKey first block. None if unparseable."""
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        mol = _TAUT.Canonicalize(mol)
        key = Chem.MolToInchiKey(mol)
        return key[:14] if key else None
    except (ValueError, RuntimeError, TypeError):
        return None


def heavy_atom_composition(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != "H":
            counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))


def reciprocal_rank(answer_smiles: str, guesses: list[str], max_rank: int = MAX_RANK) -> float:
    """Official rule: first guess whose key is accepted scores 1/position (position counts every guess)."""
    accept = {k for k in map(ik14, [s for s in str(answer_smiles).split(",") if s]) if k}
    for rank, g in enumerate(guesses[:max_rank], 1):
        if ik14(g) in accept:
            return 1.0 / rank
    return 0.0


def parse_cell(cell: str) -> list[str]:
    return [c.strip() for c in str(cell).split(";") if c.strip()]


def score(solution: pd.DataFrame, submission: pd.DataFrame, max_rank: int = MAX_RANK) -> float:
    """Mean reciprocal rank over solution molecules (solution: molecule_id, smiles)."""
    if submission["molecule_id"].duplicated().any():
        raise ValueError("Each molecule_id must appear exactly once")
    guesses = {m: parse_cell(s) for m, s in zip(submission["molecule_id"], submission["smiles"])}
    for m, g in guesses.items():
        if not g:
            raise ValueError(f"No guesses for {m}")
        if len(g) > max_rank:
            raise ValueError(f"More than {max_rank} guesses for {m}")
    rr = [reciprocal_rank(s, guesses.get(m, []), max_rank) for m, s in zip(solution["molecule_id"], solution["smiles"])]
    return float(np.mean(rr))
