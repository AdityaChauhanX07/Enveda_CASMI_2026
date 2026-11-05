"""Write a valid, scorer-aware submission: dedupe on the scorer key, keep <= 25, never empty."""
from __future__ import annotations

import pandas as pd

from .metric import MAX_RANK, ik14

FALLBACK = "CCO"


def finalize(ranked: list[str], fill: list[str] | None = None, k: int = MAX_RANK) -> list[str]:
    """Drop unparseable SMILES and repeated scorer keys (they would waste rank slots), then fill."""
    out, seen = [], set()
    for s in list(ranked) + list(fill or []):
        key = ik14(s)
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) == k:
            break
    return out or [FALLBACK]


def write(preds: dict[str, list[str]], molecule_ids: list[str], path: str = "submission.csv") -> pd.DataFrame:
    rows = [(m, ";".join(finalize(preds.get(m, [])))) for m in molecule_ids]
    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    validate(sub, molecule_ids)
    sub.to_csv(path, index=False)
    return sub


def validate(sub: pd.DataFrame, molecule_ids: list[str]) -> None:
    assert list(sub.columns) == ["molecule_id", "smiles"], sub.columns
    assert not sub.isna().any().any(), "NaN in submission"
    assert not sub.molecule_id.duplicated().any(), "duplicate molecule_id"
    assert set(sub.molecule_id) == set(molecule_ids), "molecule_id set differs from test"
    for cell in sub.smiles:
        g = [c for c in cell.split(";") if c.strip()]
        assert 1 <= len(g) <= MAX_RANK, f"bad guess count {len(g)}"
