"""CPU harness on the natural-product panel (dmitriigluzdov/casmi26-natural-product-panel-benchmark).

The panel holds every PubChem candidate in the 10 ppm window of the 250 enveda-np-examples molecules, with
fingerprint-model scores. `fold0_full` comes from the fold-safe FPNet, which never saw these molecules, so rules
built on it are not inflated by training leakage. Documentation counts are still easier here than on the hidden
test (the panel molecules are well-documented COCONUT natural products), so compare rules, not absolute scores.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path(os.environ.get("CASMI_PANEL", Path(__file__).resolve().parents[2] / "casmi_data" / "ext" / "panel"))
TIER = Path(os.environ.get("CASMI_PCTIER", Path(__file__).resolve().parents[2] / "casmi_data" / "ext" / "pubchem_prior"))
N_MOLECULES = 250


def load_candidates() -> pd.DataFrame:
    """One row per PubChem window candidate: identity, truth flag, model and documentation scores."""
    c = pd.read_parquet(PANEL / "pubchem_candidates.parquet")
    f = pd.read_parquet(PANEL / "panel_fpnet_scores.parquet", columns=["tier_row", "fold0_full", "ab_full"])
    assert (c.tier_row.to_numpy() == f.tier_row.to_numpy()).all()
    c["fold0_full"] = f.fold0_full.to_numpy()
    c["ab_full"] = f.ab_full.to_numpy()
    c["molecule"] = c.molecule.astype(str)
    c["ident"] = np.where(c.scorer_key.to_numpy() != "", c.scorer_key.to_numpy(), c.connectivity.to_numpy())
    c["doc"] = c.log_substances + c.log_pubmed
    return c


def zscore(frame: pd.DataFrame, col: str) -> np.ndarray:
    x = frame[col].to_numpy(dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    s = pd.Series(x, index=frame.index)
    mu = s.groupby(frame.molecule, sort=False).transform("mean")
    sd = s.groupby(frame.molecule, sort=False).transform("std").clip(lower=1e-9)
    z = ((s - mu) / sd).to_numpy()
    return np.where(np.isfinite(z), z, -10.0)


def reciprocal_ranks(frame: pd.DataFrame, score: np.ndarray, k: int = 25) -> pd.Series:
    """Per-molecule reciprocal rank after sorting by `score` and keeping the first row of each identity."""
    d = pd.DataFrame({"molecule": frame.molecule.to_numpy(), "ident": frame.ident.to_numpy(),
                      "truth": frame.is_truth.to_numpy(), "s": score})
    d = d.sort_values(["molecule", "s"], ascending=[True, False], kind="stable")
    d = d[~d.duplicated(["molecule", "ident"])]
    d["rank"] = d.groupby("molecule", sort=False).cumcount()
    first = d[d.truth].groupby("molecule", sort=False)["rank"].min()
    return (1.0 / (first + 1)).where(first < k, 0.0)


def mrr(frame: pd.DataFrame, score: np.ndarray, k: int = 25) -> float:
    rr = reciprocal_ranks(frame, score, k)
    return float(rr.sum() / N_MOLECULES)


def tier_smiles(rows: np.ndarray) -> list[str]:
    """Stereo-free SMILES of PubChem tier rows (memory-mapped offsets and byte buffer)."""
    off = np.load(TIER / "casmi26-pubchem-tier" / "pc_off.npy", mmap_mode="r")
    buf = np.load(TIER / "casmi26-pubchem-tier" / "pc_smiles.npy", mmap_mode="r")
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    out = [""] * len(rows)
    for i in order:
        r = rows[i]
        out[i] = bytes(buf[off[r]:off[r + 1]]).decode()
    return out
