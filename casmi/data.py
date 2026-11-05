"""Parquet loaders. Data root defaults to ../casmi_data/raw (outside the repo)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = Path(os.environ.get("CASMI_DATA", Path(__file__).resolve().parents[2] / "casmi_data" / "raw"))
META_COLS = [
    "ingest_lib", "normalized_smiles", "inchikey14", "molecular_formula", "ionization_mode", "instrument_type",
    "adduct", "precursor_mz", "precursor_error_ppm", "num_peaks", "base_peak_intensity",
    "collision_energy_ev", "collision_energy_orig_units",
]


def train_meta(cols: list[str] | None = None) -> pd.DataFrame:
    """All 2.54M training rows without peak arrays (loads in ~1 s)."""
    return pq.read_table(DATA / "train.parquet", columns=cols or META_COLS).to_pandas()


def train_rows(mask_fn=None, columns: list[str] | None = None, batch_size: int = 200_000):
    """Stream training rows (with peaks) in batches; optional row filter mask_fn(df)->bool array."""
    f = pq.ParquetFile(DATA / "train.parquet")
    for batch in f.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        yield df[mask_fn(df)] if mask_fn is not None else df


def np_examples() -> pd.DataFrame:
    """The 1,184 enveda-np-examples spectra (250 structures): timsTOF, same pipeline as the hidden test."""
    parts = list(train_rows(lambda d: (d.ingest_lib == "enveda-np-examples").to_numpy()))
    return pd.concat(parts, ignore_index=True)


def test() -> pd.DataFrame:
    return pd.read_parquet(DATA / "test.parquet")


def peaks(row) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(row["ms2_mzs"], dtype=np.float64), np.asarray(row["ms2_normalized_intensities"], dtype=np.float64)
