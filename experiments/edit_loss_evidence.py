"""Neutral-loss evidence for an edit: does any spectrum of the molecule show a peak at precursor - d_e?

A glycoside, ester or methyl ether built from a lighter relative should lose the added group on fragmentation
(hexose 162.05, acetyl 42.01 as ketene, ...), so the relative's ion appears in the spectrum. Returns, per molecule,
the edits whose loss peak is present (relative intensity >= MIN_REL within TOL Da), for products of lighter
relatives (direction -1)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casmi.edit_overlay import FORWARD  # noqa: E402
from casmi.metric import ik14  # noqa: E402

TOL, MIN_REL = 0.01, 0.01


def loss_edits(spectra: pd.DataFrame, tol: float = TOL, min_rel: float = MIN_REL) -> dict:
    """spectra: rows of one molecule (precursor_mz, ms2_mzs, ms2_normalized_intensities) -> {edit: best rel. intensity}."""
    out = {}
    for r in spectra.itertuples(index=False):
        mz = np.asarray(r.ms2_mzs, float); it = np.asarray(r.ms2_normalized_intensities, float)
        if not len(mz):
            continue
        it = it / it.max()
        for e, (d, _) in FORWARD.items():
            hit = np.abs(mz - (float(r.precursor_mz) - d)) <= tol
            if hit.any():
                v = float(it[hit].max())
                if v >= min_rel:
                    out[e] = max(out.get(e, 0.0), v)
    return out


def panel_loss_table() -> pd.DataFrame:
    t = pq.read_table(ROOT.parents[0] / "casmi_data" / "raw" / "train.parquet",
                      columns=["ingest_lib", "normalized_smiles", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"],
                      filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas()
    t["molecule"] = [ik14(s) for s in t.normalized_smiles]
    rows = [dict(molecule=m, edit=e, loss=v) for m, g in t.groupby("molecule") for e, v in loss_edits(g).items()]
    return pd.DataFrame(rows, columns=["molecule", "edit", "loss"])


if __name__ == "__main__":
    tab = panel_loss_table()
    tab.to_parquet(ROOT / "artifacts" / "panel_loss_edits.parquet")
    print(tab.groupby("edit").size().sort_values(ascending=False).to_string())
    print("molecules with any loss peak:", tab.molecule.nunique())
