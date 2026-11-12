"""Does a precursor mass-error likelihood help on the panel? Candidates come from a +-10 ppm window, while
timsTOF errors are ~1.7 ppm median, so a candidate 8 ppm away is far less likely than one 1 ppm away."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casmi.panel import PANEL, TIER, load_candidates, mrr, zscore  # noqa: E402

if __name__ == "__main__":
    c = load_candidates()
    p = pd.read_parquet(PANEL / "panel.parquet")
    p["molecule"] = p.molecule.astype(str)
    print(p.mass_error_ppm.abs().describe(percentiles=[.5, .9, .99]).round(2).to_dict())
    mass = np.load(TIER / "casmi26-pubchem-tier" / "pc_mass.npy", mmap_mode="r")
    print("pc_mass dtype", mass.dtype)
    m = np.asarray(mass[c.tier_row.to_numpy()], dtype=np.float64)
    meas = c.molecule.map(p.set_index("molecule").measured_neutral_mass).to_numpy()
    ppm = (m - meas) / meas * 1e6
    c["ppm"] = ppm
    t = c[c.is_truth]
    print("truth |ppm| from tier mass:", np.percentile(np.abs(t.ppm), [50, 90, 99, 100]).round(2))
    print("all candidates |ppm|:", np.percentile(np.abs(c.ppm), [10, 50, 90, 100]).round(2))
    base = zscore(c, "fold0_full")
    doc = c.doc.to_numpy()
    for w in (0.25, 0.4):
        b = base + w * doc
        print(f"fold-safe z + {w} doc: {mrr(c, b):.4f}")
        for sig in (1.5, 2.0, 2.5, 3.0, 4.0):
            for a in (0.5, 1.0):
                ll = -0.5 * (c.ppm.to_numpy() / sig) ** 2
                print(f"   + {a} * ll(sigma {sig}): {mrr(c, b + a * ll):.4f}")
    print(f"fold-safe z alone: {mrr(c, base):.4f}; + ll(2.0): {mrr(c, base - 0.5 * (c.ppm.to_numpy() / 2.0) ** 2):.4f}")
