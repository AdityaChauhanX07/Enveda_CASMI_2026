"""Headroom of formula determination on the panel: MRR when only candidates with the answer's formula are kept
(oracle formula), and how often the base order's top wrong guess has the wrong formula."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casmi.panel import PANEL, load_candidates, mrr, zscore  # noqa: E402

if __name__ == "__main__":
    c = load_candidates()
    sf = pd.read_parquet(PANEL / "panel_same_formula.parquet")
    assert (sf.tier_row.to_numpy() == c.tier_row.to_numpy()).all()
    c["same_formula"] = sf.same_formula.to_numpy().astype(bool)
    z = zscore(c, "fold0_full")
    for name, s in (("fold-safe z", z), ("z + 0.4 doc", z + 0.4 * c.doc.to_numpy()), ("doc", c.doc.to_numpy())):
        oracle = np.where(c.same_formula.to_numpy(), s, -1e9)
        print(f"{name:12s} all formulas {mrr(c, s):.4f} | oracle formula {mrr(c, oracle):.4f}")
    s = z + 0.4 * c.doc.to_numpy()
    d = c.assign(s=s).sort_values(["molecule", "s"], ascending=[True, False])
    d = d[~d.duplicated(["molecule", "ident"])]
    top = d.groupby("molecule").head(1)
    wrong = top[~top.is_truth]
    print("molecules with a wrong top-1:", len(wrong), "of which wrong formula:", int((~wrong.same_formula).sum()))
    sizes = c.groupby("molecule").agg(n=("ident", "nunique"), nsf=("same_formula", "sum"))
    print("window identities median", sizes.n.median(), "same-formula rows median", sizes.nsf.median())
