"""Why does the base rule (fold-safe z + 0.4 doc) miss on the panel? Compare the answer with the candidate that beats it."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casmi.panel import PANEL, load_candidates, zscore  # noqa: E402

if __name__ == "__main__":
    c = load_candidates()
    sf = pd.read_parquet(PANEL / "panel_same_formula.parquet")
    c["same_formula"] = sf.same_formula.to_numpy().astype(bool)
    c["z"] = zscore(c, "fold0_full")
    c["s"] = c.z + 0.4 * c.doc
    d = c.sort_values(["molecule", "s"], ascending=[True, False])
    d = d[~d.duplicated(["molecule", "ident"])].copy()
    d["rank"] = d.groupby("molecule").cumcount() + 1
    truth = d[d.is_truth].groupby("molecule").head(1).set_index("molecule")
    top = d[d["rank"] == 1].set_index("molecule")
    miss = truth[truth["rank"] > 1].index
    t, w = truth.loc[miss], top.loc[miss]
    print(f"misses {len(miss)} of {truth.shape[0]} ranked answers; answer rank median {t['rank'].median():.0f}, "
          f"in top 5 {(t['rank'] <= 5).mean():.2f}, in top 25 {(t['rank'] <= 25).mean():.2f}")
    print(f"winner has the answer's formula: {w.same_formula.mean():.2f}")
    print(f"winner better documented: {(w.doc.values > t.doc.values).mean():.2f}; higher z: {(w.z.values > t.z.values).mean():.2f}")
    print(f"doc gap (winner - answer) median {np.median(w.doc.values - t.doc.values):.2f}; "
          f"z gap median {np.median(w.z.values - t.z.values):.2f}")
    print("answer z quantiles over misses", np.percentile(t.z, [10, 50, 90]).round(2),
          "| winner z", np.percentile(w.z, [10, 50, 90]).round(2))
    # what a perfect spectral tie-break among the top-k by doc would give
    hits = c[c.is_truth].groupby("molecule").z.max()
    print("answer z over all 250: median", round(float(hits.median()), 2), "share above 2:", round(float((hits > 2).mean()), 2))
