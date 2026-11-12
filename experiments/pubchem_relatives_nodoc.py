"""Worst case for class 2: the answer has no documentation at all (doc set to 0 for the answer's rows, which also
lowers it in the fork-like list). How much do PubChem-relative products cost then?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.panel import load_candidates  # noqa: E402
from pubchem_relatives_merge import lists_and_gate, merge, rr  # noqa: E402

if __name__ == "__main__":
    c = load_candidates()
    c.loc[c.is_truth, "doc"] = 0.0                      # undocumented answer
    prods = pd.read_parquet(ROOT / "artifacts" / "relatives_products.parquet")
    sup = prods.groupby(["molecule", "key"]).apply(lambda g: g[["rel_row", "edit"]].drop_duplicates().shape[0]).rename("support")
    prods = prods.merge(sup.reset_index(), on=["molecule", "key"]); prods["loss_ok"] = 0.0
    L = lists_and_gate(c, 0.4); mols = sorted(L)
    b = [rr([k for k, _ in L[m][0]], m) for m in mols]
    print(f"class-2 lists with undocumented answers: base MRR {np.mean(b):.4f}")
    for w_sup, w_rel, beta, guard in ((0.5, 0.2, 0.0, np.inf), (0.5, 0.2, -1.0, np.inf), (0.5, 0.2, 0.0, 4.0), (0.5, 0.2, 0.0, 3.0),
                                      (0.5, 0.1, 0.0, np.inf), (0.25, 0.2, 0.0, np.inf)):
        g = prods.assign(rel_doc=prods.rel_doc + (w_sup / w_rel) * np.log(prods.support))
        by = {m: x for m, x in g.groupby("molecule")}; empty = g.iloc[:0]
        r2 = [rr(merge(L[m][0], by.get(m, empty), beta, w_rel, guard=guard), m) for m in mols]
        r3 = [rr(merge(L[m][1], by.get(m, empty), beta, w_rel, guard=guard), m) for m in mols]
        print(f"w_sup {w_sup} w_rel {w_rel} beta {beta} guard {guard}: undocumented class-2 {np.mean(r2) - np.mean(b):+.4f} "
              f"(hurt {sum(x < y for x, y in zip(r2, b))}, helped {sum(x > y for x, y in zip(r2, b))}) | class-3 {np.mean(r3):.4f}")
