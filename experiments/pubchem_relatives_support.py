"""Does consensus help rank PubChem-relative products? support = number of distinct (relative, edit) routes to a product."""
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
    prods = pd.read_parquet(ROOT / "artifacts" / "relatives_products.parquet")
    sup = prods.groupby(["molecule", "key"]).apply(lambda g: g[["rel_row", "edit"]].drop_duplicates().shape[0]).rename("support")
    prods = prods.merge(sup.reset_index(), on=["molecule", "key"]); prods["loss_ok"] = 0.0
    best_doc = prods.groupby(["molecule", "key"]).rel_doc.transform("max")
    hit = prods[prods.key == prods.molecule].drop_duplicates("molecule")
    print("support of answer products: mean", round(hit.support.mean(), 2), "| of all products:",
          round(prods.drop_duplicates(["molecule", "key"]).support.mean(), 2))
    L = lists_and_gate(c, 0.4)
    mols = sorted(L); n = 250
    b2 = [rr([k for k, _ in L[m][0]], m) for m in mols]
    by = {m: g for m, g in prods.groupby("molecule")}
    empty = prods.iloc[:0]
    for w_sup in (0.0, 0.5, 1.0, 2.0):
        for beta, w_rel in ((0.0, 0.2), (1.0, 0.2)):
            def mk(g):
                g = g.assign(rel_doc=g.rel_doc + (w_sup / w_rel) * np.log(g.support))     # folds w_sup*log(support) into the score
                return g
            r2 = [rr(merge(L[m][0], mk(by.get(m, empty)), beta, w_rel), m) for m in mols]
            r3 = [rr(merge(L[m][1], mk(by.get(m, empty)), beta, w_rel), m) for m in mols]
            d2, c3 = (np.sum(r2) - np.sum(b2)) / n, np.sum(r3) / n
            print(f"w_sup {w_sup} beta {beta} w_rel {w_rel}: class-2 {d2:+.4f} class-3 {c3:.4f} net25 {0.75 * d2 + 0.25 * c3:+.4f}")
    print("split halves (net at 25 % class 3):")
    for w_sup in (0.25, 0.5, 0.75):
        g = prods.assign(rel_doc=prods.rel_doc + (w_sup / 0.2) * np.log(prods.support))
        byg = {m: x for m, x in g.groupby("molecule")}
        out = []
        for half in (mols[0::2], mols[1::2]):
            b = [rr([k for k, _ in L[m][0]], m) for m in half]
            r2 = [rr(merge(L[m][0], byg.get(m, empty), 0.0, 0.2), m) for m in half]
            r3 = [rr(merge(L[m][1], byg.get(m, empty), 0.0, 0.2), m) for m in half]
            out.append(f"class-2 {(np.sum(r2) - np.sum(b)) / len(half):+.4f} class-3 {np.sum(r3) / len(half):.4f}")
        print(f"  w_sup {w_sup}: " + " | ".join(out))
