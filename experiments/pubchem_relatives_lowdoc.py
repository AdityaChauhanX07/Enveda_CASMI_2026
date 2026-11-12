"""Class-2 cost of PubChem-relative products when the answer is poorly documented (the hidden test's class 2 is
probably less documented than the panel). Splits panel molecules by the answer's documentation."""
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
    ans_doc = c[c.is_truth].groupby("molecule").doc.max()
    prods = pd.read_parquet(ROOT / "artifacts" / "relatives_products.parquet")
    sup = prods.groupby(["molecule", "key"]).apply(lambda g: g[["rel_row", "edit"]].drop_duplicates().shape[0]).rename("support")
    prods = prods.merge(sup.reset_index(), on=["molecule", "key"]); prods["loss_ok"] = 0.0
    g = prods.assign(rel_doc=prods.rel_doc + (0.5 / 0.2) * np.log(prods.support))
    by = {m: x for m, x in g.groupby("molecule")}; empty = g.iloc[:0]
    L = lists_and_gate(c, 0.4); mols = sorted(L)
    med = ans_doc.reindex(mols).median()
    for name, ms in (("answer doc below median", [m for m in mols if ans_doc.get(m, 0) < med]),
                     ("answer doc above median", [m for m in mols if ans_doc.get(m, 0) >= med]),
                     ("answer doc in lowest 20 %", [m for m in mols if ans_doc.get(m, 0) <= ans_doc.reindex(mols).quantile(0.2)])):
        b = [rr([k for k, _ in L[m][0]], m) for m in ms]
        r2 = [rr(merge(L[m][0], by.get(m, empty), 0.0, 0.2), m) for m in ms]
        r3 = [rr(merge(L[m][1], by.get(m, empty), 0.0, 0.2), m) for m in ms]
        print(f"{name:28s} n={len(ms):3d} class-2 base {np.mean(b):.4f} cost {np.mean(r2) - np.mean(b):+.4f} | class-3 {np.mean(r3):.4f}"
              f" | molecules hurt {sum(x < y for x, y in zip(r2, b))}, helped {sum(x > y for x, y in zip(r2, b))}")
    print("answer doc quantiles:", ans_doc.quantile([0.1, 0.25, 0.5, 0.75]).round(2).to_dict())
