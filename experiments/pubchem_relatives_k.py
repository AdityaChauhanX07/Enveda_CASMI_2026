"""How many relatives per edit window? Products cached with up to 6 relatives (REL_CFG=300,100,20,6); keep the top k,
recount routes, score z + 0.2 x relative doc + 0.5 x log(routes) and merge as the overlay does."""
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
    allp = pd.read_parquet(ROOT / "artifacts" / (sys.argv[1] if len(sys.argv) > 1 else "relatives_products_k6.parquet"))
    L = lists_and_gate(c, 0.4); mols = sorted(L)
    b = [rr([k for k, _ in L[m][0]], m) for m in mols]
    for k in (3, 4, 5, 6):
        p = allp[allp.rel_rank < k]
        sup = p.groupby(["molecule", "key"]).apply(lambda g: g[["rel_row", "edit"]].drop_duplicates().shape[0]).rename("support")
        p = p.merge(sup.reset_index(), on=["molecule", "key"]); p["loss_ok"] = 0.0
        gen = p[p.key == p.molecule].molecule.nunique()
        for w_sup in (0.5, 0.75):
            g = p.assign(rel_doc=p.rel_doc + (w_sup / 0.2) * np.log(p.support))
            by = {m: x for m, x in g.groupby("molecule")}; empty = g.iloc[:0]
            r2 = [rr(merge(L[m][0], by.get(m, empty), 0.0, 0.2), m) for m in mols]
            r3 = [rr(merge(L[m][1], by.get(m, empty), 0.0, 0.2), m) for m in mols]
            h = [np.mean([r3[i] for i in range(len(mols)) if i % 2 == j]) for j in (0, 1)]
            print(f"k {k} w_sup {w_sup}: answer generated {gen:3d} | products {len(p):6d} | class-2 {np.mean(r2) - np.mean(b):+.4f} "
                  f"(hurt {sum(x < y for x, y in zip(r2, b))}) | class-3 {np.mean(r3):.4f} (halves {h[0]:.3f} / {h[1]:.3f})", flush=True)
