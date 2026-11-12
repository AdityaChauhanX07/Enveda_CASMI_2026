"""Class-3 generator on the panel, step 2: merge cached PubChem-relative products into the lists and score them.

Lists: top 25 identities by fold-safe fold0 z + 0.4 doc (a stand-in for the fork), with the answer (class-2
situation) or without it (class-3 situation). Gate as in casmi/edit_overlay.py: a product enters the top
`max_rank` slots where its score z_p + beta (+ w_rel * relative's doc) beats list members' z_ens + w_doc * doc.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.panel import load_candidates, zscore  # noqa: E402
from panel_fp_matrix import PanelFP, score_frame  # noqa: E402


def lists_and_gate(c: pd.DataFrame, w_doc: float):
    L = np.load(ROOT / "artifacts" / "panel_logits_foldsafe.npz")
    ens = np.mean([0.5 * (L[f"single{k}"].astype(np.float64) + L[f"merged{k}"].astype(np.float64)) for k in range(5)], axis=0)
    c = c.assign(raw=score_frame(c, dict(zip([str(k) for k in L["keys"]], ens)), PanelFP()))
    c["z_ens"] = zscore(c, "raw")
    c["s0"] = zscore(c, "fold0_full") + 0.4 * c.doc.to_numpy()
    c["key"] = np.where(c.scorer_key.to_numpy() != "", c.scorer_key.to_numpy(), c.connectivity.to_numpy())
    c = c.sort_values(["molecule", "s0"], ascending=[True, False], kind="stable")
    c = c[~c.duplicated(["molecule", "ident"])]
    out = {}
    for m, g in c.groupby("molecule", sort=False):
        g2 = g.head(25); g3 = g[~g.is_truth].head(25)
        mk = lambda x: [(k, float(z + w_doc * d)) for k, z, d in zip(x.key, x.z_ens, x.doc)]
        out[m] = (mk(g2), mk(g3))
    return out


def merge(lst, prods, beta, w_rel, max_rank=3, max_insert=3):
    """lst: [(key, gate score)] in list order; prods: frame of one molecule's products."""
    keys = [k for k, _ in lst]; score = [s for _, s in lst]
    have = set(keys)
    p = prods[~prods.key.isin(have)].copy()
    if p.empty:
        return keys
    p["g"] = p.z_p + beta + w_rel * p.rel_doc
    p = p.sort_values("g", ascending=False).drop_duplicates("key")
    n = 0
    for k, gp in zip(p.key, p.g):
        if n >= max_insert:
            break
        top = score[:max_rank]
        beat = [i for i, s in enumerate(top) if s > gp]
        slot = (max(beat) + 1) if beat else 0
        if slot < max_rank:
            keys.insert(slot, k); score.insert(slot, gp); n += 1
    return keys[:25]


def rr(keys, truth):
    return 1.0 / (keys.index(truth) + 1) if truth in keys[:25] else 0.0


if __name__ == "__main__":
    c = load_candidates()
    prods = pd.read_parquet(ROOT / "artifacts" / "relatives_products.parquet")
    print(f"products {len(prods)} for {prods.molecule.nunique()} molecules; answer generated for "
          f"{prods[prods.key == prods.molecule].molecule.nunique()} molecules")
    hit = prods[prods.key == prods.molecule]
    if len(hit):
        print("generating edits:", hit.drop_duplicates("molecule").edit.value_counts().to_dict())
        rank_in = prods.assign(r=prods.groupby("molecule").z_p.rank(ascending=False, method="first"))
        rk = rank_in[rank_in.key == rank_in.molecule].groupby("molecule").r.min()
        print("answer's rank among its molecule's products: median", rk.median(), "| top1", (rk == 1).mean().round(2),
              "| top3", (rk <= 3).mean().round(2))
    by_m = {m: g for m, g in prods.groupby("molecule")}
    empty = prods.iloc[:0]
    for w_doc in (0.3, 0.4):
        L = lists_and_gate(c, w_doc)
        mols = sorted(L)
        for beta in (0.0, 1.0, 2.0, 3.0):
            for w_rel in (0.0, 0.2):
                r2 = [rr(merge(L[m][0], by_m.get(m, empty), beta, w_rel), m) for m in mols]
                r3 = [rr(merge(L[m][1], by_m.get(m, empty), beta, w_rel), m) for m in mols]
                b2 = [rr([k for k, _ in L[m][0]], m) for m in mols]
                n = 250
                mix = 0.75 * (np.sum(r2) - np.sum(b2)) / n + 0.25 * np.sum(r3) / n
                print(f"w_doc {w_doc} beta {beta} w_rel {w_rel}: class-2 {np.sum(b2)/n:.4f} -> {np.sum(r2)/n:.4f} | "
                      f"class-3 0 -> {np.sum(r3)/n:.4f} | net at 25 % class 3 {mix:+.4f}")
