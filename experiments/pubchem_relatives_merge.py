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


GENUINE_LOSS = {"hexose", "deoxyhexose", "pentose", "glucuronide", "acetyl", "malonyl"}


def merge(lst, prods, beta, w_rel, max_rank=3, max_insert=3, w_loss=0.0, guard=np.inf, w_relz=0.0):
    """Exactly the insertion rule of casmi/edit_overlay.py: the best `max_insert` products are each placed below the
    lowest of the original top `max_rank` members that scores at least as high, if that slot is < max_rank."""
    keys = [k for k, _ in lst]; top = [s for _, s in lst][:max_rank]
    if top and top[0] >= guard:
        return keys
    p = prods[~prods.key.isin(set(keys))]
    if p.empty:
        return keys
    g = p.z_p + beta + w_rel * p.rel_doc + w_relz * p.rel_z + w_loss * p.loss_ok
    p = p.assign(g=g).sort_values("g", ascending=False).drop_duplicates("key")
    slots = {}
    for k, gp in zip(p.key[:max_insert], p.g[:max_insert]):
        beaten = [i for i, si in enumerate(top) if si >= gp]
        r = (max(beaten) + 1) if beaten else 0
        if r < max_rank:
            slots.setdefault(r, []).append(k)
    if not slots:
        return keys
    out = []
    for i, k in enumerate(keys):
        out.extend(slots.get(i, [])); out.append(k)
    return out[:25]


def rr(keys, truth):
    return 1.0 / (keys.index(truth) + 1) if truth in keys[:25] else 0.0


if __name__ == "__main__":
    c = load_candidates()
    prods = pd.read_parquet(ROOT / "artifacts" / "relatives_products.parquet")
    loss = pd.read_parquet(ROOT / "artifacts" / "panel_loss_edits.parquet")
    lk = set(zip(loss.molecule, loss.edit))
    prods["loss_ok"] = [float(d == -1 and e in GENUINE_LOSS and (m, e) in lk) for m, e, d in zip(prods.molecule, prods.edit, prods.direction)]
    hit_all = prods[prods.key == prods.molecule]
    print("answer-generating products with loss evidence:", round(float(hit_all.loss_ok.mean()), 2) if len(hit_all) else None,
          "| all products:", round(float(prods.loss_ok.mean()), 3))
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
    L = lists_and_gate(c, 0.4)
    mols = sorted(L)
    b2 = [rr([k for k, _ in L[m][0]], m) for m in mols]
    n = 250
    grid = [dict(beta=b, w_rel=wr, w_loss=wl, guard=g, w_relz=wz)
            for b in (0.0, 1.0, 2.0, 3.0) for wr in (0.0, 0.2) for wl in (0.0, 1.5, 3.0) for g in (np.inf, 6.0) for wz in (0.0, 0.5)]
    res = []
    for kw in grid:
        r2 = [rr(merge(L[m][0], by_m.get(m, empty), **kw), m) for m in mols]
        r3 = [rr(merge(L[m][1], by_m.get(m, empty), **kw), m) for m in mols]
        d2, c3 = (np.sum(r2) - np.sum(b2)) / n, np.sum(r3) / n
        res.append(dict(**kw, class2=round(d2, 4), class3=round(c3, 4), net25=round(0.75 * d2 + 0.25 * c3, 4),
                        net10=round(0.9 * d2 + 0.1 * c3, 4)))
    res = pd.DataFrame(res).sort_values("net25", ascending=False)
    halves = [mols[0::2], mols[1::2]]
    def half_net(kw, ms):
        b = [rr([k for k, _ in L[m][0]], m) for m in ms]
        r2 = [rr(merge(L[m][0], by_m.get(m, empty), **kw), m) for m in ms]
        r3 = [rr(merge(L[m][1], by_m.get(m, empty), **kw), m) for m in ms]
        return round(0.75 * (np.sum(r2) - np.sum(b)) / len(ms) + 0.25 * np.sum(r3) / len(ms), 4)
    cols = ["beta", "w_rel", "w_loss", "guard", "w_relz"]
    res["half_a"] = [half_net(dict(zip(cols, r)), halves[0]) for r in res[cols].itertuples(index=False)]
    res["half_b"] = [half_net(dict(zip(cols, r)), halves[1]) for r in res[cols].itertuples(index=False)]
    print(f"class-2 base {np.sum(b2) / n:.4f}")
    print(res.head(20).to_string(index=False))
    print("...default-like rows:"); print(res[(res.w_rel == 0) & (res.w_loss == 0) & (res.w_relz == 0) & np.isinf(res.guard)].to_string(index=False))
    if len(hit_all):
        print("answer-generating edits:", hit_all.drop_duplicates("molecule").groupby(["edit", "direction"]).size().to_dict())
