"""Networking as an overlay on finished answer lists: only the order of each molecule's top 25 is known.

Neighbour belief: p(rank r) proportional to 1 / (r + k). Network term: expected Tanimoto between a candidate and
the neighbour's list. New order: rank score -log(1 + r) + lam * term, for ranks 2-25 only when lead_safe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from casmi.panel import N_MOLECULES, load_candidates, zscore  # noqa: E402
from network_panel import fingerprints, shortlist  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def rr_of(order_truth: dict) -> pd.Series:
    out = {}
    for m, t in order_truth.items():
        hit = np.flatnonzero(t)
        out[m] = 1.0 / (hit[0] + 1) if len(hit) and hit[0] < 25 else 0.0
    return pd.Series(out)


def main() -> None:
    c = load_candidates()
    graph_all = pd.read_parquet(ROOT / "artifacts" / "np_graph.parquet")
    zf = zscore(c, "fold0_full")
    for w_doc in (0.1, 0.4):
        d = shortlist(c, zf + w_doc * c.doc.to_numpy())
        d = d[d.rank0 < 25].reset_index(drop=True)
        fps = fingerprints(d.tier_row.to_numpy())
        lists = {m: g.index.to_numpy() for m, g in d.groupby("molecule", sort=False)}
        base = rr_of({m: d.truth_id.to_numpy()[ix] for m, ix in lists.items()})
        print(f"\nw_doc={w_doc}: base MRR {base.sum() / N_MOLECULES:.4f}")
        for cos_min, min_m in ((0.7, 6), (0.6, 8)):
            g = graph_all[(graph_all.cosine >= cos_min) & (graph_all.matches >= min_m)]
            nbr = {}
            for a, b, w in zip(g.a, g.b, g.cosine):
                nbr.setdefault(a, []).append((b, w)); nbr.setdefault(b, []).append((a, w))
            for k_rr in (1.0, 3.0):
                term = {}
                for m, ix in lists.items():
                    acc, ws = np.zeros(len(ix)), 0.0
                    for a, w in nbr.get(m, []):
                        if a not in lists:
                            continue
                        jx = lists[a]
                        p = 1.0 / (np.arange(len(jx)) + k_rr); p /= p.sum()
                        sims = np.zeros(len(ix))
                        for j, pj in zip(jx, p):
                            sims += pj * np.array(DataStructs.BulkTanimotoSimilarity(fps[j], [fps[i] for i in ix]))
                        acc += w * sims; ws += w
                    term[m] = acc / ws if ws else None
                for lam in (1.0, 2.0, 4.0, 8.0):
                    for lead_safe in (False, True):
                        new = {}
                        for m, ix in lists.items():
                            t = d.truth_id.to_numpy()[ix]
                            if term[m] is None:
                                new[m] = t; continue
                            s = -np.log1p(np.arange(len(ix))) + lam * term[m]
                            if lead_safe:
                                o = np.r_[0, 1 + np.argsort(-s[1:], kind="stable")]
                            else:
                                o = np.argsort(-s, kind="stable")
                            new[m] = t[o]
                        rr = rr_of(new)
                        delta = (rr - base).reindex(base.index)
                        print(f"  cos>={cos_min} m>={min_m} k={k_rr} lam={lam} lead_safe={lead_safe}: "
                              f"MRR {rr.sum() / N_MOLECULES:.4f} (+{(rr.sum() - base.sum()) / N_MOLECULES:.4f}) "
                              f"up {int((delta > 1e-9).sum())} down {int((delta < -1e-9).sum())}")


if __name__ == "__main__":
    main()
