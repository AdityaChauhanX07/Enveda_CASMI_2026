"""Does molecular networking improve the candidate order on the natural-product panel?

Base order: fold-safe FPNet z-score + w * documentation. Network term for a candidate c of molecule B:
the expected Tanimoto between c and the likely structures of B's spectral neighbours A, where "likely" is
the softmax of A's own base scores over its top candidates. No truth is used to build the term.

Usage: python experiments/network_panel.py [graph.parquet]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casmi.panel import N_MOLECULES, load_candidates, tier_smiles, zscore  # noqa: E402

RDLogger.DisableLog("rdApp.*")
TOP_M = 300      # candidates per molecule that can receive the network term
TOP_K = 30       # neighbour candidates that define the neighbour's likely structure
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def shortlist(c: pd.DataFrame, s0: np.ndarray) -> pd.DataFrame:
    d = c[["molecule", "ident", "tier_row", "is_truth"]].copy()
    d["s0"] = s0
    d = d.sort_values(["molecule", "s0"], ascending=[True, False], kind="stable")
    d["truth_id"] = d.groupby(["molecule", "ident"], sort=False).is_truth.transform("any")
    d = d[~d.duplicated(["molecule", "ident"])]
    d["rank0"] = d.groupby("molecule", sort=False).cumcount()
    return d[d.rank0 < TOP_M].reset_index(drop=True)


def fingerprints(rows: np.ndarray) -> list:
    smiles = tier_smiles(rows)
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(GEN.GetFingerprint(m) if m is not None else None)
    return out


def network_term(d: pd.DataFrame, fps: list, graph: pd.DataFrame, tau: float) -> np.ndarray:
    groups = {m: g.index.to_numpy() for m, g in d.groupby("molecule", sort=False)}
    prob = {}
    for m, idx in groups.items():
        top = idx[:TOP_K]
        x = d.s0.to_numpy()[top]
        p = np.exp((x - x.max()) / tau)
        prob[m] = (top, p / p.sum())
    nbr = {}
    for a, b, w in zip(graph.a, graph.b, graph.cosine):
        nbr.setdefault(a, []).append((b, w))
        nbr.setdefault(b, []).append((a, w))
    term = np.zeros(len(d))
    for m, idx in groups.items():
        if m not in nbr:
            continue
        acc, wsum = np.zeros(len(idx)), 0.0
        for a, w in nbr[m]:
            if a not in prob:
                continue
            top, p = prob[a]
            sims = np.zeros(len(idx))
            for j, pj in zip(top, p):
                if fps[j] is None:
                    continue
                v = np.array(DataStructs.BulkTanimotoSimilarity(fps[j], [fps[i] if fps[i] is not None else fps[j] for i in idx]))
                sims += pj * v
            acc += w * sims
            wsum += w
        if wsum:
            term[idx] = acc / wsum
    return term


def mrr_short(d: pd.DataFrame, score: np.ndarray, molecules=None) -> float:
    e = d.assign(s=score).sort_values(["molecule", "s"], ascending=[True, False], kind="stable")
    e["r"] = e.groupby("molecule", sort=False).cumcount()
    first = e[e.truth_id].groupby("molecule", sort=False).r.min()
    rr = (1.0 / (first + 1)).where(first < 25, 0.0)
    if molecules is None:
        return float(rr.sum() / N_MOLECULES)
    rr = rr.reindex(sorted(molecules)).fillna(0.0)
    return float(rr.mean())


def main(graph_path: str) -> None:
    t0 = time.time()
    c = load_candidates()
    zf = zscore(c, "fold0_full")
    graph_all = pd.read_parquet(graph_path)
    for w_doc in (0.0, 0.1, 0.4):
        s0 = zf + w_doc * c.doc.to_numpy()
        d = shortlist(c, s0)
        fps = fingerprints(d.tier_row.to_numpy())
        print(f"\nw_doc={w_doc}: shortlist {len(d)} rows, fingerprints in {time.time() - t0:.0f} s; "
              f"base MRR {mrr_short(d, d.s0.to_numpy()):.4f}")
        for cos_min, min_matches in ((0.7, 6), (0.8, 6), (0.6, 8)):
            g = graph_all[(graph_all.cosine >= cos_min) & (graph_all.matches >= min_matches)]
            linked = set(g.a) | set(g.b)
            for tau in (0.5, 2.0):
                term = network_term(d, fps, g, tau)
                sd = d.s0.groupby(d.molecule).transform("std").to_numpy()
                for lam in (0.5, 1.0, 2.0, 4.0):
                    s = d.s0.to_numpy() + lam * sd * term
                    print(f"  cos>={cos_min} m>={min_matches} tau={tau} lam={lam}: all {mrr_short(d, s):.4f} "
                          f"(base {mrr_short(d, d.s0.to_numpy()):.4f}) | linked {len(linked)} mols "
                          f"{mrr_short(d, s, linked):.4f} vs {mrr_short(d, d.s0.to_numpy(), linked):.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "artifacts" / "np_graph.parquet"))
