"""Replay C1's list assembly under other library-hit rules, from c1_panel.py dumps (artifacts/c1_dump_c2.pkl,
c1_dump_c1.pkl: the same PubChem lists, library hits with the answer's spectra hidden / with only the queries' own
library left out). Class 2 shows what false hits cost, the class-1 view what true hits gain.

Rules: none (PubChem lists only); first tau (every hit >= tau first, best first); top1 tau (only the best hit, if
>= tau, at rank 1); top1 tau + margin (the best hit only if it beats the second-best hit by `margin`).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT))
from casmi.edit_overlay import Chemistry, Tier  # noqa: E402


def assemble(m, pub, hits, rule, tier, chem, top_k=25):
    h = sorted(hits.get(m, {}).items(), key=lambda x: -x[1][0])
    kind, tau, margin = rule
    lead = []
    if kind == "first":
        lead = [s for _, (c, s) in h if c >= tau]
    elif kind == "top1" and h and h[0][1][0] >= tau:
        if len(h) == 1 or h[0][1][0] - h[1][1][0] >= margin:
            lead = [h[0][1][1]]
    out, seen = [], set()
    for s in lead + [tier.smiles(r) for r, _ in pub.get(m, [])]:
        k = chem.key(s)
        if k and k not in seen:
            seen.add(k); out.append(k)
        if len(out) >= top_k:
            break
    return out


def mrr(lists: dict, ids) -> float:
    tot = 0.0
    for m in ids:
        ks = lists.get(m, [])
        tot += 1.0 / (ks.index(m) + 1) if m in ks[:25] else 0.0
    return tot / 250


if __name__ == "__main__":
    art = ROOT / "artifacts"
    d2 = pickle.load(open(art / "c1_dump_c2.pkl", "rb")); d1 = pickle.load(open(art / "c1_dump_c1.pkl", "rb"))
    index = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    tier = Tier(index, index) if index else Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    ids = d2["ids"]
    n2 = sum(bool(v) for v in d2["hits"].values()); n1 = sum(bool(v) for v in d1["hits"].values())
    print(f"molecules with hits >= 0.5: class 2 (all false) {n2}, class-1 view {n1}")
    rules = [("none", 9, 0)] + [("first", t, 0) for t in (0.75, 0.8, 0.85, 0.9, 0.95)] + \
            [("top1", t, 0) for t in (0.75, 0.8, 0.85, 0.9, 0.95)] + [("top1", t, mg) for t in (0.8, 0.9) for mg in (0.05, 0.1)]
    print("rule              | class 2 | class-1 view | 0.2 x c1 + 0.55 x c2 (class 3 not modelled)")
    for r in rules:
        c2 = mrr({m: assemble(m, d2["pub"], d2["hits"], r, tier, chem) for m in ids}, ids)
        c1 = mrr({m: assemble(m, d1["pub"], d1["hits"], r, tier, chem) for m in ids}, ids)
        name = "no hits" if r[0] == "none" else f"{r[0]} {r[1]}" + (f" margin {r[2]}" if r[2] else "")
        print(f"{name:17s} | {c2:.4f}  | {c1:.4f}       | {0.2 * c1 + 0.55 * c2:.4f}")
