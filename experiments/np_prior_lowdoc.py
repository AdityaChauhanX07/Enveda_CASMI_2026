"""Natural-product membership prior on C1's lists, with the low-documentation stress test.

C1's shortlists (artifacts/c1_rows_v1.pkl, from the C1 dumps: fingerprint part z, documentation doc, library hits) are
re-scored as z + 0.25 x doc (answer doc capped at d) + library-hit bonus [+ spec2mol, pooled scale, gamma 0.75 = C1s]
+ beta x member + delta x ln(1 + LOTUS references), member = the candidate's scorer key is in a natural-product table
(LOTUS / NPAtlas: dmitriigluzdov/casmi26-natural-product-knowledge-table, natural_products.parquet). The data notebook
of the shared stack adds +1.0 for membership and 0.5 x ln(1 + references) to its PubChem channel.

Two stresses: the answer's documentation capped (as experiments/spec2mol_lowdoc.py), and the answer's membership
dropped with probability q (hidden answers may be less studied than the panel's well-known natural products; seeds
averaged).

usage: np_prior_lowdoc.py NATURAL_PRODUCTS_PARQUET
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import spec2mol_lowdoc as s2  # noqa: E402

if __name__ == "__main__":
    art = ROOT / "artifacts"
    np_tab = pd.read_parquet(sys.argv[1], columns=["scorer_key", "in_lotus", "in_npatlas", "n_references"])
    refs = {k: float(v) if pd.notna(v) else 0.0 for k, v in
            np_tab.assign(n_references=pd.to_numeric(np_tab.n_references, errors="coerce")).groupby("scorer_key").n_references.max().items()}
    data = s2.split_rows(art)
    ids = data["ids"]
    scores = json.load(open(art / "spec2mol_scores.json"))
    n_ans = sum(m in refs for m in ids)
    cand = [k for m in ids for k, _, _ in data["rows"][m] if k and k != m]
    print(f"answers in the NP table: {n_ans} of {len(ids)}; other shortlist candidates (rows): "
          f"{np.mean([k in refs for k in cand]):.3f} of {len(cand)}")
    caps = (None, 8, 6, 4)
    settings = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (0.5, 0.25)]
    for tag, name in (("c2", "class 2 (answer's spectra hidden)"), ("c1", "class-1 view")):
        sc0 = {c: {m: s2.c1_scores(m, data["rows"][m], data["hits"][tag][m], c) for m in ids} for c in caps}
        scale = s2.pooled_scale({m: list(sc0[None][m]) for m in ids}, scores)
        for s2m in (False, True):
            print(f"\n{name}, {'C1s (spec2mol 0.75)' if s2m else 'C1'}: MRR@25, answer doc as is / <= 8 / <= 6 / <= 4")
            for q in (0.0, 0.3, 0.6):
                for beta, delta in settings:
                    if q > 0 and beta == 0 and delta == 0:
                        continue
                    cells = []
                    for c in caps:
                        tot = []
                        for seed in (range(5) if q > 0 else range(1)):
                            rng = np.random.default_rng(seed)
                            drop = {m for m in ids if rng.random() < q}
                            rr = 0.0
                            for m in ids:
                                sc = dict(sc0[c][m])
                                for k in sc:
                                    if k in refs and not (k == m and m in drop):
                                        sc[k] += beta + delta * np.log1p(refs[k])
                                if s2m:
                                    keys = s2.fuse_scores(sc, scores.get(m), 0.75, scale)
                                else:
                                    keys = sorted(sc, key=lambda k: -sc[k])
                                rr += s2.rr(keys[:25], m)
                            tot.append(rr / len(ids))
                        cells.append(np.mean(tot))
                    print(f"  q {q:.1f} beta {beta:.2f} delta {delta:.2f} | " + " | ".join(f"{v:.4f}" for v in cells), flush=True)
