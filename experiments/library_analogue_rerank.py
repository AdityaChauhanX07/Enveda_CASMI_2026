"""Library-analogue support (library_analogue_feasibility.py output) as a re-ranking signal on panel class-2 lists.

(1) Support rates of answers by their documentation quartile vs wrong candidates (is the signal a panel artefact of
well-known natural products?). (2) Rule: inside the top 10, a candidate with support >= sigma moves up u places past
unsupported candidates (never past a supported one); lists as is. Answers' own spectra hidden (class 2).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casmi.panel import load_candidates

R = pd.read_parquet(ROOT / "artifacts" / "library_analogue_support.parquet").sort_values(["molecule", "rank0"])
c = load_candidates()
doc = c[c.is_truth].groupby("molecule").doc.max()
A = R[R.truth].copy(); A["doc"] = A.molecule.map(doc)
A["q"] = pd.qcut(A.doc, 4, labels=["doc q1 (lowest)", "q2", "q3", "q4 (highest)"])
W = R[~R.truth]
print("support >= 0.7: wrong candidates %.2f" % (W.support >= 0.7).mean())
for q, g in A.groupby("q", observed=True):
    print(f"  answers {q:16s} (doc {g.doc.min():.1f}-{g.doc.max():.1f}): n {len(g)}, with relative spectra {np.mean(g.n_rel_spec > 0):.2f}, support >= 0.7 {np.mean(g.support >= 0.7):.2f}")
N = 250
def rr(truth):
    t = np.flatnonzero(truth); return 1 / (t[0] + 1) if len(t) else 0.0
print("rule (inside top 10)            | class-2 delta | helped / hurt")
for sigma in (0.6, 0.7, 0.8):
    for u in (1, 2, 3, 9):
        d, h, hu = 0.0, 0, 0
        for m, g in R.groupby("molecule", sort=False):
            truth = g.truth.to_numpy(); sup = (g.support.to_numpy() >= sigma)
            order = list(range(len(g)))
            for j in range(len(order)):              # stable: each supported candidate climbs past up to u unsupported ones
                if not sup[order[j]]:
                    continue
                k = j
                while k > 0 and j - k < u and not sup[order[k - 1]]:
                    k -= 1
                order.insert(k, order.pop(j))
            a, b = rr(truth), rr(truth[order])
            d += b - a; h += b > a; hu += b < a
        print(f"support >= {sigma} up {u if u < 9 else 'top'}               | {d / N:+.4f}       | {h} / {hu}")

# (3) never past rank 1, and combined with the library overlay (contradicted candidates down 3 first, tau 0.5, >= 3
# same-adduct spectra; flags from artifacts/lib_discrim_26.parquet), still inside the top 10
sys.path.insert(0, str(ROOT / "experiments"))
from casmi import library_overlay as lo
L = pd.read_parquet(ROOT / "artifacts" / "lib_discrim_26.parquet")
P = pd.read_parquet(ROOT / "artifacts" / "panel_lists_lib.parquet")
conn = {(r.molecule, r.rank0): r.connectivity for r in P.itertuples()}
flag = {(r.molecule, r.cand) for r in L.itertuples() if (r.n_add or 0) >= 3 and (0 if pd.isna(r.sim_add) else r.sim_add) < 0.5 and not r.truth}


def climb(order, sup, u, floor):
    order = list(order)
    for j in range(len(order)):
        if not sup[order[j]]:
            continue
        k = j
        while k > floor and j - k < u and not sup[order[k - 1]]:
            k -= 1
        order.insert(k, order.pop(j))
    return order


print("combined / guarded rules (inside top 10)             | class-2 delta | helped / hurt")
for lib_first in (False, True):
    for sigma, u, floor in ((0.7, 1, 1), (0.7, 2, 1), (0.7, 2, 0), (0.6, 2, 1)):
        if not lib_first and floor == 0:
            continue
        d, h, hu = 0.0, 0, 0
        for m, g in R.groupby("molecule", sort=False):
            truth = g.truth.to_numpy(); sup = (g.support.to_numpy() >= sigma)
            order = list(range(len(g)))
            if lib_first:
                f = [(m, conn.get((m, int(r)), "")) in flag for r in g.rank0]
                order = list(lo.reorder(f, 3))
            order = climb(order, sup, u, floor)
            a = rr(truth); b = rr(truth[order])
            if lib_first:
                a = rr(truth[list(lo.reorder([(m, conn.get((m, int(r)), "")) in flag for r in g.rank0], 3))])
            d += b - a; h += b > a; hu += b < a
        print(f"{'after library overlay, ' if lib_first else ''}support >= {sigma} up {u}{', not past rank 1' if floor else ''}".ljust(53) + f"| {d / N:+.4f}       | {h} / {hu}")
