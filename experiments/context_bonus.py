"""Class-3-aware insertion: give a molecule's generated candidates a bonus only when its list looks like class 3.

Signals known at insertion time (task-3 dumps artifacts/state_t3_c2.pkl / _c3.pkl):
  lead = gate score of the fork's #1 (class 3: lower, AUC 0.77)
  gap  = best product score - best top-3 member score (class 3: higher, AUC 0.83)
Rules: weak-leader bonus  lam * max(0, tau - lead);  close-gap bonus  beta if gap >= theta.
Tuned on one half of the panel and scored on the other (both directions)."""
from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.edit_overlay import merge_sources  # noqa: E402
import insertion_replay as ir  # noqa: E402


def with_bonus(st: dict, rule: dict) -> dict:
    merged = merge_sources([(g, None, k, m) for g, k, m in st["scored"]], 0.0)
    if not merged:
        return dict(st, scored=[])
    top = st["top"]
    lead = top[0] if top else -np.inf
    gap = max(x[0] for x in merged) - (max(top) if top else -np.inf)
    b = 0.0
    if rule.get("kind") == "lead":
        b = rule["lam"] * max(0.0, rule["tau"] - lead)
    elif rule.get("kind") == "gap":
        b = rule["beta"] if gap >= rule["theta"] else 0.0
    return dict(st, scored=[(g + b, k, m) for g, _, k, m in merged])


def score(c2, c3, rule, mols):
    t2 = {m: with_bonus(c2[m], rule) for m in mols}; t3 = {m: with_bonus(c3[m], rule) for m in mols}
    return ir.evaluate(t2, t3, {}, mols)


if __name__ == "__main__":
    c2 = pickle.load(open(ROOT / "artifacts" / "state_t3_c2.pkl", "rb"))
    c3 = pickle.load(open(ROOT / "artifacts" / "state_t3_c3.pkl", "rb"))
    mols = sorted(set(c2) & set(c3)); halves = [mols[0::2], mols[1::2]]
    net = lambda e, s: (1 - s) * e["c2"] + s * e["c3"]
    rules = [{"kind": "none"}]
    rules += [dict(kind="lead", tau=t, lam=l) for t, l in itertools.product((5.0, 6.0, 7.0, 8.0), (0.25, 0.5, 1.0))]
    rules += [dict(kind="gap", theta=th, beta=b) for th, b in itertools.product((-2.0, -1.5, -1.0, -0.5), (0.5, 1.0, 1.5, 2.0))]
    for s in (0.10, 0.25):
        print(f"--- class-3 share {int(s * 100)} %: rule picked on one half, scored on the other")
        for a, bh in ((0, 1), (1, 0)):
            best = max(rules, key=lambda r: net(score(c2, c3, r, halves[a]), s))
            base_b, best_b = score(c2, c3, rules[0], halves[bh]), score(c2, c3, best, halves[bh])
            print(f"  fit half {a}: picked {best} -> held-out half net {net(base_b, s):+.4f} -> {net(best_b, s):+.4f} "
                  f"(class-2 {base_b['c2']:+.4f} -> {best_b['c2']:+.4f}, class-3 {base_b['c3']:.4f} -> {best_b['c3']:.4f})")
    print("--- full panel, selected rules")
    for r in rules[:1] + [r for r in rules if r.get("kind") == "gap" and r["beta"] in (1.0, 1.5) and r["theta"] in (-1.5, -1.0)] + \
             [r for r in rules if r.get("kind") == "lead" and r["lam"] == 0.5]:
        e = score(c2, c3, r, mols); h = [score(c2, c3, r, hm) for hm in halves]
        print(f"  {str(r):52s} class-2 {e['c2']:+.4f} (hurt {e['hurt']}) class-3 {e['c3']:.4f} | net10 {net(e, .1):+.4f} net25 {net(e, .25):+.4f} "
              f"| halves net25 {net(h[0], .25):+.4f} / {net(h[1], .25):+.4f}")
