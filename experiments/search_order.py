"""Task 4: which molecules should get the time-budgeted PubChem-relative search first?

Panel dumps from task 3 (artifacts/state_t3_c2.pkl: every answer kept, state_t3_c3.pkl: every answer removed). Each draw
marks a random 25 % of molecules as class 3 (their removed-answer state) and the rest as class 2; molecules are ordered
by a rule that only uses what the overlay knows before the search; the first `budget` share gets PubChem-relative
products, the rest keep their neighbour-edit products only. Score = change in total reciprocal rank / 250."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.edit_overlay import merge_sources  # noqa: E402
import insertion_replay as ir  # noqa: E402


def after(st: dict, searched: bool) -> list:
    sc = st["scored"] if searched else [x for x in st["scored"] if x[2]["src"] == "nb"]
    merged = merge_sources([(g, None, k, m) for g, k, m in sc], 0.0)
    return ir.insert(dict(st, scored=[(g, k, m) for g, _, k, m in merged]), {})


def features(st: dict) -> dict:
    top = list(st["top"]) + [-np.inf] * (3 - len(st["top"]))
    nb = [g for g, _, m in st["scored"] if m["src"] == "nb"]
    finite = [t for t in top if np.isfinite(t)]
    return dict(lead=top[0], best=max(finite) if finite else -np.inf, gap=(top[0] - top[1]) if np.isfinite(top[1]) else 9.0,
                nb_best=max(nb) if nb else -np.inf, nb_n=len(nb))


if __name__ == "__main__":
    c2 = pickle.load(open(ROOT / "artifacts" / "state_t3_c2.pkl", "rb"))
    c3 = pickle.load(open(ROOT / "artifacts" / "state_t3_c3.pkl", "rb"))
    mols = sorted(set(c2) & set(c3))
    # per molecule and situation: reciprocal rank before, after with / without the relative search
    R = {}
    for m in mols:
        for cls, S in ((2, c2), (3, c3)):
            st = S[m]
            R[(m, cls)] = dict(base=ir.rr(st["cand_keys"], m), nb=ir.rr(after(st, False), m), full=ir.rr(after(st, True), m),
                               f=features(st))
    rules = {
        "weakest leader first (current)": lambda f: f["lead"],
        "weakest best top-3 first": lambda f: f["best"],
        "smallest leader gap first": lambda f: f["gap"],
        "weakest leader, no neighbour products first": lambda f: (f["nb_n"] > 0, f["lead"]),
        "leader minus best neighbour product": lambda f: f["lead"] - max(f["nb_best"], f["lead"] - 3.0),
        "random": None,
    }
    budgets, draws, share = (0.4, 0.5, 0.6, 1.0), 20, 0.25
    rng = np.random.default_rng(0)
    plans = [set(rng.choice(mols, int(round(share * len(mols))), replace=False)) for _ in range(draws)]
    res = {name: {b: [] for b in budgets} for name in list(rules) + ["oracle"]}
    for d, cls3 in enumerate(plans):
        rows = []
        for m in mols:
            r = R[(m, 3 if m in cls3 else 2)]
            rows.append((m, r))
        gain_full = {m: r["full"] - r["nb"] for m, r in rows}
        for name, key in list(rules.items()) + [("oracle", None)]:
            if name == "random":
                order = list(np.random.default_rng(100 + d).permutation([m for m, _ in rows]))
            elif name == "oracle":
                order = sorted(gain_full, key=lambda m: -gain_full[m])
            else:
                order = [m for m, r in sorted(rows, key=lambda x: key(x[1]["f"]))]
            pos = {m: i for i, m in enumerate(order)}
            for b in budgets:
                k = int(round(b * len(order)))
                tot = sum((r["full"] if pos[m] < k else r["nb"]) - r["base"] for m, r in rows)
                res[name][b].append(tot / 250)
    print(f"{'order':46s} " + " ".join(f"budget {int(b * 100):3d}%" for b in budgets) + "   (mean gain over 20 draws, 25 % class 3)")
    for name, by in res.items():
        print(f"{name:46s} " + " ".join(f"{np.mean(by[b]):+11.4f}" for b in budgets))
    full = np.mean(res["oracle"][1.0])
    print(f"share of the full-coverage gain kept at 50 % budget: current {np.mean(res['weakest leader first (current)'][0.5]) / full:.0%}, "
          f"random {np.mean(res['random'][0.5]) / full:.0%}, oracle {np.mean(res['oracle'][0.5]) / full:.0%}")
