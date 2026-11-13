"""Replay the overlay's insertion step (pass 3 of casmi/edit_overlay.py) on dumped panel states under other rules.

States come from `edit_overlay_panel.py --ensemble --relatives=1 --dump_state=artifacts/state`: state_c2.pkl (every
list keeps its answer) and state_c3.pkl (every answer removed). Per molecule: scores of the top-3 list members,
the list's metric keys and every scored product (gate score, key, source, routes, documentation, z). The panel's
molecule id is the answer's metric key, so each rule's class-2 cost and class-3 gain are measured on all molecules.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N = 250


def insert(st: dict, rule: dict, max_rank: int = 3, max_insert: int = 3) -> list:
    top, ck = st["top"], st["cand_keys"]
    scored = sorted(st["scored"], key=lambda x: -x[0])
    if not scored:
        return ck
    slots = {}
    for g, key, meta in scored[: rule.get("max_insert", max_insert)]:
        d = rule.get("margin_all", 0.0)
        beaten = [i for i, si in enumerate(top) if si >= g - d]
        r = (max(beaten) + 1) if beaten else 0
        if r == 0 and top:                                   # displacing the current leader
            weak = False
            if "lead_routes" in rule and meta["routes"] < rule["lead_routes"]:
                weak = True
            if "lead_margin" in rule and g < top[0] + rule["lead_margin"]:
                weak = True
            if "lead_either" in rule:                        # needs routes >= R or a margin >= M
                R, M = rule["lead_either"]
                weak = not (meta["routes"] >= R or g >= top[0] + M)
            if "lead_strong" in rule and top[0] >= rule["lead_strong"]:
                weak = True
            if rule.get("lead_src") and meta["src"] not in rule["lead_src"]:
                weak = True
            if weak:
                r = 1
        if meta["src"] == "nb" and "nb_min_slot" in rule:
            r = max(r, rule["nb_min_slot"])
        if r < max_rank:
            slots.setdefault(r, []).append(key)
    if not slots:
        return ck
    out = []
    for i, k in enumerate(ck):
        out.extend(slots.get(i, [])); out.append(k)
    seen, final = set(), []
    for k in out:
        if k not in seen:
            seen.add(k); final.append(k)
    return final[:25]


def rr(keys: list, truth: str) -> float:
    return 1.0 / (keys.index(truth) + 1) if truth in keys[:25] else 0.0


def evaluate(c2: dict, c3: dict, rule: dict, mols=None) -> dict:
    mols = mols if mols is not None else sorted(set(c2) | set(c3))
    d2 = [(rr(insert(c2[m], rule), m) - rr(c2[m]["cand_keys"], m)) if m in c2 else 0.0 for m in mols]
    r3 = [rr(insert(c3[m], rule), m) if m in c3 else 0.0 for m in mols]
    n = N * len(mols) / len(set(c2) | set(c3))
    return dict(c2=sum(d2) / n, c3=sum(r3) / n, hurt=sum(x < 0 for x in d2), helped=sum(x > 0 for x in d2))


if __name__ == "__main__":
    c2 = pickle.load(open(ROOT / "artifacts" / "state_c2.pkl", "rb"))
    c3 = pickle.load(open(ROOT / "artifacts" / "state_c3.pkl", "rb"))
    mols = sorted(set(c2) | set(c3))
    print(f"molecules in state: class-2 {len(c2)}, class-3 {len(c3)}")
    # who hurts class 2 under the shipped rule
    print("hurt class-2 molecules under the shipped rule:")
    for m in mols:
        if m not in c2:
            continue
        before, after = rr(c2[m]["cand_keys"], m), rr(insert(c2[m], {}), m)
        if after < before:
            st = c2[m]; top = st["top"]
            above = [x for x in sorted(st["scored"], key=lambda x: -x[0])[:3]]
            desc = "; ".join(f"{x[2]['src']} g={x[0]:.2f} routes={x[2]['routes']} doc={x[2]['doc']:.1f} z={x[2]['z']:.2f}" for x in above)
            print(f"  {m}: rank {int(round(1 / before))} -> {int(round(1 / after)) if after else '>25'} | top {[round(t, 2) for t in top]} | {desc}")
    rules = {
        "shipped": {},
        "lead needs routes>=2": dict(lead_routes=2),
        "lead needs routes>=3": dict(lead_routes=3),
        "lead needs margin 0.5": dict(lead_margin=0.5),
        "lead needs margin 1": dict(lead_margin=1.0),
        "lead needs margin 2": dict(lead_margin=2.0),
        "lead needs routes>=2 or margin 1": dict(lead_either=(2, 1.0)),
        "lead needs routes>=2 or margin 2": dict(lead_either=(2, 2.0)),
        "lead kept if leader >= 4": dict(lead_strong=4.0),
        "lead kept if leader >= 6": dict(lead_strong=6.0),
        "lead only by relatives": dict(lead_src={"rel"}),
        "neighbour products slot >= 1": dict(nb_min_slot=1),
        "margin 0.5 for every slot": dict(margin_all=0.5),
        "margin 1 for every slot": dict(margin_all=1.0),
        "one insertion": dict(max_insert=1),
    }
    halves = [mols[0::2], mols[1::2]]
    print(f"{'rule':36s} {'class-2':>8s} {'hurt':>4s} {'help':>4s} {'class-3':>8s} {'net10':>7s} {'net25':>7s} | halves net25")
    for name, rule in rules.items():
        e = evaluate(c2, c3, rule)
        h = [evaluate(c2, c3, rule, hm) for hm in halves]
        net = lambda x, s: (1 - s) * x["c2"] + s * x["c3"]
        print(f"{name:36s} {e['c2']:+8.4f} {e['hurt']:4d} {e['helped']:4d} {e['c3']:8.4f} {net(e, .1):+7.4f} {net(e, .25):+7.4f} | "
              f"{net(h[0], .25):+.4f} / {net(h[1], .25):+.4f}")
