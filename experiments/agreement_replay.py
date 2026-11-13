"""Task 3: does agreement between the two generators pick out right answers?

States from `edit_overlay_panel.py --relatives=1 --rel_exclude_nb=0 --dump_state=artifacts/state_t3` hold, per molecule,
the neighbour-edit products and the PubChem-relative products including structures both proposed. Rules:
  current  - drop PubChem-relative entries the neighbour edits also proposed (what rel_exclude_nb=1 does)
  merge    - one entry per structure, the better score
  merge+b  - the same, plus b when both generators proposed it (casmi/edit_overlay.py merge_sources)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.edit_overlay import merge_sources  # noqa: E402
import insertion_replay as ir  # noqa: E402


def transform(st: dict, rule) -> dict:
    sc = st["scored"]                                       # (g, key, meta)
    if rule == "current":
        nb = {k for _, k, m in sc if m["src"] == "nb"}
        sc = [x for x in sc if not (x[2]["src"] == "rel" and x[1] in nb)]
    else:
        merged = merge_sources([(g, None, k, m) for g, k, m in sc], rule)
        sc = [(g, k, m) for g, _, k, m in merged]
    return dict(st, scored=sc)


if __name__ == "__main__":
    c2 = pickle.load(open(ROOT / "artifacts" / "state_t3_c2.pkl", "rb"))
    c3 = pickle.load(open(ROOT / "artifacts" / "state_t3_c3.pkl", "rb"))
    for name, S in (("class-2", c2), ("class-3", c3)):
        agree_mols, agree_keys, ans = 0, 0, 0
        for m, st in S.items():
            nb = {k for _, k, x in st["scored"] if x["src"] == "nb"}
            rel = {k for _, k, x in st["scored"] if x["src"] == "rel"}
            both = nb & rel
            agree_mols += bool(both); agree_keys += len(both); ans += m in both
        print(f"{name}: molecules with agreement {agree_mols}, agreed structures {agree_keys}, answer among them {ans}")
    mols = sorted(set(c2) | set(c3)); halves = [mols[0::2], mols[1::2]]
    net = lambda x, s: (1 - s) * x["c2"] + s * x["c3"]
    for rule in ("current", 0.0, 0.5, 1.0, 2.0, 3.0):
        t2 = {m: transform(st, rule) for m, st in c2.items()}
        t3 = {m: transform(st, rule) for m, st in c3.items()}
        e = ir.evaluate(t2, t3, {}); h = [ir.evaluate(t2, t3, {}, hm) for hm in halves]
        name = rule if rule == "current" else f"merge + {rule}"
        print(f"{name:12s} class-2 {e['c2']:+.4f} (hurt {e['hurt']}) class-3 {e['c3']:.4f} | net10 {net(e, .1):+.4f} "
              f"net25 {net(e, .25):+.4f} | halves net25 {net(h[0], .25):+.4f} / {net(h[1], .25):+.4f}")
