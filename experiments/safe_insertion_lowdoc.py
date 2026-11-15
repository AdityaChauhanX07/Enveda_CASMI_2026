"""Can products be inserted without touching the fork's leader, when correct answers are poorly documented?

Dumps state_<tag>_* (default t7 = a7; t8a = after the library overlay). Class-2 change against the original
fork-like lists. Class-2 lists: the answer keeps its list position, but its gate score z + 0.4 * doc is
recomputed with doc capped at d (the a7 failure: a correct answer whose documentation is low). Class-3 lists as
dumped. Rules: min_slot (never above rank min_slot + 1) and max_insert. Net at 10 / 25 % class 3.
"""
import pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.edit_overlay import merge_sources
from casmi.panel import load_candidates
import insertion_replay as ir

c = load_candidates()
doc_ans = c[c.is_truth].groupby("molecule").doc.max().to_dict()
tag = sys.argv[1] if len(sys.argv) > 1 else "t7"   # t8a: lists after the library overlay (tau 0.5)
c2 = pickle.load(open(ROOT / "artifacts" / f"state_{tag}_c2.pkl", "rb")); c3 = pickle.load(open(ROOT / "artifacts" / f"state_{tag}_c3.pkl", "rb"))
from library_overlay_panel import panel_lists
from casmi.metric import ik14
_, _, sub = panel_lists()
orig = {m: [ik14(x) for x in s_.split(";")] for m, s_ in zip(sub.molecule_id, sub.smiles)}   # class-2 baseline: the fork-like lists
merged = lambda st: dict(st, scored=[(g, k, m) for g, _, k, m in merge_sources([(g, None, k, m) for g, k, m in st["scored"]], 0.0)])
N = 250
for d in (None, 6, 4):
    for rule in ({}, {"min_slot": 1}, {"min_slot": 1, "max_insert": 1}, {"min_slot": 2}, {"min_slot": 2, "max_insert": 1}):
        d2 = 0.0; hurt = 0
        for m, st in c2.items():
            st = merged(st); ck = st["cand_keys"]
            top = list(st["top"])
            if d is not None and m in ck[:3]:
                i = ck.index(m)
                if i < len(top):
                    top[i] -= 0.4 * max(0.0, doc_ans.get(m, 0.0) - d)
            x = ir.rr(ir.insert(dict(st, top=top), rule), m) - ir.rr(orig[m], m)
            d2 += x; hurt += x < 0
        r3 = sum(ir.rr(ir.insert(merged(st), rule), m) for m, st in c3.items())
        e2, e3 = d2 / N, r3 / N
        name = "a7 rule" if not rule else ", ".join(f"{k} {v}" for k, v in rule.items())
        print(f"answer doc {'as is' if d is None else '<= ' + str(d):>6} | {name:24s} | class-2 {e2:+.4f} ({hurt} hurt) | class-3 {e3:.4f} | net 10 % {0.9 * e2 + 0.1 * e3:+.4f}, 25 % {0.75 * e2 + 0.25 * e3:+.4f}")
