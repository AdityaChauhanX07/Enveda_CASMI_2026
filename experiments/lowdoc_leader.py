"""Why a7 lost on the leaderboard: sensitivity of its insertion step to the documentation of a correct leader.

Class-2 lists of the a7 dumps (artifacts/state_t7_c2.pkl) whose answer is first: the leader's gate score
z + 0.4 * doc is recomputed with the answer's documentation lowered to d, and the insertion step is replayed.
The panel's answers are unusually well documented (median doc 13.9), which hid how much the gate leans on it.
"""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.edit_overlay import merge_sources
import insertion_replay as ir
from casmi.panel import load_candidates
c = load_candidates()
doc_ans = c[c.is_truth].groupby("molecule").doc.max().to_dict()
c2 = pickle.load(open(ROOT / "artifacts" / "state_t7_c2.pkl", "rb"))
lead = [m for m, st in c2.items() if st["cand_keys"] and st["cand_keys"][0] == m and st["top"]]
print("class-2 lists with the answer first:", len(lead), "| their answer doc: median %.1f, q10 %.1f, q90 %.1f" % tuple(np.percentile([doc_ans.get(m, np.nan) for m in lead], [50, 10, 90])))
def merged(st):
    return dict(st, scored=[(g, k, m) for g, _, k, m in merge_sources([(g, None, k, m) for g, k, m in st["scored"]], 0.0)])
for d in (None, 8, 6, 4, 2, 0):
    loss, hurt = 0.0, 0
    for m in lead:
        st = merged(c2[m])
        if d is not None:
            top = list(st["top"]); top[0] = top[0] - 0.4 * max(0.0, doc_ans.get(m, 0.0) - d)
            st = dict(st, top=top)
        r = ir.rr(ir.insert(st, {}), m)
        loss += 1.0 - r; hurt += r < 1.0
    print(f"answer doc {'as is' if d is None else '<= ' + str(d):>7}: correct leaders displaced {hurt:3d} of {len(lead)} | MRR lost on these lists {loss / len(lead):.3f} | per molecule of the panel {loss / 250:.4f}")
