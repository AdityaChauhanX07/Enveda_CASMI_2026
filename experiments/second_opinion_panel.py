"""Second-opinion re-rank of a finished top-25 list on the panel.

Stand-in for the fork: a 2-model fold-safe ensemble (like the fork's FPNet A+B, which never saw the hidden test
molecules) + 0.4 doc builds the list. Second opinion: the other fold-safe models' z + w doc over the same 25
members. New order: -alpha * log(1 + rank) + (z_2nd + w doc). alpha large keeps the list as it is."""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.panel import load_candidates, zscore  # noqa: E402
from panel_fp_matrix import PanelFP, score_frame  # noqa: E402

if __name__ == "__main__":
    c = load_candidates()
    L = np.load(ROOT / "artifacts" / "panel_logits_foldsafe.npz")
    keys = [str(k) for k in L["keys"]]
    zk = {k: 0.5 * (L[f"single{k}"].astype(np.float64) + L[f"merged{k}"].astype(np.float64)) for k in range(5)}
    fp = PanelFP()
    raw = {k: score_frame(c, dict(zip(keys, zk[k])), fp) for k in range(5)}
    doc = c.doc.to_numpy()
    res = []
    for proxy in [(0, 1), (2, 3), (3, 4), (1, 4)]:
        rest = [k for k in range(5) if k not in proxy]
        zp = zscore(c.assign(_s=np.mean([raw[k] for k in proxy], 0)), "_s")
        zo = zscore(c.assign(_s=np.mean([raw[k] for k in rest], 0)), "_s")
        d = c.assign(s0=zp + 0.4 * doc, zo=zo).sort_values(["molecule", "s0"], ascending=[True, False], kind="stable")
        d = d[~d.duplicated(["molecule", "ident"])]
        d["r"] = d.groupby("molecule").cumcount()
        d = d[d.r < 25].copy()
        def mrr(score):
            e = d.assign(s=score).sort_values(["molecule", "s"], ascending=[True, False], kind="stable")
            e["rr"] = e.groupby("molecule").cumcount()
            f = e[e.is_truth].groupby("molecule").rr.min()
            return float((1.0 / (f + 1)).sum() / 250)
        base = mrr(-d.r.to_numpy().astype(float))
        row = dict(proxy=proxy, base=round(base, 4))
        for w in (0.3, 0.4):
            for a in (0.5, 1.0, 2.0, 3.0, 5.0):
                row[f"w{w}_a{a}"] = round(mrr(-a * np.log1p(d.r.to_numpy()) + d.zo.to_numpy() + w * d.doc.to_numpy()) - base, 4)
        res.append(row)
        print(row, flush=True)
    t = pd.DataFrame(res).drop(columns=["proxy", "base"]).mean().round(4)
    print("mean gain over the 4 splits:"); print(t.to_string())
