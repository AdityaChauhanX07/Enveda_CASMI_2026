"""Panel PubChem channel with each fold-safe FPNet and their ensembles (all five never saw the panel molecules).

Logits come from the Kaggle job `casmi26-fs-panel-logits` (artifacts/panel_logits_foldsafe.npz: per model the mean
single-spectrum and mean merged logits; the engine's z is their average). Scores use the panel fingerprint matrix
(experiments/panel_fp_matrix.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.panel import PANEL, load_candidates, mrr, reciprocal_ranks, zscore  # noqa: E402
from panel_fp_matrix import PanelFP, score_frame  # noqa: E402


def zcol(c, s):
    c = c.assign(_s=s)
    return zscore(c, "_s")


if __name__ == "__main__":
    c = load_candidates()
    L = np.load(ROOT / "artifacts" / "panel_logits_foldsafe.npz")
    keys = [str(k) for k in L["keys"]]
    fp = PanelFP()
    doc = c.doc.to_numpy()
    raw, zs = {}, {}
    for k in range(5):
        z = 0.5 * (L[f"single{k}"].astype(np.float32) + L[f"merged{k}"].astype(np.float32))
        raw[k] = score_frame(c, dict(zip(keys, z)), fp)
        zs[k] = zcol(c, raw[k])
    ref = zscore(c, "fold0_full")
    print(f"cached fold0: alone {mrr(c, ref):.4f}  +0.4 doc {mrr(c, ref + 0.4 * doc):.4f}")
    for k in range(5):
        print(f"fold{k}: alone {mrr(c, zs[k]):.4f}  +0.4 doc {mrr(c, zs[k] + 0.4 * doc):.4f}")
    ens_logit = np.mean([raw[k] for k in range(5)], axis=0)       # mean logits = mean raw scores (linear)
    ze = zcol(c, ens_logit)
    zm = np.mean([zs[k] for k in range(5)], axis=0)
    z14 = zcol(c, np.mean([raw[k] for k in range(1, 5)], axis=0))
    for name, s in (("mean of 5 (logits)", ze), ("mean of 5 z-scores", zm), ("folds 1-4", z14)):
        print(f"{name}: alone {mrr(c, s):.4f}", " ".join(f"+{w} doc {mrr(c, s + w * doc):.4f}" for w in (0.3, 0.4, 0.5, 0.6)))
    base = reciprocal_ranks(c, zs[0] + 0.4 * doc); new = reciprocal_ranks(c, ze + 0.4 * doc)
    j = base.index.union(new.index)
    b, n = base.reindex(j, fill_value=0), new.reindex(j, fill_value=0)
    print(f"paired, +0.4 doc: up {(n > b).sum()} down {(n < b).sum()} | sum diff {float((n - b).sum()) / 250:.4f}")
    rng = np.random.default_rng(0); d = (n - b).to_numpy(); d = np.concatenate([d, np.zeros(250 - len(d))])
    boots = [rng.choice(d, 250).mean() for _ in range(5000)]
    print("bootstrap 95% interval of the gain:", np.percentile(boots, [2.5, 97.5]).round(4))
