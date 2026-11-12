"""How should a molecule's spectra be combined into one fingerprint prediction? (panel, five fold-safe FPNets)

Per-spectrum and per-polarity merged logits come from the Kaggle CPU job `casmi26-fs-panel-spectra`. The engine's
rule is z = 0.5 * (mean single + mean merged); variants re-weight spectra by quality or pool differently."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.panel import load_candidates, mrr, zscore  # noqa: E402
from panel_fp_matrix import PanelFP, score_frame  # noqa: E402

D = ROOT / "artifacts" / "fs_spec"

if __name__ == "__main__":
    L = np.load(D / "panel_spectrum_logits.npz")
    meta = pd.read_parquet(D / "panel_spectrum_meta.parquet")
    mmeta = pd.read_parquet(D / "panel_merged_meta.parquet")
    S = np.mean([L[f"single{k}"].astype(np.float32) for k in range(5)], axis=0)     # 5-model mean per spectrum
    M = np.mean([L[f"merged{k}"].astype(np.float32) for k in range(5)], axis=0)
    c = load_candidates(); fp = PanelFP(); doc = c.doc.to_numpy()
    si = meta.groupby("molecule").indices; mi = mmeta.groupby("molecule").indices
    def run(name, fn):
        z = {m: fn(m) for m in si}
        sc = zscore(c.assign(_s=score_frame(c, z, fp)), "_s")
        print(f"{name:42s} alone {mrr(c, sc):.4f} | +0.3 doc {mrr(c, sc + 0.3 * doc):.4f} | +0.4 doc {mrr(c, sc + 0.4 * doc):.4f}", flush=True)
    w_bp = np.log10(meta.base_peak.fillna(meta.base_peak.median()).clip(lower=10).to_numpy())
    npk = meta.n_prep.to_numpy().astype(float)
    def wmean(m, w):
        i = si[m]; ww = w[i]; return (S[i] * ww[:, None]).sum(0) / ww.sum()
    run("engine: 0.5 single + 0.5 merged", lambda m: 0.5 * (S[si[m]].mean(0) + M[mi[m]].mean(0)))
    run("single only", lambda m: S[si[m]].mean(0))
    run("merged only", lambda m: M[mi[m]].mean(0))
    run("0.7 single + 0.3 merged", lambda m: 0.7 * S[si[m]].mean(0) + 0.3 * M[mi[m]].mean(0))
    run("0.3 single + 0.7 merged", lambda m: 0.3 * S[si[m]].mean(0) + 0.7 * M[mi[m]].mean(0))
    run("single weighted by log10 base peak, + merged", lambda m: 0.5 * (wmean(m, w_bp) + M[mi[m]].mean(0)))
    run("single weighted by base peak^0.5, + merged", lambda m: 0.5 * (wmean(m, np.sqrt(np.nan_to_num(meta.base_peak.to_numpy(), nan=1e4))) + M[mi[m]].mean(0)))
    run("single weighted by #peaks, + merged", lambda m: 0.5 * (wmean(m, np.maximum(npk, 1)) + M[mi[m]].mean(0)))
    run("max over singles + merged (0.5/0.5)", lambda m: 0.5 * (S[si[m]].max(0) + M[mi[m]].mean(0)))
    run("mean prob (single+merged) -> logit", lambda m: np.log(np.clip(np.concatenate([1 / (1 + np.exp(-S[si[m]])), 1 / (1 + np.exp(-M[mi[m]]))]).mean(0), 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - np.concatenate([1 / (1 + np.exp(-S[si[m]])), 1 / (1 + np.exp(-M[mi[m]]))]).mean(0), 1e-6, 1)))
    run("sum (not mean) of single logits + merged", lambda m: 0.5 * (S[si[m]].sum(0) + M[mi[m]].mean(0)))
