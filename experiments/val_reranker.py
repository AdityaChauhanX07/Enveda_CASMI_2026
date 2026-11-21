"""Learned isomer re-ranker for the 0.495 chain, trained and scored on a leak-free validation table.

Input: the output folder of a run of kozykappa's `enveda-casmi-2026-local-validation-set` notebook (the pinned 0.465
pipeline + the chain's two neural scorers on held-out molecules whose structures no trained part of the pipeline has
seen): val_candidates.parquet (final 25 + up to 40 outside candidates per molecule, channel ranks, ICEBERG / GLACIER,
neural scores, is_true), val_labels.parquet, submission_CUSTOM.csv (the chain's consensus order at that stage).

Baseline: the CUSTOM consensus order (rank 1 = 1/(3 + b) + 0.25/(3 + n), ranks 2+ with 0.75; b baseline rank,
n neural rank). Model: LightGBM lambdarank on within-list features, 5-fold cross-validation grouped by molecule (each
molecule is scored by a model that never saw it), repeated over seeds. Reported: MRR@25 of the baseline and of the
re-ranked lists over all molecules, per-molecule paired differences with a bootstrap interval, and the same with the
re-ranker limited to the final 25 (no outside candidates) and to same-formula members of the leader (rr8's scope).

usage: val_reranker.py VAL_OUT_DIR [--seeds=3] [--outside=1] [--min_lib=0.9]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OFFSET = 3.0


def rrf_custom(b_rank: np.ndarray, n_score: np.ndarray, w_top=0.25, w_mid=0.75) -> np.ndarray:
    """The chain's leader-safe consensus order (indices), as consensus.leader_safe_order."""
    n_rank = np.empty(len(n_score)); n_rank[np.argsort(-n_score, kind="stable")] = np.arange(1, len(n_score) + 1)
    top = 1 / (OFFSET + b_rank) + w_top / (OFFSET + n_rank)
    mid = 1 / (OFFSET + b_rank) + w_mid / (OFFSET + n_rank)
    lead = int(np.argsort(-top, kind="stable")[0])
    rest = [int(i) for i in np.argsort(-mid, kind="stable") if int(i) != lead]
    return np.array([lead] + rest)


def zin(v: pd.Series, g: pd.Series) -> pd.Series:
    v = pd.to_numeric(v, errors="coerce")
    gg = v.groupby(g)
    return ((v - gg.transform("mean")) / gg.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)).fillna(0.0)


def rank_in(v: pd.Series, g: pd.Series, ascending=False) -> pd.Series:
    v = pd.to_numeric(v, errors="coerce").fillna(-1e9 if not ascending else 1e9)
    return v.groupby(g).rank(ascending=ascending, method="first")


def features(c: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    g = c.molecule_id
    f = pd.DataFrame(index=c.index)
    b = pd.to_numeric(c.baseline_rank, errors="coerce").fillna(99)
    f["log_b"] = np.log(b)
    f["outside"] = (c.pool != "final25").astype(int) if "pool" in c else (b > 25).astype(int)
    for col in ("neural_score", "neural_pub", "neural_ours", "neural_cos", "neural_fp"):
        if col in c:
            f[f"z_{col}"] = zin(c[col], g)
            f[f"r_{col}"] = np.log(rank_in(c[col], g))
    if "neural_std" in c:
        f["neural_std"] = pd.to_numeric(c.neural_std, errors="coerce")
    for col in ("ice", "gl"):
        if col in c:
            v = pd.to_numeric(c[col], errors="coerce")
            f[f"{col}_na"] = v.isna().astype(int)
            f[f"z_{col}"] = zin(v, g)
    for col in ("main_rank", "prefusion_rank", "engine2_rank", "pubchem_rank"):
        if col in c:
            v = pd.to_numeric(c[col], errors="coerce").fillna(0)
            f[f"log_{col}"] = np.log(np.where(v > 0, v, 200))
            f[f"has_{col}"] = (v > 0).astype(int)
    if "library_max" in c:
        f["library_max"] = pd.to_numeric(c.library_max, errors="coerce").fillna(0)
    if "mass_ppm" in c:
        f["abs_ppm"] = pd.to_numeric(c.mass_ppm, errors="coerce").abs().fillna(50)
    if "cand_formula" in c:
        lead_formula = c.assign(_b=b).sort_values("_b").groupby("molecule_id").cand_formula.first()
        f["same_formula_as_lead"] = (c.cand_formula.values == c.molecule_id.map(lead_formula).values).astype(int)
        f["n_same_formula"] = c.groupby(["molecule_id", "cand_formula"]).smiles.transform("count").fillna(1)
    f["list_n"] = c.groupby("molecule_id").smiles.transform("count")
    return f, list(f.columns)


def mrr_of_orders(c: pd.DataFrame, score: np.ndarray, ids: list[str]) -> pd.Series:
    d = c.assign(_s=score).sort_values(["molecule_id", "_s"], ascending=[True, False], kind="stable")
    d = d[~d.duplicated(["molecule_id", "metric_key"]) | d.metric_key.isna()]
    d["_r"] = d.groupby("molecule_id").cumcount() + 1
    first = d[d.is_true].groupby("molecule_id")._r.min()
    rr = (1.0 / first).where(first <= 25, 0.0)
    return rr.reindex(ids).fillna(0.0)


def main() -> None:
    import lightgbm as lgb
    root = Path(sys.argv[1])
    args = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    seeds = int(args.get("seeds", 3)); use_out = int(args.get("outside", 1)); min_lib = float(args.get("min_lib", 0.9))
    c = pd.read_parquet(root / "val_candidates.parquet")
    lab = pd.read_parquet(root / "val_labels.parquet")
    ids = lab.molecule_id.astype(str).tolist()
    c["molecule_id"] = c.molecule_id.astype(str)
    c["is_true"] = c.is_true.astype(bool)
    print(f"candidates {len(c)} | molecules {c.molecule_id.nunique()} of {len(ids)} | truth in final 25 "
          f"{c[c.is_true & c.pool.eq('final25')].molecule_id.nunique()} | truth anywhere {c[c.is_true].molecule_id.nunique()}")
    if not use_out:
        c = c[c.pool.eq("final25")].copy()
    c = c.reset_index(drop=True)
    # baseline: the chain's consensus order on the final 25; outside candidates after them in their own order
    base = np.zeros(len(c))
    for mid, g in c.groupby("molecule_id", sort=False):
        fin = g[g.pool.eq("final25")].sort_values("baseline_rank", kind="stable")
        ns = pd.to_numeric(fin.neural_score, errors="coerce").fillna(-1e9).to_numpy()
        order = rrf_custom(pd.to_numeric(fin.baseline_rank).to_numpy(float), ns) if fin.neural_ok.astype(bool).all() else np.arange(len(fin))
        sc = np.empty(len(fin)); sc[order] = -np.arange(len(fin))
        base[fin.index.to_numpy()] = sc + 1000
        out = g[~g.pool.eq("final25")]
        base[out.index.to_numpy()] = -pd.to_numeric(out.baseline_rank, errors="coerce").fillna(999).to_numpy()
    c["base_score"] = base
    c["base_rank"] = c.groupby("molecule_id").base_score.rank(ascending=False, method="first")
    F, names = features(c)
    F["log_base_rank"] = np.log(c.base_rank)
    names.append("log_base_rank")
    rr_base = mrr_of_orders(c, base, ids)
    print(f"baseline (CUSTOM consensus) MRR@25 {rr_base.mean():.4f} | top-1 {(rr_base == 1).mean():.3f}")
    mols = np.array(sorted(c.molecule_id.unique()))
    protect = c.groupby("molecule_id").library_max.max() >= min_lib if "library_max" in c else pd.Series(False, index=mols)
    preds = np.zeros((seeds, len(c)))
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        fold = dict(zip(mols, rng.permutation(len(mols)) % 5))
        fk = c.molecule_id.map(fold).to_numpy()
        for k in range(5):
            tr, te = fk != k, fk == k
            dtr = c[tr]
            grp = dtr.groupby("molecule_id", sort=False).size().to_numpy()
            # LightGBM ranker needs rows sorted by group
            order = np.argsort(pd.Categorical(dtr.molecule_id, categories=dtr.molecule_id.unique()).codes, kind="stable")
            Xtr = F[tr].iloc[order]; ytr = dtr.is_true.astype(int).iloc[order]
            grp = pd.Series(dtr.molecule_id.iloc[order]).groupby(dtr.molecule_id.iloc[order], sort=False).size().to_numpy()
            m = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15,
                               min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=s, verbose=-1, n_jobs=4)
            m.fit(Xtr[names], ytr, group=grp)
            preds[s, te] = m.predict(F[te][names])
    pred = preds.mean(0)
    keep = c.molecule_id.map(protect).fillna(False).to_numpy()
    final = np.where(keep, base, pred)
    rr_new = mrr_of_orders(c, final, ids)
    diff = (rr_new - rr_base).to_numpy()
    boot = np.random.default_rng(0).choice(diff, (5000, len(diff))).mean(1)
    print(f"re-ranked MRR@25 {rr_new.mean():.4f} | top-1 {(rr_new == 1).mean():.3f} | paired diff {diff.mean():+.4f} "
          f"(95 % bootstrap {np.percentile(boot, 2.5):+.4f} .. {np.percentile(boot, 97.5):+.4f}) | improved {int((diff > 0).sum())} "
          f"hurt {int((diff < 0).sum())} | protected lists {int(protect.sum())}")
    # blends with the baseline order (the deployable form: how far to trust the model)
    zb = c.groupby("molecule_id").base_score.transform(lambda v: (v.rank() - v.rank().mean()) / (v.rank().std() + 1e-9)).to_numpy()
    zp = pd.Series(pred).groupby(c.molecule_id).transform(lambda v: (v - v.mean()) / (v.std() + 1e-9)).to_numpy()
    for a in (0.25, 0.5, 1.0, 2.0):
        sc = np.where(keep, base, zb + a * zp)
        r = mrr_of_orders(c, sc, ids)
        print(f"  blend baseline-rank z + {a:g} x model z: MRR@25 {r.mean():.4f} ({r.mean() - rr_base.mean():+.4f})")
    if "is_np" in lab:
        npm = set(lab.molecule_id[lab.is_np.astype(bool)].astype(str))
        sel = [m in npm for m in ids]
        print(f"natural products ({sum(sel)}): baseline {rr_base[sel].mean():.4f} -> {rr_new[sel].mean():.4f}")
    m_all = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=4)
    order = np.argsort(pd.Categorical(c.molecule_id, categories=c.molecule_id.unique()).codes, kind="stable")
    grp = c.molecule_id.iloc[order].groupby(c.molecule_id.iloc[order], sort=False).size().to_numpy()
    m_all.fit(F.iloc[order][names], c.is_true.astype(int).iloc[order], group=grp)
    imp = pd.Series(m_all.booster_.feature_importance("gain"), index=names).sort_values(ascending=False)
    print("feature gain (all-data model):", ", ".join(f"{k} {v:.0f}" for k, v in imp.head(12).items()))


if __name__ == "__main__":
    main()
