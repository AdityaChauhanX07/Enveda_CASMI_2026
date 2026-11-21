"""Precursor survival prior: does how much precursor survives at each collision energy separate isomers?

Energy-resolved MS tells isomers apart by their breakdown curves: an O-glycoside loses its sugar at low energy, a
C-glycoside or an ether survives. Each spectrum keeps a residual precursor peak (56 % of the panel's spectra, falling
with energy), so every test molecule carries a few points of its own breakdown curve.

Model: log10(precursor share + 1e-3) per spectrum, LightGBM regression on the structure descriptors of
adduct_prior.py + collision energy + ion type + precursor m/z, trained on enveda-180 spectra (same instrument) of the
test's ion types; panel answers and every structure within Morgan Tanimoto >= --excl of one left out.
Score per candidate: -sum over the molecule's spectra of (observed - predicted)^2 / (2 sd^2), sd = held-out residual sd.
Lists re-ranked like adduct_prior.py; with --adduct=REPORT_DIR the ion-type log-likelihood is added as well.

usage: survival_prior.py WORK_DIR [--excl=0.6] [--max_struct=80000] [--lams=...]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import adduct_prior as ap  # noqa: E402

TYPES = ["[M+H]+", "[M-H]-", "[M+CH2O2-H]-", "[M+NH4]+", "[M+Na]+"]


def prec_share(mzs, its, prec, tol=0.01) -> float:
    mz = np.asarray(mzs, np.float64); it = np.asarray(its, np.float64)
    tot = it.sum()
    if not len(mz) or tot <= 0:
        return np.nan
    return float(it[np.abs(mz - prec) < tol].sum() / tot)


def mean_ce(x) -> float:
    try:
        v = [float(e) for e in x]
        return float(np.mean(v)) if v else np.nan
    except Exception:
        return np.nan


def target(share):
    return np.log10(np.asarray(share, np.float64) + 1e-3)


def spectra_table(lib: str) -> pd.DataFrame:
    """Per spectrum of one library: structure key, ion type, collision energy, precursor m/z, precursor share."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from casmi.data import DATA
    pf = pq.ParquetFile(DATA / "train.parquet")
    cols = ["inchikey14", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]
    out = []
    for g in range(pf.num_row_groups):
        lab = pf.read_row_group(g, columns=["ingest_lib", "adduct"])
        m = pc.and_(pc.equal(lab["ingest_lib"], lib), pc.is_in(lab["adduct"], value_set=pa.array(TYPES)))
        idx = np.flatnonzero(np.asarray(m.to_numpy(zero_copy_only=False), dtype=bool))
        if not len(idx):
            continue
        t = pf.read_row_group(g, columns=cols).take(pa.array(idx)).to_pandas()
        out.append(pd.DataFrame({
            "ik": t.inchikey14.to_numpy(), "adduct": t.adduct.to_numpy(), "prec": t.precursor_mz.to_numpy(),
            "ce": [mean_ce(x) for x in t.collision_energy_ev],
            "share": [prec_share(a, b, p) for a, b, p in zip(t.ms2_mzs, t.ms2_normalized_intensities, t.precursor_mz)]}))
        print(f"  row group {g + 1}/{pf.num_row_groups}: {sum(len(o) for o in out)} spectra", flush=True)
    return pd.concat(out, ignore_index=True)


def design(desc: np.ndarray, adduct, ce, prec) -> np.ndarray:
    onehot = np.stack([np.asarray(adduct) == a for a in TYPES], 1).astype(np.float32)
    return np.hstack([desc, onehot, np.asarray(ce, np.float32)[:, None], np.asarray(prec, np.float32)[:, None]])


def main() -> None:
    work = Path(sys.argv[1]); work.mkdir(parents=True, exist_ok=True)
    args = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    excl = float(args.get("excl", 0.6)); max_struct = int(args.get("max_struct", 80000))
    lams = [float(x) for x in args.get("lams", "0,0.05,0.1,0.2,0.3,0.5,0.75,1").split(",")]
    import lightgbm as lgb
    from casmi.metric import ik14
    import dreams_local as dl

    sp_path = work / "e180_spectra.parquet"
    if sp_path.is_file():
        T = pd.read_parquet(sp_path)
    else:
        T = spectra_table("enveda-180"); T.to_parquet(sp_path)
    T = T[np.isfinite(T.share) & np.isfinite(T.ce)].reset_index(drop=True)
    L = pd.read_parquet(work / "e180_labels.parquet")          # written by adduct_prior.py (same structures, same order)
    XL = np.load(work / "e180_desc.npy")
    drop = np.load(work / f"e180_excl_{excl:g}.npy")
    row_of = {k: i for i, k in enumerate(L.index)}
    T["row"] = T.ik.map(row_of)
    T = T[T.row.notna()].reset_index(drop=True); T["row"] = T.row.astype(int)
    T = T[~drop[T.row.to_numpy()]].reset_index(drop=True)
    rng = np.random.default_rng(0)
    structs = T.row.unique()
    keep = set(rng.choice(structs, min(max_struct, len(structs)), replace=False).tolist())
    T = T[T.row.isin(keep)].reset_index(drop=True)
    val_s = set(rng.choice(sorted(keep), max(1, len(keep) // 20), replace=False).tolist())
    is_val = T.row.isin(val_s).to_numpy()
    print(f"spectra {len(T)} of {len(keep)} structures (validation {int(is_val.sum())}) | share > 0: {(T.share > 0).mean():.3f}", flush=True)
    X = design(XL[T.row.to_numpy()], T.adduct, T.ce, T.prec); y = target(T.share)
    mp_path = work / f"surv_model_{excl:g}_{max_struct}.txt"
    if mp_path.is_file():
        booster = lgb.Booster(model_file=str(mp_path))
    else:
        reg = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8,
                                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=4)
        reg.fit(X[~is_val], y[~is_val])
        booster = reg.booster_; booster.save_model(str(mp_path))
    pv = booster.predict(X[is_val])
    sd = float(np.std(y[is_val] - pv)); base_sd = float(np.std(y[is_val]))
    print(f"held-out structures: residual sd {sd:.3f} (target sd {base_sd:.3f}, R^2 {1 - sd ** 2 / base_sd ** 2:.3f})", flush=True)

    # ---- panel ----
    test = pd.read_parquet(ART / "np_test.parquet")
    test["m"] = [ik14(s_) for s_ in test.normalized_smiles]
    test = test[test.adduct.isin(TYPES)].copy()
    test["ce_m"] = [mean_ce(x) for x in test.collision_energy_ev]
    test["share"] = [prec_share(a, b, p) for a, b, p in zip(test.ms2_mzs, test.ms2_normalized_intensities, test.precursor_mz)]
    test = test[np.isfinite(test.share) & np.isfinite(test.ce_m)]
    spec = {m: g for m, g in test.groupby("m")}
    cache = pickle.load(open(work / "panel_desc.pkl", "rb"))
    # answers: does the model transfer?
    ans = test.groupby("m").normalized_smiles.first().to_dict()
    Xa = design(np.vstack([cache[ans[m]] for m in test.m]), test.adduct, test.ce_m, test.precursor_mz)
    ya, pa = target(test.share), booster.predict(Xa)
    print(f"panel answers ({len(test)} spectra): residual sd {np.std(ya - pa):.3f}, target sd {np.std(ya):.3f}, "
          f"corr {np.corrcoef(ya, pa)[0, 1]:.3f}", flush=True)
    add_ll = None
    if args.get("adduct"):
        add_ll = pickle.load(open(Path(args["adduct"]) / "panel_ll.pkl", "rb"))
    lists, heads = {}, {}
    for name, (lp, dp) in {
            "c1k_class2": (ART / "c1k_dump_c2.pkl.sub.csv", ART / "c1k_dump_c2.pkl"),
            "doc_misleading": (ART / "c1k_doc025cap4_dump_c2.pkl.sub.csv", ART / "c1k_doc025cap4_dump_c2.pkl"),
            "doc_heavy": (ART / "dreams" / "c1v1_c2_lists.csv", ART / "c1_dump_c2_v1.pkl"),
            "class1_visible": (ART / "c1k_dump_c1.pkl.sub.csv", ART / "c1k_dump_c1.pkl")}.items():
        sub = pd.read_csv(lp, dtype=str, keep_default_na=False)
        lists[name] = {m: [x for x in s.split(";") if x] for m, s in zip(sub.molecule_id, sub.smiles)}
        hits = pickle.load(open(dp, "rb"))["hits"] if dp.is_file() else {}
        heads[name] = {}
        for m, cand in lists[name].items():
            ks = {s_ for c, s_ in hits.get(m, {}).values() if c >= 0.95}
            if ks:
                heads[name][m] = next((i for i, c in enumerate(cand) if c not in ks), len(cand))
    keys = dl._key_cache([s_ for d in lists.values() for c in d.values() for s_ in c], dl.metric_key)
    for name, D in lists.items():
        res = {lam: [] for lam in lams}
        for m, cand in D.items():
            g = spec.get(m)
            if g is None or len(cand) < 2 or any(s_ not in cache for s_ in cand):
                for lam in lams:
                    kk = [keys.get(s_, "") for s_ in cand[:25]]
                    res[lam].append(1.0 / (kk.index(m) + 1) if m in kk else 0.0)
                continue
            yo = target(g.share.to_numpy())
            v = np.zeros(len(cand))
            for j, s_ in enumerate(cand):
                Xc = design(np.repeat(cache[s_][None], len(g), 0), g.adduct, g.ce_m, g.precursor_mz)
                v[j] = -float(((yo - booster.predict(Xc)) ** 2).sum()) / (2 * sd ** 2)
            b = 1.0 / (3.0 + np.arange(1, len(cand) + 1))
            h = heads[name].get(m, 0)
            for lam in lams:
                s = b / b.std() + lam * ap.zd(v)
                new = [cand[i] for i in np.argsort(-s, kind="stable")]
                if h:
                    top = cand[:h]; new = top + [c for c in new if c not in set(top)]
                nk = [keys.get(s_, "") for s_ in new[:25]]
                res[lam].append(1.0 / (nk.index(m) + 1) if m in nk else 0.0)
        print(f"{name:15s} MRR@25 (/250) at lambda " + " | ".join(f"{lam:g}: {np.sum(res[lam]) / 250:.4f}" for lam in lams), flush=True)


if __name__ == "__main__":
    main()
