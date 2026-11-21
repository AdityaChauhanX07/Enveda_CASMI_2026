"""Ion-type prior: does the set of ion types a molecule was observed as separate the answer from its isomers?

Every test molecule comes with several spectra, each labelled with its ion type ([M+H]+, [M-H]-, [M+CH2O2-H]-,
[M+NH4]+, [M+Na]+, ...). Which ion types a compound forms is chemistry: acids and phenols deprotonate, amines
protonate and never show [M+NH4]+, glycosides and other polyols take formate and ammonium. Per-spectrum scorers see
the spectra that exist, not the ion types that are missing, so the molecule-level pattern is unused evidence.

Model: per ion type a, P(a observed | structure), LightGBM on RDKit descriptors and ionisation SMARTS counts, trained on
enveda-180 (same instrument; one row per structure; labels = ion types with at least one spectrum). For the panel
evaluation the panel answers and every training structure within Morgan Tanimoto >= --excl of one are left out, so the
answers are novel to the model (the panel over-rates anything that leans on close analogues).
Score per candidate: log-likelihood of the molecule's observed ion-type set, optionally with a per-ion-type logistic
recalibration fitted on the other panel folds (5-fold over molecules). Lists re-ranked by
1 / (3 + rank) / sd + lambda x zd(LL); library hits >= 0.95 stay on top; MRR per regime and by answer novelty.

usage: adduct_prior.py WORK_DIR [--excl=0.6] [--feat=desc|morgan] [--lams=0,0.1,...] [--types=all|core] [--calib=0|1]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

ION_TYPES = ["[M+H]+", "[M-H]-", "[M+CH2O2-H]-", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M+Cl]-"]
LABELS = ION_TYPES + ["pos", "neg"]

SMARTS = {
    "cooh": "[CX3](=O)[OX2H1,OX1-]", "phenol": "c[OX2H1]", "catechol": "c([OX2H1])c[OX2H1]",
    "enol": "[CX3]=[CX3][OX2H1]", "dicarbonyl_ch": "[CX3](=O)[CX4;H1,H2][CX3](=O)",
    "vinylogous_acid": "[OX2H1][CX3]=[CX3][CX3]=O", "sulfonic": "S(=O)(=O)[OX2H1,OX1-]",
    "sulfate": "O[SX4](=O)(=O)[OX2H1,OX1-]", "phosphate": "P(=O)([OX2H1,OX1-])", "tetrazole": "c1nnn[nH]1",
    "imide": "[CX3](=O)[NX3H1][CX3](=O)", "sulfonamide_nh": "S(=O)(=O)[NX3H1,NX3H2]",
    "hydroxamic": "C(=O)N[OX2H1]", "thiol": "[SX2H1]",
    "amine_prim": "[NX3H2;!$(NC=[O,S,N]);!$(N-a);!$(NS=O)][CX4]",
    "amine_sec": "[NX3H1;!$(NC=[O,S,N]);!$(N-a);!$(NS=O)]([CX4])[CX4]",
    "amine_tert": "[NX3H0;!$(NC=[O,S,N]);!$(N-a);!$(NS=O)]([CX4])([CX4])[CX4]",
    "guanidine": "[NX3]C(=[NX2])[NX3]", "amidine": "[NX3][CX3]=[NX2;!R]",
    "pyridine_n": "[nX2;r6]", "imidazole_n": "[nX2;r5]", "aniline": "c[NX3;H2,H1;!$(NC=O)]",
    "quat_n": "[NX4+]", "n_oxide": "[NX3+,nX3+][OX1-]",
    "amide": "[CX3](=O)[NX3]", "lactam": "[CX3;R](=O)[NX3;R]", "ester": "[CX3](=O)[OX2][#6]",
    "lactone": "[CX3;R](=O)[OX2;R]", "acetal": "[CX4]([OX2])([OX2])", "pyranose": "[OX2;r6][CX4;r6]([OX2])[CX4;r6]",
    "oh_prim": "[CX4H2][OX2H1]", "oh_sec": "[CX4H1][OX2H1]", "oh_tert": "[CX4H0][OX2H1]",
    "ketone": "[#6][CX3](=O)[#6]", "aldehyde": "[CX3H1](=O)", "enone": "[CX3]=[CX3][CX3]=O",
    "ether": "[OD2]([#6])[#6]", "methoxy_ar": "c[OX2][CH3]", "epoxide": "C1OC1", "nitro": "[NX3+](=O)[O-]",
    "nitrile": "C#N", "alkene": "[CX3]=[CX3]", "halogen": "[F,Cl,Br,I]", "carbonyl": "[CX3]=O",
    "flavone": "O=c1cc(-c2ccccc2)oc2ccccc12", "coumarin": "O=c1ccc2ccccc2o1", "quinone": "O=C1C=CC(=O)C=C1",
    "indole": "c1ccc2[nH]ccc2c1", "steroid": "C1CCC2C(C1)CCC1C2CCC2CCCC12",
}
DESC = ["MolWt", "MolLogP", "MolMR", "TPSA", "NumHDonors", "NumHAcceptors", "NHOHCount", "NOCount",
        "NumRotatableBonds", "RingCount", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
        "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms", "NumValenceElectrons", "MaxPartialCharge",
        "MinPartialCharge", "MaxAbsPartialCharge"] + [f"PEOE_VSA{i}" for i in range(1, 15)] + \
       [f"SMR_VSA{i}" for i in range(1, 11)] + [f"SlogP_VSA{i}" for i in range(1, 13)] + \
       [f"EState_VSA{i}" for i in range(1, 12)] + [f"VSA_EState{i}" for i in range(1, 11)]
_PAT = None
_FNS = None


def _init():
    global _PAT, _FNS
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    _PAT = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
    fns = dict(Descriptors.descList)
    _FNS = [(n, fns[n]) for n in DESC if n in fns] + [(n, f) for n, f in Descriptors.descList if n.startswith("fr_")]


def n_features() -> int:
    if _FNS is None:
        _init()
    return len(_FNS) + len(_PAT) + 8


def featurize(smi: str) -> np.ndarray:
    from rdkit import Chem
    if _FNS is None:
        _init()
    out = np.full(n_features(), np.nan, np.float32)
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return out
    i = 0
    for _, f in _FNS:
        try:
            out[i] = f(m)
        except Exception:
            pass
        i += 1
    for p in _PAT.values():
        out[i] = len(m.GetSubstructMatches(p)) if p is not None else np.nan
        i += 1
    counts = {}
    for a in m.GetAtoms():
        counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    charge = sum(a.GetFormalCharge() for a in m.GetAtoms())
    for sym in ("C", "N", "O", "S", "P"):
        out[i] = counts.get(sym, 0); i += 1
    out[i] = sum(counts.get(s, 0) for s in ("F", "Cl", "Br", "I")); i += 1
    out[i] = charge; i += 1
    out[i] = counts.get("O", 0) / max(counts.get("C", 1), 1); i += 1
    return np.nan_to_num(out, nan=np.nan, posinf=1e6, neginf=-1e6)


def featurize_many(smiles: list[str], procs: int = 4) -> np.ndarray:
    with mp.get_context("fork").Pool(procs, initializer=_init) as pool:
        rows = pool.map(featurize, smiles, chunksize=256)
    return np.vstack(rows) if rows else np.zeros((0, n_features()), np.float32)


def morgan_counts(smiles: list[str], n_bits: int = 1024) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), np.uint8)
    for i, s_ in enumerate(smiles):
        m = Chem.MolFromSmiles(s_) if s_ else None
        if m is not None:
            out[i] = np.minimum(gen.GetCountFingerprintAsNumPy(m), 255)
    return out


def labels_of(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per structure: SMILES and a 0/1 column per label (ion type with >= 1 spectrum, any pos, any neg)."""
    g = frame.groupby("inchikey14", sort=True)
    out = pd.DataFrame({"smiles": g.normalized_smiles.first()})
    for a in ION_TYPES:
        out[a] = g.adduct.apply(lambda s, a=a: int((s == a).any()))
    out["pos"] = g.ionization_mode.apply(lambda s: int((s == "positive").any()))
    out["neg"] = g.ionization_mode.apply(lambda s: int((s == "negative").any()))
    return out


def train_models(X: np.ndarray, Y: pd.DataFrame, seed: int = 0) -> dict:
    import lightgbm as lgb
    models = {}
    for a in LABELS:
        y = Y[a].to_numpy()
        if y.sum() < 50:
            continue
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                 random_state=seed, verbose=-1, n_jobs=4)
        clf.fit(X, y)
        models[a] = clf
    return models


def predict(models: dict, X: np.ndarray) -> dict:
    return {a: np.clip(m.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4) for a, m in models.items()}


def auc(y, p) -> float:
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.r_[pos, neg], kind="stable")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def zd(v, eps=0.01):
    return (v - v.mean()) / (v.std() + eps)


def main() -> None:
    work = Path(sys.argv[1]); work.mkdir(parents=True, exist_ok=True)
    args = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    excl = float(args.get("excl", 0.6))
    feat = args.get("feat", "desc")
    lams = [float(x) for x in args.get("lams", "0,0.05,0.1,0.2,0.3,0.5,0.75,1").split(",")]
    types = args.get("types", "all")
    calib = int(args.get("calib", "1"))
    from casmi.metric import ik14
    import dreams_local as dl
    from rdkit import DataStructs

    # ---- training structures: enveda-180, one row per structure ----
    lab_path = work / "e180_labels.parquet"
    if lab_path.is_file():
        L = pd.read_parquet(lab_path)
    else:
        from casmi.panel import read_library
        e = read_library("enveda-180", columns=["inchikey14", "normalized_smiles", "adduct", "ionization_mode"])
        L = labels_of(e); del e
        L.to_parquet(lab_path)
    print("enveda-180 structures", len(L), "| label rates", {a: round(float(L[a].mean()), 3) for a in LABELS}, flush=True)
    fx_path = work / "e180_desc.npy"
    if fx_path.is_file():
        XL = np.load(fx_path)
    else:
        XL = featurize_many(L.smiles.tolist()); np.save(fx_path, XL)
    if feat == "morgan":
        mg_path = work / "e180_morgan.npy"
        if mg_path.is_file():
            ML = np.load(mg_path)
        else:
            ML = morgan_counts(L.smiles.tolist()); np.save(mg_path, ML)
        XL = np.hstack([XL, ML.astype(np.float32)])

    # ---- panel: observed ion types per answer, lists per regime ----
    test = pd.read_parquet(ART / "np_test.parquet")
    test["m"] = [ik14(s_) for s_ in test.normalized_smiles]
    obs = {}
    for m, g in test.groupby("m"):
        y = {a: int((g.adduct == a).any()) for a in ION_TYPES}
        y["pos"] = int((g.ionization_mode == "positive").any()); y["neg"] = int((g.ionization_mode == "negative").any())
        obs[m] = y
    ans_smiles = test.groupby("m").normalized_smiles.first().to_dict()
    regimes = {
        "c1k_class2": (ART / "c1k_dump_c2.pkl.sub.csv", ART / "c1k_dump_c2.pkl"),
        "doc_misleading": (ART / "c1k_doc025cap4_dump_c2.pkl.sub.csv", ART / "c1k_doc025cap4_dump_c2.pkl"),
        "doc_heavy": (ART / "dreams" / "c1v1_c2_lists.csv", ART / "c1_dump_c2_v1.pkl"),
        "c1e_class2": (ART / "c1e_dump_c2.pkl.sub.csv", ART / "c1e_dump_c2.pkl"),
        "class1_visible": (ART / "c1k_dump_c1.pkl.sub.csv", ART / "c1k_dump_c1.pkl"),
    }
    lists, heads = {}, {}
    for name, (lp, dp) in regimes.items():
        if not lp.is_file():
            continue
        sub = pd.read_csv(lp, dtype=str, keep_default_na=False)
        lists[name] = {m: [x for x in s.split(";") if x] for m, s in zip(sub.molecule_id, sub.smiles)}
        heads[name] = {}
        if dp.is_file():
            hits = pickle.load(open(dp, "rb"))["hits"]
            for m, cand in lists[name].items():
                ks = {s_ for c, s_ in hits.get(m, {}).values() if c >= 0.95}
                if ks:
                    heads[name][m] = next((i for i, c in enumerate(cand) if c not in ks), len(cand))
    all_smiles = sorted({s_ for d in lists.values() for c in d.values() for s_ in c} | set(ans_smiles.values()))
    keys = dl._key_cache(all_smiles, dl.metric_key)
    cf_path = work / "panel_desc.pkl"
    cache = pickle.load(open(cf_path, "rb")) if cf_path.is_file() else {}
    todo = [s_ for s_ in all_smiles if s_ not in cache]
    if todo:
        for s_, row in zip(todo, featurize_many(todo)):
            cache[s_] = row
        pickle.dump(cache, open(cf_path, "wb"))
    if feat == "morgan":
        mg = dict(zip(all_smiles, morgan_counts(all_smiles).astype(np.float32)))

    def X_of(smiles):
        x = np.vstack([cache[s_] for s_ in smiles])
        return np.hstack([x, np.vstack([mg[s_] for s_ in smiles])]) if feat == "morgan" else x

    # ---- leave the panel answers and their neighbourhood out of training ----
    ans_fp = [dl.fp(s_) for s_ in ans_smiles.values()]
    ans_fp = [f for f in ans_fp if f is not None]
    ex_path = work / f"e180_excl_{excl:g}.npy"
    if ex_path.is_file():
        drop = np.load(ex_path)
    else:
        drop = np.zeros(len(L), bool)
        ans_k = set(ans_smiles)
        tr_fp = []
        for i, s_ in enumerate(L.smiles):
            f = dl.fp(s_); tr_fp.append(f)
        ok = [i for i, f in enumerate(tr_fp) if f is not None]
        fps = [tr_fp[i] for i in ok]
        for f in ans_fp:
            t = np.array(DataStructs.BulkTanimotoSimilarity(f, fps))
            drop[np.array(ok)[t >= excl]] = True
        drop |= L.index.isin(list(ans_k)) | L.index.isin([k[:14] for k in ans_k])
        np.save(ex_path, drop)
    print(f"training structures {int((~drop).sum())} (left out {int(drop.sum())}: panel answers + Tanimoto >= {excl:g})", flush=True)
    mp_path = work / f"models_{feat}_{excl:g}.pkl"
    if mp_path.is_file():
        models = pickle.load(open(mp_path, "rb"))
    else:
        models = train_models(XL[~drop], L[~drop].reset_index(drop=True))
        pickle.dump(models, open(mp_path, "wb"))
    use = [a for a in LABELS if a in models] if types == "all" else [a for a in ["[M+H]+", "[M-H]-", "[M+CH2O2-H]-", "[M+NH4]+", "[M+Na]+", "neg"] if a in models]
    print("ion types used:", use, flush=True)

    # ---- transfer: AUC on the panel answers ----
    ms = sorted(m for m in obs if m in ans_smiles)
    Pa = predict(models, X_of([ans_smiles[m] for m in ms]))
    print("panel answers: AUC (rate observed / mean predicted)")
    for a in use:
        y = np.array([obs[m][a] for m in ms])
        print(f"  {a:14s} AUC {auc(y, Pa[a]):.3f}  ({y.mean():.3f} / {Pa[a].mean():.3f})")

    # per-ion-type logistic recalibration on the answers, 5 folds over molecules (fit on 4, apply to the 5th)
    rng = np.random.default_rng(0)
    fold = dict(zip(ms, rng.permutation(len(ms)) % 5))

    def logit(p):
        return np.log(p / (1 - p))

    calib_par = {}
    if calib:
        from sklearn.linear_model import LogisticRegression
        for k in range(5):
            tr = [i for i, m in enumerate(ms) if fold[m] != k]
            for a in use:
                y = np.array([obs[ms[i]][a] for i in tr])
                x = logit(Pa[a][tr]).reshape(-1, 1)
                if y.min() == y.max():
                    calib_par[(k, a)] = (1.0, 0.0); continue
                lr = LogisticRegression(C=1.0).fit(x, y)
                calib_par[(k, a)] = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        for a in use:
            print(f"  calibration {a:14s} slope / intercept (fold 0): {calib_par[(0, a)][0]:.2f} / {calib_par[(0, a)][1]:.2f}")

    def ll(m, smiles):
        P = predict(models, X_of(smiles))
        tot = np.zeros(len(smiles))
        for a in use:
            p = P[a]
            if calib:
                b, c = calib_par[(fold.get(m, 0), a)]
                p = 1 / (1 + np.exp(-(b * logit(p) + c)))
            p = np.clip(p, 1e-4, 1 - 1e-4)
            tot += np.log(p) if obs[m][a] else np.log(1 - p)
        return tot

    nov = pd.read_parquet(ART / "dreams" / "panel_novelty.parquet") if (ART / "dreams" / "panel_novelty.parquet").is_file() else None
    nov_of = {}
    if nov is not None:
        kc = [c for c in nov.columns if c.lower() in ("m", "molecule", "molecule_id", "ik14", "key")]
        vc = [c for c in nov.columns if "tani" in c.lower() or "sim" in c.lower() or "nearest" in c.lower()]
        if kc and vc:
            nov_of = dict(zip(nov[kc[0]], nov[vc[0]]))
    report = {}
    for name, D in lists.items():
        rows = []
        for m, cand in D.items():
            if m not in obs or len(cand) < 2:
                continue
            kk = [keys.get(s_, "") for s_ in cand]
            v = ll(m, cand)
            b = 1.0 / (3.0 + np.arange(1, len(cand) + 1))
            h = heads[name].get(m, 0)
            ans_pos = kk.index(m) if m in kk else -1
            res = {}
            for lam in lams:
                s = b / b.std() + lam * zd(v)
                new = [cand[i] for i in np.argsort(-s, kind="stable")]
                if h:
                    top = cand[:h]; new = top + [c for c in new if c not in set(top)]
                nk = [keys.get(s_, "") for s_ in new[:25]]
                res[lam] = 1.0 / (nk.index(m) + 1) if m in nk else 0.0
            same = ans_pos >= 0 and ans_pos > 0
            rows.append(dict(m=m, base=res[0.0], **{f"l{lam:g}": res[lam] for lam in lams}, ans_rank=ans_pos + 1,
                             ans_ll_rank=(int((v > v[ans_pos]).sum()) + 1) if ans_pos >= 0 else -1, n=len(cand),
                             nov=nov_of.get(m, np.nan)))
        R = pd.DataFrame(rows)
        line = " | ".join(f"{lam:g}: {R[f'l{lam:g}'].sum() / 250:.4f}" for lam in lams)
        inl = R[R.ans_rank > 0]
        print(f"{name:15s} n {len(R)} | MRR@25 (/250) at lambda {line}")
        print(f"{'':15s} answer in list {len(inl)}; LL rank of the answer: mean {inl.ans_ll_rank.mean():.2f} of {inl.n.mean():.1f}; "
              f"LL ranks answer first {int((inl.ans_ll_rank == 1).sum())}; random expectation {(1 / inl.n).sum():.1f}")
        if R.nov.notna().any():
            for lo, hi in ((0, 0.7), (0.7, 1.01)):
                sel = R[(R.nov >= lo) & (R.nov < hi)]
                if len(sel):
                    print(f"{'':15s} answer novelty {lo}-{hi}: n {len(sel)} | " + " | ".join(
                        f"{lam:g}: {sel[f'l{lam:g}'].mean() - sel.base.mean():+.4f}" for lam in lams if lam > 0))
        report[name] = {f"{lam:g}": float(R[f"l{lam:g}"].sum() / 250) for lam in lams}
    json.dump(dict(excl=excl, feat=feat, types=types, calib=calib, report=report), open(work / f"report_{feat}_{excl:g}_{types}_c{calib}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
