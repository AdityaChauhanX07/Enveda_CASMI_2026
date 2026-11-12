"""Gated neighbour-edit generator for a finished submission (runs inside the Kaggle notebook, GPU optional).

For a test molecule B with a spectral neighbour A whose neutral mass differs by a known modification, the edit
(or its reverse) is applied to A's top candidates. A product enters B's top 3 only where it beats B's own
candidates on the fold-safe FPNet z-score plus a bonus BETA (B's candidates also get 0.4 * documentation).
The stack's own candidates keep their relative order. Panel check (experiments/neighbour_edit_gate.py --reverse
--top3, bonus 3): class-2 lists -0.0044, class-3 lists +0.057, break-even at ~7 % class 3.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = dict(cos_min=0.5, min_matches=4, top_neighbour=3, beta=3.0, w_doc=0.4, max_rank=3, max_insert=3, ensemble=1,
              min_slot=0, guard=float("inf"),
              window_ppm=10.0, window_sample=200, mass_tol=0.003, max_seconds=1200, device="cuda", tol=0.01,
              top_n=60, min_rel=0.01, seed=0)
FORWARD = {
    "O-methyl": (14.01565, "[OX2H1:1]>>[O:1]C"),
    "N-methyl": (14.01565, "[NX3;H1,H2;!$(NC=O):1]>>[N:1]C"),
    "C-methyl(ar)": (14.01565, "[cH1:1]>>[c:1]C"),
    "OH(ar)": (15.99491, "[cH1:1]>>[c:1]O"),
    "OH(aliph)": (15.99491, "[CX4;!H0:1]>>[C:1]O"),
    "hexose": (162.05282, "[OX2H1:1]>>[O:1]C1OC(CO)C(O)C(O)C1O"),
    "deoxyhexose": (146.05791, "[OX2H1:1]>>[O:1]C1OC(C)C(O)C(O)C1O"),
    "pentose": (132.04226, "[OX2H1:1]>>[O:1]C1OCC(O)C(O)C1O"),
    "glucuronide": (176.03209, "[OX2H1:1]>>[O:1]C1OC(C(=O)O)C(O)C(O)C1O"),
    "acetyl": (42.01057, "[OX2H1:1]>>[O:1]C(C)=O"),
    "H2": (2.01565, "[C:1]=[C:2]>>[C:1][C:2]"),
    "prenyl(ar)": (68.06260, "[cH1:1]>>[c:1]CC=C(C)C"),
    "malonyl": (86.00039, "[OX2H1:1]>>[O:1]C(=O)CC(=O)O"),
}
REVERSE = {
    "O-methyl": (14.01565, "[CH3][OX2:1][#6:2]>>[O:1][#6:2]"),
    "OH": (15.99491, "[OX2H1][#6:1]>>[#6:1]"),
    "hexose": (162.05282, "[O:1]C1OC(CO)C(O)C(O)C1O>>[O:1]"),
    "deoxyhexose": (146.05791, "[O:1]C1OC(C)C(O)C(O)C1O>>[O:1]"),
    "acetyl": (42.01057, "[O:1]C(C)=O>>[O:1]"),
}
INPUT = Path("/kaggle/input")


def _find(pattern: str, root: Path = INPUT) -> Path:
    hits = sorted(glob.glob(str(root / "**" / pattern), recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[0])


def _load_v4(code_dir: Path) -> dict:
    """chem, spectra and fpnet of the v4 engine code under a private package name."""
    pkg = types.ModuleType("_v4casmi"); pkg.__path__ = [str(code_dir)]; sys.modules["_v4casmi"] = pkg
    mods = {}
    for name in ("chem", "spectra", "fpnet"):
        spec = importlib.util.spec_from_file_location(f"_v4casmi.{name}", code_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


# ---------------- spectral graph (same rule as the networking overlay) ----------------
def _prepare(mz, it, precursor, top_n, min_rel):
    mz = np.asarray(mz, dtype=np.float64); it = np.asarray(it, dtype=np.float64)
    keep = (mz < precursor - 1.0) & np.isfinite(it) & (it > 0)
    mz, it = mz[keep], it[keep]
    if len(it) == 0:
        return mz, it
    keep = it >= min_rel * it.max(); mz, it = mz[keep], it[keep]
    if len(it) > top_n:
        idx = np.argpartition(-it, top_n)[:top_n]; mz, it = mz[idx], it[idx]
    o = np.argsort(mz); w = np.sqrt(it[o])
    return mz[o], w / np.linalg.norm(w)


def _modcos(a, b, tol):
    (pa, ma, wa), (pb, mb, wb) = a, b
    if len(ma) == 0 or len(mb) == 0:
        return 0.0, 0
    shift = pa - pb
    hit = np.abs(ma[:, None] - mb[None, :]) <= tol
    if abs(shift) > tol:
        hit |= np.abs(ma[:, None] - (mb[None, :] + shift)) <= tol
    ii, jj = np.nonzero(hit)
    if len(ii) == 0:
        return 0.0, 0
    prod = wa[ii] * wb[jj]; ua, ub, tot, n = set(), set(), 0.0, 0
    for q in np.argsort(-prod, kind="stable"):
        i, j = ii[q], jj[q]
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); tot += prod[q]; n += 1
    return float(tot), n


def neighbours(test: pd.DataFrame, p: dict) -> dict:
    groups = {}
    for r in test.itertuples(index=False):
        mz, w = _prepare(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz), p["top_n"], p["min_rel"])
        groups.setdefault(str(r.adduct), []).append((str(r.molecule_id), (float(r.precursor_mz), mz, w)))
    best = {}
    for items in groups.values():
        for x in range(len(items)):
            for y in range(x + 1, len(items)):
                (ma, sa), (mb, sb) = items[x], items[y]
                if ma == mb:
                    continue
                s, n = _modcos(sa, sb, p["tol"])
                if n >= p["min_matches"] and s >= p["cos_min"]:
                    key = (ma, mb) if ma < mb else (mb, ma)
                    best[key] = max(best.get(key, 0.0), s)
    out = {}
    for (a, b), s in best.items():
        out.setdefault(a, []).append((b, s)); out.setdefault(b, []).append((a, s))
    return out


# ---------------- structures ----------------
class Chemistry:
    def __init__(self, fp_bits: np.ndarray):
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator as g
        from rdkit.Chem.Descriptors import ExactMolWt
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog("rdApp.*")
        self.Chem, self.MACCS, self.mw = Chem, MACCSkeys, ExactMolWt
        self.taut = rdMolStandardize.TautomerEnumerator()
        self.gens = [g.GetMorganGenerator(radius=2, fpSize=4096), g.GetMorganGenerator(radius=3, fpSize=4096),
                     g.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=g.GetMorganFeatureAtomInvGen()),
                     g.GetRDKitFPGenerator(fpSize=2048, maxPath=6), g.GetAtomPairGenerator(fpSize=2048),
                     g.GetTopologicalTorsionGenerator(fpSize=2048)]
        self.bits = fp_bits
        self.fwd = {k: (d, AllChem.ReactionFromSmarts(s)) for k, (d, s) in FORWARD.items()}
        self.rev = {k: (d, AllChem.ReactionFromSmarts(s)) for k, (d, s) in REVERSE.items()}
        self._key = {}

    def key(self, smiles: str) -> str:
        """The scorer's identity: stereo removed, canonical tautomer, InChIKey first block."""
        if smiles not in self._key:
            k = ""
            try:
                m = self.Chem.MolFromSmiles(smiles)
                if m is not None:
                    self.Chem.RemoveStereochemistry(m)
                    k = self.Chem.MolToInchiKey(self.taut.Canonicalize(m))[:14]
            except Exception:
                k = ""
            self._key[smiles] = k
        return self._key[smiles]

    def fingerprint(self, smiles: str):
        m = self.Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        parts = [gen.GetFingerprintAsNumPy(m).astype(np.float32) for gen in self.gens]
        parts.append(np.array(list(self.MACCS.GenMACCSKeys(m)), dtype=np.float32))
        return np.concatenate(parts)[self.bits]

    def mass(self, smiles: str) -> float:
        m = self.Chem.MolFromSmiles(smiles)
        return float(self.mw(m)) if m is not None else float("nan")

    def products(self, smiles: str, rxn, cap: int = 400) -> dict:
        out = {}
        m = self.Chem.MolFromSmiles(smiles)
        if m is None:
            return out
        for ps in rxn.RunReactants((m,), maxProducts=cap):
            q = ps[0]
            try:
                self.Chem.SanitizeMol(q)
                s = self.Chem.MolToSmiles(q)
                k = self.key(s)
                if k:
                    out.setdefault(k, s)
            except Exception:
                continue
        return out


class Tier:
    """Memory-mapped PubChem tier: masses (sorted), SMILES, connectivity keys and documentation counts."""

    def __init__(self, root: Path, doc_root: Path | None = None):
        L = lambda n, **kw: np.load(root / n, mmap_mode="r", **kw)
        self.mass, self.off, self.smi = L("pc_mass.npy"), L("pc_off.npy"), L("pc_smiles.npy")
        if doc_root is None:
            doc_root = root.parent if (root.parent / "pc_ik14.npy").is_file() else root
        self.ik = np.load(doc_root / "pc_ik14.npy", mmap_mode="r")
        self.lsid = np.load(doc_root / "pc_lsid.npy", mmap_mode="r")
        self.lpmid = np.load(doc_root / "pc_lpmid.npy", mmap_mode="r")

    def window(self, mass: float, ppm: float) -> tuple[int, int]:
        tol = mass * ppm * 1e-6
        return int(np.searchsorted(self.mass, mass - tol, "left")), int(np.searchsorted(self.mass, mass + tol, "right"))

    def smiles(self, row: int) -> str:
        return bytes(self.smi[self.off[row]:self.off[row + 1]]).decode()

    def doc(self, mass: float, connectivity: str) -> float:
        lo, hi = int(np.searchsorted(self.mass, mass - 5e-4, "left")), int(np.searchsorted(self.mass, mass + 5e-4, "right"))
        if hi <= lo:
            return 0.0
        hit = np.flatnonzero(np.asarray(self.ik[lo:hi]) == connectivity.encode())
        if len(hit) == 0:
            return 0.0
        r = lo + int(hit[0])
        return float(self.lsid[r]) + float(self.lpmid[r])


# ---------------- fold-safe FPNet ----------------
def query_logits(bank, v4, spectra: list[dict]) -> np.ndarray:
    """Molecule-level logits exactly as the engine builds them: mean of single and per-polarity merged views."""
    fp, ch, sp = v4["fpnet"], v4["chem"], v4["spectra"]
    single = []
    for s in spectra:
        pm, pi = fp.prep_peaks(s["mz"], s["it"], s["prec"])
        ok = s["ce"] is not None and not np.isnan(s["ce"])
        single.append(dict(mz=pm, it=pi, prec=float(s["prec"]), adduct_ix=ch.adduct_index(s["adduct"]),
                           ce=float(s["ce"]) if ok else 0.0, ce_known=1.0 if ok else 0.0,
                           n_merged=min(int(s.get("ce_n", 1)), 8), mode=float(s["mode"])))
    merged = []
    for md in (1, -1):
        grp = [s for s in spectra if s["mode"] == md]
        if not grp:
            continue
        mm, ii = sp.merge_spectra([(s["mz"], s["it"]) for s in grp])
        prec = float(np.median([s["prec"] for s in grp]))
        pm, pi = fp.prep_peaks(mm, ii, prec)
        ces = [s["ce"] for s in grp if s["ce"] is not None and not np.isnan(s["ce"])]
        ads = [s["adduct"] for s in grp]
        merged.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=ch.adduct_index(max(ads, key=ads.count)),
                           ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                           n_merged=min(sum(max(1, int(s.get("ce_n", 1))) for s in grp), 8), mode=float(md)))
    zs, _ = bank.logits_el(single)
    zm, _ = bank.logits_el(merged)
    return (0.5 * (zs.mean(0) + zm.mean(0))).astype(np.float64)


def molecule_spectra(test: pd.DataFrame, v4) -> tuple[dict, dict]:
    test = test.copy()
    test["nm"] = v4["chem"].neutral_mass(test.precursor_mz.values.astype(np.float64), test.adduct.values)
    spectra, target = {}, {}
    for mid, sub in test.groupby("molecule_id", sort=False):
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        if not len(nms):
            continue
        target[str(mid)] = float(np.median(nms))
        out = []
        for r in sub.itertuples():
            ce = r.collision_energy_ev
            arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
            out.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                            mode=1 if r.ionization_mode == "positive" else -1, adduct=r.adduct, prec=float(r.precursor_mz),
                            ce=float(arr.mean()) if len(arr) else np.nan, ce_n=int(len(arr)) if len(arr) else 1))
        spectra[str(mid)] = out
    return spectra, target


# ---------------- overlay ----------------
def propose(lists: dict, nbr: dict, target: dict, chem: Chemistry, p: dict, t0: float) -> dict:
    """molecule -> {key: smiles} edit products not already in its list."""
    out = {}
    for b, cand in lists.items():
        if time.time() - t0 > p["max_seconds"] / 2:
            break
        if b not in nbr or b not in target:
            continue
        have = {chem.key(s) for s in cand}
        tol_b = target[b] * 10e-6 + 1e-3
        for a, _ in nbr[b]:
            if a not in target or a not in lists:
                continue
            dm = target[b] - target[a]
            todo = [r for d, r in chem.fwd.values() if abs(dm - d) <= p["mass_tol"]]
            todo += [r for d, r in chem.rev.values() if abs(dm + d) <= p["mass_tol"]]
            for rxn in todo:
                for smi in lists[a][: p["top_neighbour"]]:
                    for k, s in chem.products(smi, rxn).items():
                        if k in have or abs(chem.mass(s) - target[b]) > tol_b:
                            continue
                        out.setdefault(b, {})[k] = s
    return out


def apply(test_path, submission_path, output_path, report_path=None, p=PARAMS, v4_code=None, fold_model=None,
          tier_root=None, fp_bits=None, logits_fn=None, v4=None) -> dict:
    """logits_fn(molecule_id) and v4 replace the model run and the engine code (offline checks only)."""
    t0 = time.time()
    p = dict(PARAMS, **(p or {}))
    test = pd.read_parquet(test_path)
    sub = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
    lists = {str(m): s.split(";") for m, s in zip(sub.molecule_id, sub.smiles)}
    v4 = v4 or _load_v4(Path(v4_code) if v4_code else _find("casmi26-v4b-models/code/casmi/fpnet.py").parent)
    fold_model = Path(fold_model) if fold_model else _find("casmi26-fold-safe-fpnet/fpnet_fold0.pt")
    bits = np.load(fp_bits) if fp_bits else np.load(fold_model.parent / "fp_bits.npy")
    if tier_root:
        tier = Tier(Path(tier_root))
    else:
        doc = _find("pc_ik14.npy").parent
        tier = Tier(doc / "casmi26-pubchem-tier" if (doc / "casmi26-pubchem-tier" / "pc_mass.npy").is_file()
                    else _find("casmi26-pubchem-tier/pc_mass.npy").parent, doc)
    chem = Chemistry(bits)
    spectra, target = molecule_spectra(test, v4)
    nbr = neighbours(test, p)
    prods = propose(lists, nbr, target, chem, p, t0)
    report = dict(params=p, molecules=len(lists), with_neighbour=len(nbr), with_products=len(prods),
                  products=sum(len(v) for v in prods.values()), inserted=0, molecules_changed=0)
    new = {}
    bank = None
    if not logits_fn and spectra:
        # ensemble=1: all five fold-safe models (logits averaged inside ModelBank); panel 0.243 alone vs 0.208 for fold 0
        paths = sorted(str(x) for x in fold_model.parent.glob("fpnet_fold*.pt")) if p.get("ensemble", 1) else [str(fold_model)]
        bank = v4["fpnet"].ModelBank(paths, device=p["device"])
        first = next(iter(spectra))
        z0 = query_logits(bank, v4, spectra[first])
        report["model_check"] = dict(molecule=first, models=len(paths), bits=int(len(z0)), mean_logit=round(float(z0.mean()), 3),
                                     finite=bool(np.isfinite(z0).all()))
    if prods:
        rng = np.random.default_rng(p["seed"])
        for b, P in prods.items():
            if time.time() - t0 > p["max_seconds"]:
                report["stopped_early"] = True
                break
            z = logits_fn(b) if logits_fn else query_logits(bank, v4, spectra[b])
            if z is None:
                continue
            lo, hi = tier.window(target[b], p["window_ppm"])
            rows = np.arange(lo, hi) if hi - lo <= p["window_sample"] else np.sort(rng.choice(np.arange(lo, hi), p["window_sample"], replace=False))
            ref = [chem.fingerprint(tier.smiles(int(r))) for r in rows]
            ref = np.array([f @ z for f in ref if f is not None])
            if len(ref) < 10:
                continue
            mu, sd = float(ref.mean()), float(ref.std() + 1e-9)
            cand = lists[b]
            top = []
            for s in cand[: p["max_rank"]]:
                f = chem.fingerprint(s)
                if f is None:
                    top.append(np.inf); continue
                conn = chem.Chem.MolToInchiKey(chem.Chem.MolFromSmiles(s))[:14]
                top.append((f @ z - mu) / sd + p["w_doc"] * tier.doc(chem.mass(s), conn))
            scored = []
            for k, s in P.items():
                f = chem.fingerprint(s)
                if f is not None:
                    scored.append(((f @ z - mu) / sd + p["beta"], s))
            scored.sort(key=lambda x: -x[0])
            if top and top[0] >= p["guard"]:
                continue
            slots = {}
            for sp_, s in scored[: p["max_insert"]]:
                beaten_by = [i for i, si in enumerate(top) if si >= sp_]
                r = max((max(beaten_by) + 1) if beaten_by else 0, p["min_slot"])
                if r < p["max_rank"]:
                    slots.setdefault(r, []).append(s)
            if not slots:
                continue
            out = []
            for i, s in enumerate(cand):
                out.extend(slots.get(i, []))
                out.append(s)
            seen, final = set(), []
            for s in out:
                k = chem.key(s) or s
                if k not in seen:
                    seen.add(k); final.append(s)
            new[b] = final[:25]
            report["inserted"] += sum(len(v) for v in slots.values()); report["molecules_changed"] += 1
    out = sub.copy()
    out["smiles"] = [";".join(new.get(str(m), lists[str(m)])) for m in out.molecule_id]
    assert out.molecule_id.tolist() == sub.molecule_id.tolist()
    assert out.smiles.str.split(";").map(len).between(1, 25).all()
    out.to_csv(output_path, index=False)
    report["seconds"] = round(time.time() - t0, 1)
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2))
    return report
