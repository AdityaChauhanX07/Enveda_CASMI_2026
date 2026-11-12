"""Class-3 generator on the panel, step 1: propose products from PubChem relatives and cache them.

For each panel molecule (target neutral mass M, query logits z = mean of the five fold-safe FPNets):
  for every edit e: lighter window M - d_e (+-10 ppm) -> forward edit; heavier window M + d_e -> strip edit;
  rows pre-filtered to the best-documented N_DOC, scored by z (standardised on a random window sample) + doc;
  the top K_REL relatives per edit are edited; products near M are scored by z (standardised on the target window).
Products of every molecule go to artifacts/relatives_products.parquet; experiments/pubchem_relatives_merge.py
merges them into the class-2 / class-3 lists and scores them.
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

N_DOC, N_SAMPLE, N_FULL, K_REL, PPM = 500, 100, 20, 3, 10.0
G = {}


def _init():
    from rdkit.Chem import AllChem, rdFingerprintGenerator
    from casmi.edit_overlay import Chemistry, Tier
    from pubchem_relatives_coverage import STRIP
    G["chem"] = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    G["tier"] = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    G["strip"] = {k: AllChem.ReactionFromSmarts(s) for k, s in STRIP.items()}
    bits = np.load(EXT / "foldsafe" / "fp_bits.npy")
    G["sel_e"] = np.flatnonzero(bits < 4096); G["raw_e"] = bits[G["sel_e"]]       # ECFP4 block of the selected bits
    G["ecfp4"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    L = np.load(ROOT / "artifacts" / "panel_logits_foldsafe.npz")
    zs = [0.5 * (L[f"single{k}"].astype(np.float64) + L[f"merged{k}"].astype(np.float64)) for k in range(5)]
    G["z"] = dict(zip([str(k) for k in L["keys"]], np.mean(zs, axis=0)))


def _partial(smiles: str, ze: np.ndarray) -> float:
    m = G["chem"].Chem.MolFromSmiles(smiles)
    if m is None:
        return -np.inf
    return float(G["ecfp4"].GetFingerprintAsNumPy(m)[G["raw_e"]].astype(np.float64) @ ze)


def _molecule(args):
    """Products of the best relatives in every edit window: pass 1 ECFP4-partial score on the best-documented rows,
    pass 2 full score on the N_FULL best, relatives ranked by full z + 0.3 doc."""
    m, M, mu_t, sd_t = args
    if not G:
        _init()
    chem, tier, z = G["chem"], G["tier"], G["z"][m]
    ze = z[G["sel_e"]]
    rng = np.random.default_rng(abs(hash(m)) % 2**32)
    out = []
    jobs = [(e, d, -1, rx) for e, (d, rx) in chem.fwd.items()] + \
           [(e, d, +1, G["strip"][e]) for e, (d, _) in chem.fwd.items() if e in G["strip"]]
    for e, d, sign, rx in jobs:
        lo, hi = tier.window(M + sign * d, PPM)
        if hi <= lo:
            continue
        rows = np.arange(lo, hi)
        doc = np.asarray(tier.lsid[lo:hi], np.float64) + np.asarray(tier.lpmid[lo:hi], np.float64)
        sample = rows if len(rows) <= N_SAMPLE else rng.choice(rows, N_SAMPLE, replace=False)
        ref = [chem.fingerprint(tier.smiles(int(r))) for r in sample[:30]]            # full-score scale
        ref = np.array([f @ z for f in ref if f is not None])
        if len(ref) < 5:
            continue
        mu, sd = ref.mean(), ref.std() + 1e-9
        keep = np.argsort(-doc, kind="stable")[:N_DOC]
        part = np.array([_partial(tier.smiles(int(rows[j])), ze) for j in keep])
        best = keep[np.argsort(-part, kind="stable")[:N_FULL]]
        cand = []
        for j in best:
            f = chem.fingerprint(tier.smiles(int(rows[j])))
            if f is not None:
                cand.append((float((f @ z - mu) / sd), j))
        cand.sort(key=lambda x: -(x[0] + 0.3 * doc[x[1]]))
        for rank, (zr, j) in enumerate(cand[:K_REL]):
            r = int(rows[j])
            for key, ps in chem.products(tier.smiles(r), rx, cap=50).items():
                pm = chem.mass(ps)
                if not np.isfinite(pm) or abs(pm - M) > M * 15e-6:
                    continue
                f = chem.fingerprint(ps)
                if f is None:
                    continue
                out.append(dict(molecule=m, key=key, smiles=ps, z_p=float((f @ z - mu_t) / sd_t), edit=e, direction=sign,
                                rel_row=r, rel_rank=rank, rel_z=zr, rel_doc=float(doc[j])))
    return out


if __name__ == "__main__":
    from casmi.panel import PANEL, load_candidates
    from panel_fp_matrix import PanelFP, score_frame
    t0 = time.time()
    _init()
    c = load_candidates()
    raw = score_frame(c, G["z"], PanelFP())
    c["raw"] = raw
    st = c.groupby("molecule").raw.agg(["mean", "std"])
    p = pd.read_parquet(PANEL / "panel.parquet"); p["molecule"] = p.molecule.astype(str)
    args = [(m, float(M), float(st.loc[m, "mean"]), float(st.loc[m, "std"])) for m, M in zip(p.molecule, p.measured_neutral_mass)
            if m in st.index and m in G["z"]]
    if "--n" in sys.argv:
        args = args[: int(sys.argv[sys.argv.index("--n") + 1])]
    G.clear()
    rows = []
    with Pool(4) as pool:
        for i, res in enumerate(pool.imap_unordered(_molecule, args)):
            rows.extend(res)
            if i % 25 == 0:
                print(f"{i + 1}/{len(args)} molecules, {len(rows)} products, {time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(ROOT / "artifacts" / "relatives_products.parquet")
    print(f"done: {len(out)} products for {out.molecule.nunique()} molecules in {time.time() - t0:.0f}s")
