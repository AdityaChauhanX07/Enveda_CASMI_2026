"""Cache for fast relative-selection experiments on the panel.

Pass `build`: for every panel molecule and edit window, the 500 best-documented rows with their documentation,
ECFP4-partial score, and the full ensemble z (window-standardised) for the 60 best by partial score.
Pass `eval`: pick relatives with any rule over the cache, compute their products (memoised per (row, edit)), score
z_p + 0.2 x relative doc + 0.5 x log(routes) and merge as the overlay does."""
from __future__ import annotations

import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
CACHE = ROOT / "artifacts" / "relatives_cache"
N_DOC = 500
G = {}


def _init():
    from casmi.edit_overlay import Chemistry, Tier
    G["chem"] = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    G["tier"] = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    L = np.load(ROOT / "artifacts" / "panel_logits_foldsafe.npz")
    zs = [0.5 * (L[f"single{k}"].astype(np.float64) + L[f"merged{k}"].astype(np.float64)) for k in range(5)]
    G["z"] = dict(zip([str(k) for k in L["keys"]], np.mean(zs, axis=0)))


def _build_one(args):
    m, M = args
    if not G:
        _init()
    chem, tier, z = G["chem"], G["tier"], G["z"][m]
    z_e = z[chem.sel_e]
    rng = np.random.default_rng(abs(hash(m)) % 2**32)
    out = {}
    for e, (d, _) in chem.fwd.items():
        for sign in (-1, +1):
            lo, hi = tier.window(M + sign * d, 10.0)
            if hi <= lo:
                continue
            doc = np.asarray(tier.lsid[lo:hi], np.float64) + np.asarray(tier.lpmid[lo:hi], np.float64)
            offs = np.asarray(tier.off[lo:hi + 1], np.int64); blob = bytes(tier.smi[offs[0]:offs[-1]])
            smi_at = lambda j: blob[offs[j] - offs[0]:offs[j + 1] - offs[0]].decode()
            sample = np.arange(hi - lo) if hi - lo <= 30 else rng.choice(hi - lo, 30, replace=False)
            ref = [chem.fingerprint(smi_at(int(j))) for j in sample]
            ref = np.array([f @ z for f in ref if f is not None])
            if len(ref) < 5:
                continue
            mu, sd = ref.mean(), ref.std() + 1e-9
            keep = np.argsort(-doc, kind="stable")[:N_DOC]
            part = np.array([chem.partial(smi_at(int(j)), z_e) for j in keep])
            full = np.full(len(keep), np.nan)
            want = set()
            for n in (50, 100, 150, 200, 300, 500):          # top 40 by partial within every pre-filter size
                want.update(np.argsort(-part[:n], kind="stable")[:40].tolist())
            for i in sorted(want):
                f = chem.fingerprint(smi_at(int(keep[i])))
                if f is not None:
                    full[i] = (f @ z - mu) / sd
            out[(e, sign)] = dict(rows=lo + keep, doc=doc[keep], part=part, full=full,
                                  smiles=[smi_at(int(j)) for j in keep])
    return m, out


def build():
    from casmi.panel import PANEL
    CACHE.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(PANEL / "panel.parquet"); p["molecule"] = p.molecule.astype(str)
    _init()
    args = [(m, float(M)) for m, M in zip(p.molecule, p.measured_neutral_mass) if m in G["z"]]
    G.clear(); t0 = time.time()
    with Pool(4) as pool:
        for i, (m, out) in enumerate(pool.imap_unordered(_build_one, args)):
            pickle.dump(out, open(CACHE / f"{m}.pkl", "wb"))
            if i % 25 == 0:
                print(f"{i + 1}/{len(args)} {time.time() - t0:.0f}s", flush=True)
    print("built", len(args), "molecules", round(time.time() - t0), "s")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()


def evaluate(variants: list[dict]) -> None:
    from casmi.edit_overlay import Chemistry
    from casmi.panel import load_candidates
    from panel_fp_matrix import PanelFP, score_frame
    from pubchem_relatives_merge import lists_and_gate, merge, rr
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    L = np.load(ROOT / "artifacts" / "panel_logits_foldsafe.npz")
    zs = [0.5 * (L[f"single{k}"].astype(np.float64) + L[f"merged{k}"].astype(np.float64)) for k in range(5)]
    Z = dict(zip([str(k) for k in L["keys"]], np.mean(zs, axis=0)))
    c = load_candidates()
    c["raw"] = score_frame(c, Z, PanelFP())
    st = c.groupby("molecule").raw.agg(["mean", "std"])
    lists = lists_and_gate(c, 0.4); mols = sorted(lists)
    base2 = [rr([k for k, _ in lists[m][0]], m) for m in mols]
    p = pd.read_parquet(ROOT.parents[0] / "casmi_data" / "ext" / "panel" / "panel.parquet"); p["molecule"] = p.molecule.astype(str)
    M = dict(zip(p.molecule, p.measured_neutral_mass))
    cache = {m: pickle.load(open(CACHE / f"{m}.pkl", "rb")) for m in mols if (CACHE / f"{m}.pkl").is_file()}
    prod_memo, fp_memo = {}, {}

    def products(m, e, sign, row, smi):
        k = (m, row, e, sign)
        if k not in prod_memo:
            rx = chem.fwd[e][1] if sign < 0 else chem.strip[e][1]
            out = []
            for key, ps in chem.products(smi, rx, cap=50).items():
                if key not in fp_memo:
                    pm = chem.mass(ps)
                    f = chem.fingerprint(ps) if np.isfinite(pm) and abs(pm - M[m]) <= M[m] * 15e-6 else None
                    fp_memo[key] = f
                f = fp_memo[key]
                if f is not None:
                    out.append((key, float((f @ Z[m] - st.loc[m, "mean"]) / st.loc[m, "std"])))
            prod_memo[k] = out
        return prod_memo[k]

    for v in variants:
        t0 = time.time()
        r2, r3, gen = [], [], 0
        for m in mols:
            acc = {}
            for (e, sign), w in cache.get(m, {}).items():
                nd = min(v["n_doc"], len(w["rows"]))
                part = w["part"][:nd]
                top = np.argsort(-part, kind="stable")[: v["n_full"]]
                top = [i for i in top if np.isfinite(w["full"][i])]
                top.sort(key=lambda i: -(w["full"][i] + v["w_sel"] * w["doc"][i]))
                for i in top[: v["k"]]:
                    for key, zp in products(m, e, sign, int(w["rows"][i]), w["smiles"][i]):
                        a = acc.setdefault(key, dict(zp=zp, doc=0.0, routes=set()))
                        a["doc"] = max(a["doc"], float(w["doc"][i])); a["routes"].add((int(w["rows"][i]), e, sign))
            gen += m in acc
            rows = [dict(key=k, z_p=a["zp"], rel_doc=a["doc"] + (v["w_sup"] / v["w_rel"]) * np.log(len(a["routes"])),
                         rel_z=0.0, loss_ok=0.0) for k, a in acc.items()]
            pr = pd.DataFrame(rows, columns=["key", "z_p", "rel_doc", "rel_z", "loss_ok"])
            r2.append(rr(merge(lists[m][0], pr, v.get("beta", 0.0), v["w_rel"]), m))
            r3.append(rr(merge(lists[m][1], pr, v.get("beta", 0.0), v["w_rel"]), m))
        h = [np.mean([r3[i] for i in range(len(mols)) if i % 2 == j]) for j in (0, 1)]
        print(f"{v}: generated {gen:3d} | class-2 {np.mean(r2) - np.mean(base2):+.4f} (hurt {sum(x < y for x, y in zip(r2, base2))}) "
              f"| class-3 {np.mean(r3):.4f} (halves {h[0]:.3f} / {h[1]:.3f}) | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__" and sys.argv[1] == "eval":
    base = dict(n_doc=300, n_full=20, w_sel=0.3, k=3, w_rel=0.2, w_sup=0.5)
    grid = [base] + [dict(base, n_doc=n) for n in (50, 100, 150, 200, 500)] + [dict(base, n_full=n) for n in (10, 40)] + \
           [dict(base, w_sel=w) for w in (0.0, 0.6, 1.0)] + [dict(base, n_doc=100, w_sel=0.6), dict(base, n_doc=150, n_full=40)]
    evaluate(grid)


if __name__ == "__main__" and sys.argv[1] == "eval2":
    grid = []
    for nd, nf in ((300, 40), (150, 40), (100, 20), (200, 40)):
        for w_rel, beta in ((0.2, 0.0), (0.15, 0.0), (0.1, 0.0), (0.2, -0.5), (0.2, -1.0), (0.15, -0.5)):
            grid.append(dict(n_doc=nd, n_full=nf, w_sel=0.3, k=3, w_rel=w_rel, w_sup=0.5, beta=beta))
    evaluate(grid)
