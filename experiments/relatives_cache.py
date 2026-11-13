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

    def family(m, v):
        acc = {}
        for (e, sign), w in cache.get(m, {}).items():
            nd = min(v["n_doc"], len(w["rows"]))
            top = np.argsort(-w["part"][:nd], kind="stable")[: v["n_full"]]
            top = [i for i in top if np.isfinite(w["full"][i])]
            top.sort(key=lambda i: -(w["full"][i] + v["w_sel"] * w["doc"][i]))
            for i in top[: v["k"]]:
                for key, zp in products(m, e, sign, int(w["rows"][i]), w["smiles"][i]):
                    a = acc.setdefault(key, dict(zp=zp, doc=0.0, routes=set()))
                    a["doc"] = max(a["doc"], float(w["doc"][i])); a["routes"].add((int(w["rows"][i]), e, sign))
        return acc

    if variants and variants[0].get("rerank"):
        for v in variants:
            r2 = []
            for m in mols:
                acc = family(m, v)
                lst = lists[m][0]
                keys = [k for k, _ in lst]; g = [x for _, x in lst]
                kk = v["rerank"]
                bonus = [(v["w_rel"] * acc[k]["doc"] + v["w_sup"] * np.log(len(acc[k]["routes"]))) if k in acc else 0.0 for k in keys[:kk]]
                if v["mode"] == "gate":
                    sc = [g[i] + v["lam"] * bonus[i] for i in range(len(bonus))]
                else:
                    sc = [-np.log1p(i) + v["lam"] * bonus[i] for i in range(len(bonus))]
                order = sorted(range(len(sc)), key=lambda i: -sc[i])
                r2.append(rr([keys[i] for i in order] + keys[kk:], m))
            d = np.array(r2) - np.array(base2)
            print(f"{v}: class-2 {np.mean(r2):.4f} vs {np.mean(base2):.4f} ({np.mean(d):+.4f}; up {(d > 0).sum()}, down {(d < 0).sum()})", flush=True)
        return

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
                sel = v.get("sel", "full")
                if sel == "part":                  # ECFP4 partial only, standardised over the pre-filtered rows
                    ps = (part - part.mean()) / (part.std() + 1e-9)
                    top = list(np.argsort(-(ps + v["w_sel"] * w["doc"][:nd]), kind="stable")[: v["n_full"]])
                    top.sort(key=lambda i: -(ps[i] + v["w_sel"] * w["doc"][i]))
                elif sel == "full_local" and top:   # full score standardised over the n_full candidates (no window sample)
                    fv = np.array([w["full"][i] for i in top]); fz = (fv - fv.mean()) / (fv.std() + 1e-9)
                    loc = dict(zip(top, fz))
                    top.sort(key=lambda i: -(loc[i] + v["w_sel"] * w["doc"][i]))
                else:
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


if __name__ == "__main__" and sys.argv[1] == "family":
    base = dict(n_doc=150, n_full=40, w_sel=0.3, k=3, w_rel=0.15, w_sup=0.5)
    grid = [dict(base, rerank=kk, mode=mode, lam=lam) for kk in (5, 10) for mode in ("rank", "gate") for lam in (0.25, 0.5, 1.0)]
    evaluate(grid)


if __name__ == "__main__" and sys.argv[1] == "family0":
    base = dict(n_doc=150, n_full=40, w_sel=0.3, k=3, w_rel=0.15, w_sup=0.5, mode="gate")
    evaluate([dict(base, rerank=5, lam=0.0), dict(base, rerank=5, lam=0.25), dict(base, rerank=5, lam=0.1)])


if __name__ == "__main__" and sys.argv[1] == "sel":
    base = dict(n_doc=150, n_full=40, w_sel=0.3, k=3, w_rel=0.15, w_sup=0.5)
    evaluate([base, dict(base, sel="full_local"), dict(base, sel="part"), dict(base, sel="part", w_sel=0.1), dict(base, sel="part", k=4)])


def window_contributions(v: dict) -> None:
    """For the chosen selection rule: per (edit, direction) window, products made, answers rebuilt, answers rebuilt
    by no other window, and the class-3 / class-2 effect of dropping the window."""
    from casmi.edit_overlay import Chemistry
    from casmi.panel import load_candidates
    from panel_fp_matrix import PanelFP, score_frame
    from pubchem_relatives_merge import lists_and_gate, merge, rr
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    L = np.load(ROOT / "artifacts" / "panel_logits_foldsafe.npz")
    zs = [0.5 * (L[f"single{k}"].astype(np.float64) + L[f"merged{k}"].astype(np.float64)) for k in range(5)]
    Z = dict(zip([str(k) for k in L["keys"]], np.mean(zs, axis=0)))
    c = load_candidates(); c["raw"] = score_frame(c, Z, PanelFP())
    st = c.groupby("molecule").raw.agg(["mean", "std"])
    lists = lists_and_gate(c, 0.4); mols = sorted(lists)
    p = pd.read_parquet(EXT / "panel" / "panel.parquet"); p["molecule"] = p.molecule.astype(str)
    M = dict(zip(p.molecule, p.measured_neutral_mass))
    cache = {m: pickle.load(open(CACHE / f"{m}.pkl", "rb")) for m in mols if (CACHE / f"{m}.pkl").is_file()}
    per = {}                                           # molecule -> window -> [(key, zp, row, doc)]
    for m in mols:
        per[m] = {}
        for (e, sign), w in cache.get(m, {}).items():
            nd = min(v["n_doc"], len(w["rows"]))
            top = [i for i in np.argsort(-w["part"][:nd], kind="stable")[: v["n_full"]] if np.isfinite(w["full"][i])]
            if not top:
                continue
            fv = np.array([w["full"][i] for i in top]); fz = dict(zip(top, (fv - fv.mean()) / (fv.std() + 1e-9)))
            top.sort(key=lambda i: -(fz[i] + v["w_sel"] * w["doc"][i]))
            out = []
            rx = chem.fwd[e][1] if sign < 0 else chem.strip[e][1]
            for i in top[: v["k"]]:
                for key, ps in chem.products(w["smiles"][i], rx, cap=50).items():
                    pm = chem.mass(ps)
                    f = chem.fingerprint(ps) if np.isfinite(pm) and abs(pm - M[m]) <= M[m] * 15e-6 else None
                    if f is not None:
                        out.append((key, float((f @ Z[m] - st.loc[m, "mean"]) / st.loc[m, "std"]), int(w["rows"][i]), float(w["doc"][i])))
            per[m][(e, sign)] = out

    def score(drop=None):
        r2, r3 = [], []
        for m in mols:
            acc = {}
            for win, out in per[m].items():
                if win == drop:
                    continue
                for key, zp, row, doc in out:
                    a = acc.setdefault(key, dict(zp=zp, doc=0.0, routes=set()))
                    a["doc"] = max(a["doc"], doc); a["routes"].add((row,) + win)
            pr = pd.DataFrame([dict(key=k, z_p=a["zp"], rel_doc=a["doc"] + (v["w_sup"] / v["w_rel"]) * np.log(len(a["routes"])),
                                    rel_z=0.0, loss_ok=0.0) for k, a in acc.items()], columns=["key", "z_p", "rel_doc", "rel_z", "loss_ok"])
            r2.append(rr(merge(lists[m][0], pr, 0.0, v["w_rel"]), m) - rr([k for k, _ in lists[m][0]], m))
            r3.append(rr(merge(lists[m][1], pr, 0.0, v["w_rel"]), m))
        return np.sum(r2) / N_PANEL, np.sum(r3) / N_PANEL

    base2, base3 = score()
    print(f"all windows: class-2 {base2:+.4f} class-3 {base3:.4f}")
    wins = sorted({w for m in per for w in per[m]})
    for win in wins:
        made = sum(len(per[m].get(win, [])) for m in mols)
        hits = [m for m in mols if any(k == m for k, *_ in per[m].get(win, []))]
        only = [m for m in hits if not any(k == m for w2, o in per[m].items() if w2 != win for k, *_ in o)]
        d2, d3 = score(drop=win)
        print(f"{win[0]:14s} {'lighter' if win[1] < 0 else 'heavier':8s} products {made:6d} | answers {len(hits):3d} (only route {len(only):2d}) | "
              f"drop it: class-2 {d2 - base2:+.4f} class-3 {d3 - base3:+.4f}", flush=True)


N_PANEL = 250
if __name__ == "__main__" and sys.argv[1] == "windows":
    window_contributions(dict(n_doc=150, n_full=40, w_sel=0.3, k=3, w_rel=0.15, w_sup=0.5))


# ---------------- task 2: extension cache for new natural-product edits (heavier relatives -> strip) ----------------
NEW_MASS = {"sulfate": 79.95681, "galloyl": 152.01096, "C-hexose(ar)": 162.05282, "coumaroyl": 146.03678,
            "feruloyl": 176.04734, "caffeoyl": 162.03169, "O-prenyl": 68.06260, "sinapoyl": 206.05791}
CACHE_NEW = ROOT / "artifacts" / "relatives_cache_new"


def _build_new_one(args):
    """Same rows/scores as _build_one, for the new edits' heavier-relative windows only."""
    m, M = args
    if not G:
        _init()
    chem, tier, z = G["chem"], G["tier"], G["z"][m]
    z_e = z[chem.sel_e]
    out = {}
    for e, d in NEW_MASS.items():
        lo, hi = tier.window(M + d, 10.0)
        if hi <= lo:
            continue
        doc = np.asarray(tier.lsid[lo:hi], np.float64) + np.asarray(tier.lpmid[lo:hi], np.float64)
        offs = np.asarray(tier.off[lo:hi + 1], np.int64); blob = bytes(tier.smi[offs[0]:offs[-1]])
        smi_at = lambda j: blob[offs[j] - offs[0]:offs[j + 1] - offs[0]].decode()
        keep = np.argsort(-doc, kind="stable")[:N_DOC]
        part = np.array([chem.partial(smi_at(int(j)), z_e) for j in keep])
        full = np.full(len(keep), np.nan)
        want = set()
        for n in (50, 100, 150, 200, 300, 500):
            want.update(np.argsort(-part[:n], kind="stable")[:40].tolist())
        for i in sorted(want):
            f = chem.fingerprint(smi_at(int(keep[i])))
            if f is not None:
                full[i] = float(f @ z)                      # raw score; standardised over candidates at evaluation
        out[(e, +1)] = dict(rows=lo + keep, doc=doc[keep], part=part, full=full, smiles=[smi_at(int(j)) for j in keep])
    return m, out


def build_new():
    from casmi.panel import PANEL
    CACHE_NEW.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(PANEL / "panel.parquet"); p["molecule"] = p.molecule.astype(str)
    _init()
    args = [(m, float(M)) for m, M in zip(p.molecule, p.measured_neutral_mass) if m in G["z"]]
    G.clear(); t0 = time.time()
    with Pool(4) as pool:
        for i, (m, out) in enumerate(pool.imap_unordered(_build_new_one, args)):
            pickle.dump(out, open(CACHE_NEW / f"{m}.pkl", "wb"))
            if i % 25 == 0:
                print(f"{i + 1}/{len(args)} {time.time() - t0:.0f}s", flush=True)
    print("built new", len(args), "molecules", round(time.time() - t0), "s")


if __name__ == "__main__" and sys.argv[1] == "build_new":
    build_new()


DEAD = {("malonyl", -1), ("malonyl", +1), ("prenyl(ar)", -1), ("prenyl(ar)", +1), ("glucuronide", +1),
        ("pentose", -1), ("C-methyl(ar)", -1)}


def evaluate_sets(v: dict) -> None:
    from rdkit.Chem import AllChem
    from casmi.edit_overlay import Chemistry
    from casmi.panel import load_candidates
    from new_edits_coverage import NEW_STRIP
    from panel_fp_matrix import PanelFP, score_frame
    from pubchem_relatives_merge import lists_and_gate, merge, rr
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    new_rx = {e: AllChem.ReactionFromSmarts(s) for e, s in NEW_STRIP.items()}
    L = np.load(ROOT / "artifacts" / "panel_logits_foldsafe.npz")
    zs = [0.5 * (L[f"single{k}"].astype(np.float64) + L[f"merged{k}"].astype(np.float64)) for k in range(5)]
    Z = dict(zip([str(k) for k in L["keys"]], np.mean(zs, axis=0)))
    c = load_candidates(); c["raw"] = score_frame(c, Z, PanelFP())
    st = c.groupby("molecule").raw.agg(["mean", "std"])
    lists = lists_and_gate(c, 0.4); mols = sorted(lists)
    p = pd.read_parquet(EXT / "panel" / "panel.parquet"); p["molecule"] = p.molecule.astype(str)
    M = dict(zip(p.molecule, p.measured_neutral_mass))
    per = {}
    t0 = time.time()
    for m in mols:
        per[m] = {}
        sources = []
        if (CACHE / f"{m}.pkl").is_file():
            sources += [(win, w, False) for win, w in pickle.load(open(CACHE / f"{m}.pkl", "rb")).items()]
        if (CACHE_NEW / f"{m}.pkl").is_file():
            sources += [(("new:" + win[0], win[1]), w, True) for win, w in pickle.load(open(CACHE_NEW / f"{m}.pkl", "rb")).items()]
        for win, w, is_new in sources:
            nd = min(v["n_doc"], len(w["rows"]))
            top = [i for i in np.argsort(-w["part"][:nd], kind="stable")[: v["n_full"]] if np.isfinite(w["full"][i])]
            if not top:
                continue
            fv = np.array([w["full"][i] for i in top]); fz = dict(zip(top, (fv - fv.mean()) / (fv.std() + 1e-9)))
            top.sort(key=lambda i: -(fz[i] + v["w_sel"] * w["doc"][i]))
            e, sign = win
            rx = new_rx[e[4:]] if is_new else (chem.fwd[e][1] if sign < 0 else chem.strip[e][1])
            out = []
            for i in top[: v["k"]]:
                for key, ps in chem.products(w["smiles"][i], rx, cap=50).items():
                    pm = chem.mass(ps)
                    f = chem.fingerprint(ps) if np.isfinite(pm) and abs(pm - M[m]) <= M[m] * 15e-6 else None
                    if f is not None:
                        out.append((key, float((f @ Z[m] - st.loc[m, "mean"]) / st.loc[m, "std"]), int(w["rows"][i]), float(w["doc"][i])))
            per[m][win] = out
    print(f"products computed in {time.time() - t0:.0f}s", flush=True)
    allwins = sorted({w for m in per for w in per[m]})
    old = [w for w in allwins if not w[0].startswith("new:")]
    new = [w for w in allwins if w[0].startswith("new:")]

    def score(keep, ms=mols):
        keep = set(keep); r2, r3 = [], []
        for m in ms:
            acc = {}
            for win, out in per[m].items():
                if win not in keep:
                    continue
                for key, zp, row, doc in out:
                    a = acc.setdefault(key, dict(zp=zp, doc=0.0, routes=set()))
                    a["doc"] = max(a["doc"], doc); a["routes"].add((row,) + win)
            pr = pd.DataFrame([dict(key=k, z_p=a["zp"], rel_doc=a["doc"] + (v["w_sup"] / v["w_rel"]) * np.log(len(a["routes"])),
                                    rel_z=0.0, loss_ok=0.0) for k, a in acc.items()], columns=["key", "z_p", "rel_doc", "rel_z", "loss_ok"])
            r2.append(rr(merge(lists[m][0], pr, 0.0, v["w_rel"]), m) - rr([k for k, _ in lists[m][0]], m))
            r3.append(rr(merge(lists[m][1], pr, 0.0, v["w_rel"]), m))
        n = N_PANEL * len(ms) / len(mols)
        return np.sum(r2) / n, np.sum(r3) / n, sum(x < 0 for x in r2), sum(m in {k for o in per[m].values() for k, *_ in o if True} for m in ms)

    def show(name, keep):
        d2, c3, hurt, _ = score(keep)
        h = [score(keep, mols[j::2]) for j in (0, 1)]
        print(f"{name:44s} windows {len(keep):2d} | class-2 {d2:+.4f} (hurt {hurt}) | class-3 {c3:.4f} | "
              f"net10 {0.9 * d2 + 0.1 * c3:+.4f} net25 {0.75 * d2 + 0.25 * c3:+.4f} | halves class-3 {h[0][1]:.3f} / {h[1][1]:.3f}", flush=True)
        return c3

    base = [w for w in old]
    trimmed = [w for w in old if w not in DEAD]
    show("A: current 26 windows", base)
    c_trim = show("B: minus the 7 dead windows", trimmed)
    show("C: B + all new strip windows", trimmed + new)
    gains = {}
    for w in new:
        gains[w] = show(f"   B + {w[0][4:]}", trimmed + [w]) - c_trim
    good = [w for w, g in gains.items() if g > 0.001]
    show("E: B + new windows that help (> +0.001 alone)", trimmed + good)
    print("helpful new windows:", [w[0][4:] for w in good])


if __name__ == "__main__" and sys.argv[1] == "sets":
    evaluate_sets(dict(n_doc=150, n_full=40, w_sel=0.3, k=3, w_rel=0.15, w_sup=0.5))
