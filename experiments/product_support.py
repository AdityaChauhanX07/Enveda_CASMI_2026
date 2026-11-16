"""Spectral support for generated products: can it pick the one product a9 inserts (class 3)?

Dumps state_<tag>_c2.pkl / _c3.pkl from edit_overlay_panel.py (with product SMILES). For the best `top` products of
each list (by gate score), support = best modified cosine between a query spectrum of the molecule and a same-adduct
train.parquet spectrum of any one-edit relative of the product (30 edit rules, standard InChIKey14; the molecule's
own answer spectra are never used). Rules are replayed with a9's guard (min_slot 1, max_insert 1):
  - products re-ordered by g + beta * support (support below sigma counts as 0)
  - only products with support >= sigma inserted
Class-2 change against the original fork-like lists, also with the answer's gate documentation capped at d (the a7
failure scenario of safe_insertion_lowdoc.py).

usage: product_support.py [tag=t10] [top=10]
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi import library_overlay as lo  # noqa: E402
from casmi.data import DATA  # noqa: E402
from casmi.edit_overlay import merge_sources  # noqa: E402
import insertion_replay as ir  # noqa: E402
import library_analogue_feasibility as laf  # noqa: E402


def merged(st):
    return dict(st, scored=[(g, k, m) for g, _, k, m in merge_sources([(g, None, k, m) for g, k, m in st["scored"]], 0.0)])


def load_refs_with_precursor(keys, adducts, cap=50):
    import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
    pf = pq.ParquetFile(DATA / "train.parquet"); kset = pa.array(sorted(keys)); aset = pa.array(sorted(adducts))
    p = dict(lo.PARAMS); refs, count = {}, {}
    for g in range(pf.num_row_groups):
        meta = pf.read_row_group(g, columns=["inchikey14", "adduct"])
        mask = pc.and_(pc.is_in(meta["inchikey14"], value_set=kset), pc.is_in(meta["adduct"], value_set=aset))
        idx = np.flatnonzero(np.asarray(mask.to_numpy(zero_copy_only=False), dtype=bool))
        if not len(idx):
            continue
        ks = meta["inchikey14"].take(pa.array(idx)).to_pylist(); ad = meta["adduct"].take(pa.array(idx)).to_pylist()
        want = {}
        for i, k, a in zip(idx, ks, ad):
            if count.get((k, a), 0) < cap:
                count[(k, a)] = count.get((k, a), 0) + 1; want[int(i)] = (k, a)
        off = 0
        for b in pf.iter_batches(batch_size=8192, row_groups=[g], columns=["precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]):
            rows = [i for i in range(off, off + b.num_rows) if i in want]
            if rows:
                d = b.take(pa.array([i - off for i in rows])).to_pydict()
                for i, pm, mz, it in zip(rows, d["precursor_mz"], d["ms2_mzs"], d["ms2_normalized_intensities"]):
                    if pm is None or mz is None:
                        continue
                    s = laf.prep(dict(mz=mz, it=it, pm=pm), p)
                    if len(s[1]):
                        refs.setdefault(want[i], []).append(s)
            off += b.num_rows
    return refs


def insert_with(st, beta, sigma, gate, sup, rule):
    sc = []
    for g, k, meta in st["scored"]:
        s = sup.get(k, 0.0)
        s = s if s >= sigma else 0.0
        if gate and s <= 0.0:
            continue
        sc.append((g + beta * s, k, meta))
    return ir.insert(dict(st, scored=sc), rule)


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag, top = args.get("tag", "t10"), int(args.get("top", 10))
    t0 = time.time()
    states = {s: pickle.load(open(ROOT / "artifacts" / f"state_{tag}_{s}.pkl", "rb")) for s in ("c2", "c3")}
    work, prod_of = [], {}
    for s, st_all in states.items():
        for m, st in st_all.items():
            st_m = merged(st)
            for g, k, meta in sorted(st_m["scored"], key=lambda x: -x[0])[:top]:
                smi = st["smiles"].get(k)
                if smi:
                    work.append((m, k, smi)); prod_of.setdefault((s, m), []).append(k)
    work = list({(m, k): (m, 0, smi) for m, k, smi in work}.items())
    from multiprocessing import Pool
    with Pool(4, initializer=laf._init) as pool:
        out = pool.map(laf._relatives, [(m, k, smi) for (m, k), (_, _, smi) in work], chunksize=20)
    rel = dict(out)                                   # (molecule, product key) -> {relative std key: edit}
    print(f"{len(rel)} products, relatives generated in {time.time() - t0:.0f} s")
    import pandas as pd
    t = pd.read_parquet(ROOT / "artifacts" / "np_test.parquet", columns=["molecule_id", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    Q = {}
    for r in t.itertuples(index=False):
        s_ = laf.prep(dict(mz=r.ms2_mzs, it=r.ms2_normalized_intensities, pm=r.precursor_mz), dict(lo.PARAMS))
        if len(s_[1]):
            Q.setdefault(r.molecule_id, {}).setdefault(r.adduct, []).append(s_)
    keys = {k for v in rel.values() for k in v} - set(states["c3"])
    refs = load_refs_with_precursor(keys, {a for q in Q.values() for a in q})
    print(f"reference spectra for {len({k for k, _ in refs})} relative structures in {time.time() - t0:.0f} s")
    sup = {}
    for (m, k), rk in rel.items():
        best = 0.0
        for r_ in rk:
            if r_ == m:
                continue
            for a, qs in Q.get(m, {}).items():
                for ref in refs.get((r_, a), ()):
                    for q in qs:
                        c, n = laf.modcos(q, ref)
                        if n >= 3 and c > best:
                            best = c
        sup[(m, k)] = best
    pickle.dump(sup, open(ROOT / "artifacts" / f"product_support_{tag}.pkl", "wb"))
    print(f"support computed in {time.time() - t0:.0f} s")
    # how often is the answer supported among class-3 products, vs the other top products
    a_sup = [sup.get((m, m), None) for m in states["c3"] if (m, m) in sup]
    o_sup = [v for (m, k), v in sup.items() if k != m and m in states["c3"]]
    print(f"class-3 answers among top-{top} products: {len(a_sup)}; support >= 0.6 {np.mean(np.array(a_sup) >= 0.6):.2f}, >= 0.7 {np.mean(np.array(a_sup) >= 0.7):.2f} | "
          f"other products {np.mean(np.array(o_sup) >= 0.6):.2f}, {np.mean(np.array(o_sup) >= 0.7):.2f}")
    from casmi.panel import load_candidates
    from casmi.metric import ik14
    from library_overlay_panel import panel_lists
    c = load_candidates(); doc_ans = c[c.is_truth].groupby("molecule").doc.max().to_dict()
    _, _, sub = panel_lists()
    orig = {m: [ik14(x) for x in s_.split(";")] for m, s_ in zip(sub.molecule_id, sub.smiles)}
    rule = {"min_slot": 1, "max_insert": 1}
    N = 250
    print("rule (a9 guard: min_slot 1, max_insert 1)      | class 3 | class 2 (doc as is / <= 6 / <= 4) | net 10 % / 25 % (doc <= 6)")
    for beta, sigma, gate in ((0.0, 1.1, False), (0.5, 0.6, False), (1.0, 0.6, False), (2.0, 0.6, False), (1.0, 0.7, False),
                              (2.0, 0.7, False), (0.0, 0.6, True), (0.0, 0.7, True), (1.0, 0.6, True)):
        s3 = {m: {k: sup.get((m, k), 0.0) for k in prod_of.get(("c3", m), [])} for m in states["c3"]}
        r3 = sum(ir.rr(insert_with(merged(st), beta, sigma, gate, s3[m], rule), m) for m, st in states["c3"].items()) / N
        d2 = {}
        for d in (None, 6, 4):
            tot = 0.0
            for m, st in states["c2"].items():
                stm = merged(st); ck = stm["cand_keys"]; top_ = list(stm["top"])
                if d is not None and m in ck[:3] and ck.index(m) < len(top_):
                    top_[ck.index(m)] -= 0.4 * max(0.0, doc_ans.get(m, 0.0) - d)
                s2 = {k: sup.get((m, k), 0.0) for k in prod_of.get(("c2", m), [])}
                tot += ir.rr(insert_with(dict(stm, top=top_), beta, sigma, gate, s2, rule), m) - ir.rr(orig[m], m)
            d2[d] = tot / N
        name = "baseline (g only)" if sigma > 1 else f"{'gate' if gate else 're-rank'} beta {beta} sigma {sigma}"
        print(f"{name:47s} | {r3:.4f}  | {d2[None]:+.4f} / {d2[6]:+.4f} / {d2[4]:+.4f}        | {0.9 * d2[6] + 0.1 * r3:+.4f} / {0.75 * d2[6] + 0.25 * r3:+.4f}")
