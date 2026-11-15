"""Feasibility of library-analogue support: does a candidate's one-edit relative with public spectra match the query?

Panel lists (fork-like order, top 10 per molecule, class 2: the answer's own spectra hidden). For each candidate,
every one-edit relative (the edit overlay's 13 edits each way + 4 extra strip edits) is looked up in train.parquet;
support = best modified cosine between a query spectrum and a same-adduct spectrum of any relative (shift = the
precursor difference). Reports how often answers and wrong candidates get support, and the separation.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi import library_overlay as lo  # noqa: E402
from casmi.data import DATA  # noqa: E402
from casmi.edit_overlay import Chemistry  # noqa: E402
from library_overlay_panel import panel_lists  # noqa: E402

TOP = 10


def prep(r, p):
    mz, w = lo._prepare(r["mz"], r["it"], r["pm"], p["top_n"], p["min_rel"])
    return (float(r["pm"]), mz, w)


def modcos(a, b, tol=0.01):
    (pa, ma, wa), (pb, mb, wb) = a, b
    if len(ma) == 0 or len(mb) == 0:
        return 0.0, 0
    shift = pa - pb
    d0 = np.abs(ma[:, None] - mb[None, :]) <= tol
    d1 = np.abs(ma[:, None] - (mb[None, :] + shift)) <= tol if abs(shift) > tol else np.zeros_like(d0)
    ii, jj = np.nonzero(d0 | d1)
    if len(ii) == 0:
        return 0.0, 0
    prod = wa[ii] * wb[jj]
    ua, ub, tot, n = set(), set(), 0.0, 0
    for k in np.argsort(-prod, kind="stable"):
        i, j = ii[k], jj[k]
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); tot += prod[k]; n += 1
    return float(min(1.0, tot)), n


_W = {}


def _init():
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    _W["chem"] = chem
    _W["rules"] = [("+" + e, rx) for e, (_, rx) in chem.fwd.items()] + [("-" + e, rx) for e, (_, rx) in chem.strip.items()] + \
                  [("-" + e, rx) for e, (_, rx) in chem.rel_extra.items()]


def _relatives(task):
    """One-edit relatives of one candidate as standard InChIKey14 (the library's key; no tautomer step)."""
    m, i, s = task
    Chem = _W["chem"].Chem
    mol = Chem.MolFromSmiles(s)
    out = {}
    if mol is None:
        return (m, i), out
    for name, rx in _W["rules"]:
        for ps in rx.RunReactants((mol,), maxProducts=50):
            q = ps[0]
            try:
                Chem.SanitizeMol(q)
                k = Chem.MolToInchiKey(q)[:14]
            except Exception:
                continue
            if k:
                out.setdefault(k, name)
    return (m, i), out


if __name__ == "__main__":
    t0 = time.time()
    art, sol, sub = panel_lists()
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    Chem = chem.Chem
    std = lambda s: (Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] if Chem.MolFromSmiles(s) is not None else "")
    rules = [("+" + e, rx) for e, (_, rx) in chem.fwd.items()] + [("-" + e, rx) for e, (_, rx) in chem.strip.items()] + \
            [("-" + e, rx) for e, (_, rx) in chem.rel_extra.items()]
    lists = {m: s.split(";")[:TOP] for m, s in zip(sub.molecule_id, sub.smiles)}
    from multiprocessing import Pool
    work = [(m, i, s) for m, cand in lists.items() for i, s in enumerate(cand)]
    with Pool(4, initializer=_init) as pool:          # (molecule, position) -> {relative std key: edit}
        rel = dict(pool.map(_relatives, work, chunksize=20))
    print(f"relatives generated in {time.time() - t0:.0f} s; mean per candidate {np.mean([len(v) for v in rel.values()]):.1f}")
    # query spectra with precursor (np-examples rows of each molecule)
    t = pd.read_parquet(art / "np_test.parquet", columns=["molecule_id", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    p = dict(lo.PARAMS)
    Q = {}
    for r in t.itertuples(index=False):
        s = prep(dict(mz=r.ms2_mzs, it=r.ms2_normalized_intensities, pm=r.precursor_mz), p)
        if len(s[1]):
            Q.setdefault(r.molecule_id, {}).setdefault(r.adduct, []).append(s)
    keys = {k for v in rel.values() for k in v} - set(lists)          # the answers' own spectra stay hidden
    adducts = {a for q in Q.values() for a in q}
    # reference spectra with precursor: reuse the overlay's reader, keeping precursor via a small wrapper
    import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
    pf = pq.ParquetFile(DATA / "train.parquet"); kset = pa.array(sorted(keys)); aset = pa.array(sorted(adducts))
    refs, count = {}, {}
    for g in range(pf.num_row_groups):
        meta = pf.read_row_group(g, columns=["inchikey14", "adduct"])
        mask = pc.and_(pc.is_in(meta["inchikey14"], value_set=kset), pc.is_in(meta["adduct"], value_set=aset))
        idx = np.flatnonzero(np.asarray(mask.to_numpy(zero_copy_only=False), dtype=bool))
        if not len(idx):
            continue
        ks = meta["inchikey14"].take(pa.array(idx)).to_pylist(); ad = meta["adduct"].take(pa.array(idx)).to_pylist()
        want = {}
        for i, k, a in zip(idx, ks, ad):
            if count.get((k, a), 0) < 50:
                count[(k, a)] = count.get((k, a), 0) + 1; want[int(i)] = (k, a)
        off = 0
        for b in pf.iter_batches(batch_size=8192, row_groups=[g], columns=["precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]):
            rows = [i for i in range(off, off + b.num_rows) if i in want]
            if rows:
                d = b.take(pa.array([i - off for i in rows])).to_pydict()
                for i, pm, mz, it in zip(rows, d["precursor_mz"], d["ms2_mzs"], d["ms2_normalized_intensities"]):
                    if pm is None or mz is None:
                        continue
                    s = prep(dict(mz=mz, it=it, pm=pm), p)
                    if len(s[1]):
                        refs.setdefault(want[i], []).append(s)
            off += b.num_rows
    print(f"reference spectra for {len({k for k, _ in refs})} relative structures in {time.time() - t0:.0f} s")
    rows = []
    for m, cand in lists.items():
        q = Q.get(m, {})
        for i, s in enumerate(cand):
            best, bn, n_rel_spec = 0.0, 0, 0
            for rk in rel[(m, i)]:
                if rk == m:
                    continue
                hit = False
                for a, qs in q.items():
                    for r in refs.get((rk, a), ()):
                        hit = True
                        for qq in qs:
                            c, n = modcos(qq, r)
                            if c > best:
                                best, bn = c, n
                n_rel_spec += int(hit)
            rows.append(dict(molecule=m, rank0=i, truth=(std(s) == m), n_rel=len(rel[(m, i)]), n_rel_spec=n_rel_spec, support=best, matched=bn))
    R = pd.DataFrame(rows)
    R.to_parquet(art / "library_analogue_support.parquet")
    print(f"done in {time.time() - t0:.0f} s")
    for name, g in (("answers", R[R.truth]), ("wrong", R[~R.truth])):
        print(f"{name}: {len(g)}, with a library-spectra relative {np.mean(g.n_rel_spec > 0):.2f}, "
              f"support >= 0.5 {np.mean(g.support >= 0.5):.2f}, >= 0.6 {np.mean(g.support >= 0.6):.2f}, >= 0.7 {np.mean(g.support >= 0.7):.2f}")
    x = R[R.n_rel_spec > 0]
    r = x.support.rank().to_numpy(); y = x.truth.to_numpy(); n1, n0 = y.sum(), (~y).sum()
    print("AUC of support among candidates with a library-spectra relative: %.3f" % ((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)))
