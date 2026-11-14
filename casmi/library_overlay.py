"""Library-consistency overlay for a finished submission (self-contained: numpy, pandas, pyarrow, RDKit).

A test molecule in class 2 or 3 has no public spectra, and a class-1 answer's own reference spectra match the
query. So a candidate whose public reference spectra with the query's adduct ALL fail to match any query
spectrum (best cosine < tau) is very unlikely to be the answer. Such candidates move down `down` places (or to
the end with down < 0); every other candidate keeps its relative order, and no candidate is added or removed.

Panel (experiments/library_overlay_panel.py, 250 np-examples, the answer's own spectra hidden): best same-adduct
cosine < 0.5 for 1 % of answers measured in another library vs 48 % of wrong candidates with reference spectra.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = dict(tau=0.5, down=3, min_ref=1, max_ref=300, top_n=60, min_rel=0.01, tol=0.01, max_rank=25,
              max_seconds=1200, hide_self=0)
# hide_self: panel only (molecule id = the answer's InChIKey14) - hides the answer's own reference spectra


def _prepare(mz, it, precursor, top_n, min_rel):
    mz = np.asarray(mz, dtype=np.float64); it = np.asarray(it, dtype=np.float64)
    keep = (mz < precursor - 1.0) & np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[keep], it[keep]
    if len(it) == 0:
        return mz, it
    keep = it >= min_rel * it.max()
    mz, it = mz[keep], it[keep]
    if len(it) > top_n:
        idx = np.argpartition(-it, top_n)[:top_n]
        mz, it = mz[idx], it[idx]
    o = np.argsort(mz)
    w = np.sqrt(it[o])
    return mz[o], w / np.linalg.norm(w)


def _cos(a, b, tol):
    """Greedy one-to-one peak matching within tol Da; sqrt-intensity unit vectors -> cosine in [0, 1]."""
    (ma, wa), (mb, wb) = a, b
    if len(ma) == 0 or len(mb) == 0:
        return 0.0
    ii, jj = np.nonzero(np.abs(ma[:, None] - mb[None, :]) <= tol)
    if len(ii) == 0:
        return 0.0
    prod = wa[ii] * wb[jj]
    used_a, used_b, total = set(), set(), 0.0
    for k in np.argsort(-prod, kind="stable"):
        i, j = ii[k], jj[k]
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); total += prod[k]
    return float(min(1.0, total))


def contradicted(queries: dict, refs: dict, key: str, p) -> tuple[bool, int, float]:
    """(flag, number of same-adduct reference spectra, best cosine). Stops at the first match >= tau."""
    n, best = 0, 0.0
    for adduct, qs in queries.items():
        for r in refs.get((key, adduct), ()):
            n += 1
            for q in qs:
                s = _cos(q, r, p["tol"])
                if s > best:
                    best = s
                if best >= p["tau"]:
                    return False, n, best
    return n >= p["min_ref"], n, best


def reorder(flags: list, down: int) -> np.ndarray:
    pos = np.arange(len(flags), dtype=np.float64)
    f = np.asarray(flags, dtype=bool)
    key = pos + np.where(f, 1e6 if down < 0 else down + 0.5, 0.0)
    return np.argsort(key, kind="stable")


def load_queries(test_path, p) -> dict:
    cols = ["molecule_id", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
    test = pd.read_parquet(test_path, columns=cols)
    out = {}
    for r in test.itertuples(index=False):
        try:
            s = _prepare(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz), p["top_n"], p["min_rel"])
        except Exception:                      # an unreadable spectrum is left out
            continue
        if len(s[0]):
            out.setdefault(str(r.molecule_id), {}).setdefault(str(r.adduct), []).append(s)
    return out


def load_refs(train_path, keys: set, adducts: set, p, exclude_libs=()) -> dict:
    """Reference spectra of the candidate structures with the queries' adducts, at most max_ref per (key, adduct).
    Two passes over the row groups keep memory small: the key / adduct columns find the rows first, then only those
    rows' spectra are decoded, in small batches. exclude_libs: panel only."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    refs = {}
    if not keys or not adducts:
        return refs
    pf = pq.ParquetFile(str(train_path))
    kset, aset = pa.array(sorted(keys)), pa.array(sorted(adducts))
    lset = pa.array(sorted(exclude_libs)) if exclude_libs else None
    count = {}
    for g in range(pf.num_row_groups):
        meta = pf.read_row_group(g, columns=["inchikey14", "adduct"] + (["ingest_lib"] if lset is not None else []))
        mask = pc.and_(pc.is_in(meta["inchikey14"], value_set=kset), pc.is_in(meta["adduct"], value_set=aset))
        if lset is not None:
            mask = pc.and_(mask, pc.invert(pc.is_in(meta["ingest_lib"], value_set=lset)))
        idx = np.flatnonzero(np.asarray(mask.to_numpy(zero_copy_only=False), dtype=bool))
        if len(idx) == 0:
            continue
        ks = meta["inchikey14"].take(pa.array(idx)).to_pylist(); ad = meta["adduct"].take(pa.array(idx)).to_pylist()
        want = {}
        for i, k, a in zip(idx, ks, ad):           # the cap is applied in file order, before any spectrum is read
            if count.get((k, a), 0) < p["max_ref"]:
                count[(k, a)] = count.get((k, a), 0) + 1
                want[int(i)] = (k, a)
        if not want:
            continue
        off = 0
        for batch in pf.iter_batches(batch_size=8192, row_groups=[g], columns=["precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]):
            n = batch.num_rows
            rows = [i for i in range(off, off + n) if i in want]
            if rows:
                sel = batch.take(pa.array([i - off for i in rows])).to_pydict()
                for i, pm, mz, it in zip(rows, sel["precursor_mz"], sel["ms2_mzs"], sel["ms2_normalized_intensities"]):
                    if mz is None or it is None or pm is None:
                        continue
                    try:
                        s_ = _prepare(mz, it, float(pm), p["top_n"], p["min_rel"])
                    except Exception:
                        continue
                    if len(s_[0]):
                        refs.setdefault(want[i], []).append(s_)
            off += n
    return refs


def apply(test_path, submission_path, train_path, output_path, report_path=None, p=None) -> dict:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    p = {**PARAMS, **(p or {})}
    sub = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
    lists = {m: s.split(";") for m, s in zip(sub.molecule_id, sub.smiles)}
    report = dict(molecules=len(sub), with_queries=0, candidates=0, with_reference=0, flagged=0, molecules_changed=0,
                  rank1_changed=0, errors=0, error_samples=[], stopped_early=False)
    queries = load_queries(test_path, p)
    keys = {}

    def ik14(s):
        if s not in keys:
            try:
                m = Chem.MolFromSmiles(s)
                keys[s] = Chem.MolToInchiKey(m)[:14] if m is not None else ""
            except Exception:
                keys[s] = ""
        return keys[s]

    need = {ik14(s) for m, cand in lists.items() if m in queries for s in cand[: p["max_rank"]]} - {""}
    adducts = {a for q in queries.values() for a in q}
    refs = load_refs(train_path, need, adducts, p)
    report["load_seconds"] = round(time.time() - t0, 1)
    new = {}
    for m, cand in lists.items():
        if time.time() - t0 > p["max_seconds"]:
            report["stopped_early"] = True
            break
        q = queries.get(m)
        if not q or len(cand) < 2:
            continue
        try:
            report["with_queries"] += 1
            flags = []
            for i, s in enumerate(cand):
                f = False
                k = ik14(s) if i < p["max_rank"] else ""
                if k and not (p["hide_self"] and k == m):
                    report["candidates"] += 1
                    f, n, _ = contradicted(q, refs, k, p)
                    report["with_reference"] += int(n >= p["min_ref"])
                flags.append(f)
            if not any(flags):
                continue
            order = reorder(flags, int(p["down"]))
            report["flagged"] += int(sum(flags))
            if (order != np.arange(len(cand))).any():
                new[m] = [cand[i] for i in order]
                report["molecules_changed"] += 1
                report["rank1_changed"] += int(order[0] != 0)
        except Exception as e:             # one odd molecule keeps its order; the others are still checked
            report["errors"] += 1
            if len(report["error_samples"]) < 3:
                report["error_samples"].append(f"{m}: {type(e).__name__}: {str(e)[:120]}")
    out = sub.copy()
    out["smiles"] = [";".join(new.get(m, lists[m])) for m in out.molecule_id]
    assert out.molecule_id.tolist() == sub.molecule_id.tolist()
    assert all(sorted(a.split(";")) == sorted(b.split(";")) for a, b in zip(out.smiles, sub.smiles))
    out.to_csv(output_path, index=False)
    report["seconds"] = round(time.time() - t0, 1)
    report["error_samples"] = report["error_samples"] or None
    if report_path:
        Path(report_path).write_text(json.dumps(dict(report, params=p), indent=2))
    return report
