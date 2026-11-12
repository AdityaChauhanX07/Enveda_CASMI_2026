"""Networking overlay for a finished submission (self-contained: numpy, pandas, RDKit).

For every test molecule with a spectral neighbour (modified cosine over spectra with the same adduct), each of
its candidates gets the expected Tanimoto to the neighbour's answer list, where list position r carries belief
1 / (r + k). The list is re-sorted by -log(1 + r) + lam * term. Molecules without a neighbour keep their order.
Panel check (experiments/network_overlay_panel.py): lam 4, k 1 gave +0.006 to +0.009 MRR, about 2 molecules up
for every 1 down.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = dict(cos_min=0.6, min_matches=8, k=1.0, lam=4.0, tol=0.01, top_n=60, min_rel=0.01, max_list=25)


def _prepare(mz, it, precursor, top_n, min_rel):
    mz = np.asarray(mz, dtype=np.float64); it = np.asarray(it, dtype=np.float64)
    keep = (mz < precursor - 1.0) & np.isfinite(it) & (it > 0)
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
    prod = wa[ii] * wb[jj]
    ua, ub, tot, n = set(), set(), 0.0, 0
    for q in np.argsort(-prod, kind="stable"):
        i, j = ii[q], jj[q]
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); tot += prod[q]; n += 1
    return float(tot), n


def graph(test: pd.DataFrame, p=PARAMS) -> dict:
    """molecule -> [(neighbour, cosine)] for pairs above the thresholds."""
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
                if n < p["min_matches"] or s < p["cos_min"]:
                    continue
                key = (ma, mb) if ma < mb else (mb, ma)
                best[key] = max(best.get(key, 0.0), s)
    nbr = {}
    for (a, b), s in best.items():
        nbr.setdefault(a, []).append((b, s)); nbr.setdefault(b, []).append((a, s))
    return nbr


def apply(test_path, submission_path, output_path, report_path=None, p=PARAMS) -> dict:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cols = ["molecule_id", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
    test = pd.read_parquet(test_path, columns=cols)
    sub = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
    lists = {m: s.split(";") for m, s in zip(sub.molecule_id, sub.smiles)}
    nbr = graph(test, p)
    fps = {}

    def fp(s):
        if s not in fps:
            m = Chem.MolFromSmiles(s)
            fps[s] = gen.GetFingerprint(m) if m is not None else None
        return fps[s]

    changed, top1, linked = 0, 0, 0
    new = {}
    for m, cand in lists.items():
        if m not in nbr or len(cand) < 2:
            continue
        own = [fp(s) for s in cand]
        if any(f is None for f in own):
            continue
        acc, wsum = np.zeros(len(cand)), 0.0
        for a, w in nbr[m]:
            other = [fp(s) for s in lists.get(a, [])[: p["max_list"]]]
            other = [(r, f) for r, f in enumerate(other) if f is not None]
            if not other:
                continue
            pr = np.array([1.0 / (r + p["k"]) for r, _ in other]); pr /= pr.sum()
            sims = np.zeros(len(cand))
            for pj, (_, f) in zip(pr, other):
                sims += pj * np.asarray(DataStructs.BulkTanimotoSimilarity(f, own))
            acc += w * sims; wsum += w
        if not wsum:
            continue
        linked += 1
        score = -np.log1p(np.arange(len(cand))) + p["lam"] * acc / wsum
        order = np.argsort(-score, kind="stable")
        if (order != np.arange(len(cand))).any():
            changed += 1
            top1 += int(order[0] != 0)
            new[m] = [cand[i] for i in order]
    out = sub.copy()
    out["smiles"] = [";".join(new.get(m, lists[m])) for m in out.molecule_id]
    assert out.molecule_id.tolist() == sub.molecule_id.tolist()
    assert all(sorted(a.split(";")) == sorted(b.split(";")) for a, b in zip(out.smiles, sub.smiles))
    out.to_csv(output_path, index=False)
    report = dict(params=p, molecules=len(sub), linked=linked, edges=sum(len(v) for v in nbr.values()) // 2,
                  reordered=changed, rank1_changed=top1)
    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2))
    return report
