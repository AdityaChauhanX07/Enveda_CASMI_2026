"""spec2mol on C1's lists when answers are less documented than on the panel (the a7 stress test).

From the C1 dumps (artifacts/c1_dump_{c2,c1}_v1.pkl), each molecule's PubChem rows are split back into a fingerprint
part (score - 0.25 x doc) and documentation (as experiments/c1_docweight.py). The answer's documentation is capped at d,
C1's lists are rebuilt (library-hit bonus 4 x (cos - 0.8)), and each top 25 is re-ranked with spec2mol at rank level:
1/(3 + C1 rank) + w/(3 + spec2mol rank). The spec2mol scores (artifacts/spec2mol_scores.json) come from the Kaggle job
casmi26-spec2mol-panel v2, which scores every dumped candidate. spec2mol held out the panel's molecules, so the check is
not leaky. Class 2: the answer's spectra hidden from the library search. Class-1 view: only the queries' own library left
out.

usage: spec2mol_lowdoc.py [SCORES_JSON]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT))
from casmi.edit_overlay import Chemistry, Tier  # noqa: E402

_CHEM = None


def _keys(smiles: list) -> list:
    return [_CHEM.key(s) for s in smiles]


def split_rows(art: Path) -> dict:
    """{"rows": {m: [(key, z, doc)]}, "hits": {tag: {m: [(key, cos)]}}}, cached in artifacts/c1_rows_v1.pkl."""
    cache = art / "c1_rows_v1.pkl"
    if cache.is_file():
        return pickle.load(open(cache, "rb"))
    global _CHEM
    tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    _CHEM = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    dumps = {tag: pickle.load(open(art / f"c1_dump_{tag}_v1.pkl", "rb")) for tag in ("c2", "c1")}
    d2 = dumps["c2"]
    ids = d2["ids"]
    pub = {m: [(tier.smiles(r), total, float(tier.lsid[r]) + float(tier.lpmid[r])) for r, total in d2["pub"].get(m, [])] for m in ids}
    hit = {tag: {m: [(s_, c) for _, (c, s_) in d["hits"].get(m, {}).items()] for m in ids} for tag, d in dumps.items()}
    jobs = [[s for s, _, _ in pub[m]] for m in ids] + [[s for s, _ in hit[tag][m]] for tag in ("c2", "c1") for m in ids]
    with mp.get_context("fork").Pool(4) as pool:
        keys = pool.map(_keys, jobs, chunksize=4)
    n = len(ids)
    out = {"ids": ids, "rows": {}, "hits": {"c2": {}, "c1": {}}}
    for i, m in enumerate(ids):
        out["rows"][m] = [(k, total - 0.25 * doc, doc) for k, (_, total, doc) in zip(keys[i], pub[m])]
        for j, tag in enumerate(("c2", "c1")):
            out["hits"][tag][m] = [(k, c) for k, (_, c) in zip(keys[n * (j + 1) + i], hit[tag][m])]
    pickle.dump(out, open(cache, "wb"))
    return out


def c1_scores(m: str, rows: list, hits: list, cap, w_doc: float = 0.25) -> dict:
    """C1's score per candidate key over the whole dumped shortlist (the answer's documentation capped at cap)."""
    sc = {}
    for k, z, doc in rows:
        if not k or k in sc:
            continue
        sc[k] = z + w_doc * (min(doc, cap) if (cap is not None and k == m) else doc)
    base = min(sc.values()) if sc else 0.0
    for k, c in hits:
        if k and c >= 0.8:                     # C1 records hits at cos >= 0.8 (the dumps keep >= 0.5)
            sc[k] = sc.get(k, base) + 4.0 * (c - 0.8)
    return sc


def c1_list(m: str, rows: list, hits: list, cap, w_doc: float = 0.25, top_k: int = 25) -> list:
    return [k for k, _ in sorted(c1_scores(m, rows, hits, cap, w_doc).items(), key=lambda x: -x[1])][:top_k]


def fuse(keys: list, neural: dict | None, w: float, offset: float = 3.0) -> list:
    """Rank-level fusion: 1/(offset + list rank) + w/(offset + neural rank), ranks within `keys`."""
    if not neural or w <= 0 or not keys:
        return keys
    sc = np.array([neural.get(k, -1e9) for k in keys])
    nr = np.empty(len(keys)); nr[np.argsort(-sc, kind="stable")] = np.arange(1, len(keys) + 1)
    fused = 1.0 / (offset + np.arange(1, len(keys) + 1)) + w / (offset + nr)
    return [keys[i] for i in np.argsort(-fused, kind="stable")]


def pooled_scale(shortlists: dict, scores: dict, min_n: int = 5) -> float:
    """The spread of spec2mol scores within a shortlist, pooled over molecules: the median per-molecule standard
    deviation over shortlists with at least min_n scored candidates."""
    sds = []
    for m, keys in shortlists.items():
        v = np.array([scores.get(m, {}).get(k, np.nan) for k in keys], dtype=float)
        v = v[np.isfinite(v) & (v > -1e8)]
        if len(v) >= min_n:
            sds.append(v.std())
    return float(np.median(sds)) if sds else 1.0


def fuse_scores(sc: dict, neural: dict | None, gamma: float, scale: float | None = None) -> list:
    """Score-level fusion over the whole shortlist: C1 score + gamma x spec2mol score / scale (scale None: each
    shortlist's own standard deviation, which inflates small differences on short lists)."""
    keys = sorted(sc, key=lambda k: -sc[k])
    if not neural or gamma <= 0:
        return keys
    v = np.array([neural.get(k, np.nan) for k in keys], dtype=float)
    ok = np.isfinite(v) & (v > -1e8)
    if ok.sum() < 2:
        return keys
    mu, sd = v[ok].mean(), (v[ok].std() if scale is None else scale) + 1e-9
    zs = np.where(ok, (v - mu) / sd, (v[ok].min() - mu) / sd)   # unscored candidates count as the lowest scored
    total = np.array([sc[k] for k in keys]) + gamma * zs
    return [keys[i] for i in np.argsort(-total, kind="stable")]


def rr(keys: list, m: str) -> float:
    return 1.0 / (keys.index(m) + 1) if m in keys[:25] else 0.0


if __name__ == "__main__":
    art = ROOT / "artifacts"
    scores = json.load(open(sys.argv[1] if len(sys.argv) > 1 else art / "spec2mol_scores.json"))
    data = split_rows(art)
    ids = data["ids"]
    missing = [m for m in ids if m not in scores]
    print(f"molecules {len(ids)}, with spec2mol scores {len(ids) - len(missing)}")
    caps = (None, 8, 6, 4)
    modes = [("top-25 RRF", w) for w in (0.0, 0.1, 0.25, 0.5, 1.0)] + [("shortlist RRF", w) for w in (0.25, 0.5, 1.0)] + \
            [("score", g) for g in (0.1, 0.25, 0.5, 1.0, 2.0)] + [("score pooled", g) for g in (0.25, 0.5, 0.75, 1.0, 1.5)]
    for tag, name in (("c2", "class 2 (answer's spectra hidden)"), ("c1", "class-1 view (queries' own library left out)")):
        print(f"\n{name}: MRR@25 | columns: answer's documentation as is / capped")
        print("fusion             | " + " | ".join("as is " if c is None else f"<= {c:<3d}" for c in caps))
        sc = {c: {m: c1_scores(m, data["rows"][m], data["hits"][tag][m], c) for m in ids} for c in caps}
        ranked = {c: {m: sorted(sc[c][m], key=lambda k: -sc[c][m][k]) for m in ids} for c in caps}
        pooled = pooled_scale({m: list(sc[None][m]) for m in ids}, scores)
        print(f"pooled spec2mol scale (median within-shortlist sd): {pooled:.3f}")
        for kind, w in modes:
            cells = []
            for c in caps:
                if kind == "top-25 RRF":
                    lists = {m: fuse(ranked[c][m][:25], scores.get(m), w) for m in ids}
                elif kind == "shortlist RRF":
                    lists = {m: fuse(ranked[c][m], scores.get(m), w) for m in ids}
                elif kind == "score":
                    lists = {m: fuse_scores(sc[c][m], scores.get(m), w) for m in ids}
                else:
                    lists = {m: fuse_scores(sc[c][m], scores.get(m), w, pooled) for m in ids}
                cells.append(np.mean([rr(lists[m][:25], m) for m in ids]))
            print(f"{kind:13s} {w:4.2f} | " + " | ".join(f"{v:.4f}" for v in cells), flush=True)
