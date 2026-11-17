"""DreaMS analogue support recomputed locally from the raw neighbour dump (tools/build_dreams_support_dump.py).

Per panel spectrum the dump holds its K + 40 nearest MassSpecGym spectra (index, cosine, SMILES, InChIKey first block).
Support of a candidate = aggregation over the k nearest kept neighbours of cosine^p x Tanimoto (Morgan radius 2,
2,048 bits), averaged over the molecule's spectra, as casmi/dreams_overlay.py computes it. Variants:
  view    hidden: the answer's own structure removed from the neighbours (class 2 / 3: no public spectra);
          visible: nothing removed (class 1)
  noself  1: neighbours with the candidate's own structure do not count for it (pure analogue support; a class-2
          answer has no public spectra, so a candidate matching its own library spectrum is never the class-2 answer)
  agg     max (overlay) | top3 (mean of the 3 best products)
Lists are re-ranked by 1/(3 + rank) / sd + lambda x z(support) (z within the list), with an optional library-first
lead guard (--keep=DUMP.pkl: hits >= lib_first in the real-pipeline dump stay on top while they head the list).

usage: dreams_local.py NEIGHBOURS.json.gz LISTS.csv [--view=hidden] [--p=4] [--k=50] [--noself=0] [--agg=max]
       [--src=all|msg|train] [--keep=DUMP.pkl] [--lib_first=0.95] [--lams=0,0.5,0.75,1,1.25,1.5] [--grid=1|src]
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casmi.metric import ik14 as metric_key  # noqa: E402

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_fp, _ik = {}, {}



_KEYS = Path(__file__).resolve().parents[1] / "artifacts" / "dreams" / "metric_keys.pkl"


def _key_cache(smiles, key_fn):
    """metric keys (RDKit tautomer canonicalisation is slow), cached on disk across runs"""
    import pickle
    cache = pickle.load(open(_KEYS, "rb")) if _KEYS.is_file() else {}
    new = {s_: (key_fn(s_) or "") for s_ in set(smiles) if s_ not in cache}
    if new:
        cache.update(new); _KEYS.parent.mkdir(parents=True, exist_ok=True); pickle.dump(cache, open(_KEYS, "wb"))
    return cache

def fp(s_):
    if s_ not in _fp:
        m_ = Chem.MolFromSmiles(str(s_)); _fp[s_] = GEN.GetFingerprint(m_) if m_ is not None else None
    return _fp[s_]


def std_ik(s_):
    if s_ not in _ik:
        m_ = Chem.MolFromSmiles(str(s_)); _ik[s_] = Chem.MolToInchiKey(m_)[:14] if m_ is not None else ""
    return _ik[s_]


MODES = None                                         # --posonly: panel spectra ion modes (artifacts/dreams/panel_modes.json)


def supports(nbd, lists, view="hidden", p=4.0, k=50, noself=False, agg="max", src="all", self_tau=9.0, selfok=None):
    """{molecule: np.array of support per candidate of its list}"""
    by_m = {}
    for i, rec in enumerate(nbd):
        by_m.setdefault(rec["m"], []).append((i, rec))
    if MODES is not None:                           # positive-mode spectra only, when the molecule has any
        by_m = {m: ([x for x in v if MODES[x[0]] == "pos"] or v) for m, v in by_m.items()}
    by_m = {m: [rec for _, rec in v] for m, v in by_m.items()}
    out = {}
    for m, cand in lists.items():
        recs = by_m.get(m)
        if not recs:
            continue
        cfp = [fp(c) for c in cand]; cik = [std_ik(c) for c in cand]
        sup = np.zeros(len(cand))
        for rec in recs:
            nb = [x for x in rec["nb"] if not (view == "hidden" and x[3] == rec["ik14"])
                  and (src == "all" or len(x) < 5 or x[4] == (0 if src == "msg" else 1))][:k]
            nb = [(fp(x[2]), x[1], x[3]) for x in nb]
            nb = [x for x in nb if x[0] is not None]
            if not nb:
                continue
            fs = [x[0] for x in nb]; w = np.array([max(x[1], 0.0) ** p for x in nb]); niks = [x[2] for x in nb]
            for t, f in enumerate(cfp):
                if f is None:
                    continue
                prod = w * np.array(DataStructs.BulkTanimotoSimilarity(f, fs))
                if noself and cik[t] not in (selfok or {}).get(m, ()):   # library-hit candidates keep their own spectra
                    own = np.array([ik == cik[t] for ik in niks])
                    prod = np.where(own & (np.array([x[1] for x in nb]) < self_tau), 0.0, prod)
                sup[t] += float(prod.max()) if agg == "max" else float(np.sort(prod)[-3:].mean())
        out[m] = sup / len(recs)
    return out


EPS = 1e-9                                           # --eps: damped z = (s - mean) / (sd + eps)


def rerank(cand, sup, lam, n_head=0):
    base = 1.0 / (3.0 + np.arange(1, len(cand) + 1))
    z = (sup - sup.mean()) / (sup.std() + EPS)
    new = [cand[i] for i in np.argsort(-(base / base.std() + lam * z), kind="stable")]
    if n_head:
        head = cand[:n_head]; new = head + [c for c in new if c not in set(head)]
    return new


def evaluate(lists, sups, heads, lams, keys):
    def rr(m, lst):
        ks = [keys[s_] for s_ in lst[:25]]
        return 1.0 / (ks.index(m) + 1) if m in ks else 0.0
    ms = [m for m in lists if m in sups and len(lists[m]) > 1]
    res = []
    for lam in lams:
        res.append(np.mean([rr(m, rerank(lists[m], sups[m], lam, heads.get(m, 0))) for m in ms]))
    return ms, res


def main() -> None:
    nbd = json.loads(gzip.decompress(Path(sys.argv[1]).read_bytes()))
    sub = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)
    args = dict(a[2:].split("=", 1) for a in sys.argv[3:] if a.startswith("--") and "=" in a)
    global EPS, MODES
    EPS = float(args.get("eps", 1e-9))
    if args.get("posonly") == "1":
        MODES = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "dreams" / "panel_modes.json").read_text())
    lists = {m: smi.split(";") for m, smi in zip(sub.molecule_id, sub.smiles)}
    keys = _key_cache([s_ for cand in lists.values() for s_ in cand], metric_key)
    lams = [float(x) for x in args.get("lams", "0,0.5,0.75,1,1.25,1.5").split(",")]
    heads = {}
    if "keep" in args:
        hits = pickle.load(open(args["keep"], "rb"))["hits"]
        tau = float(args.get("lib_first", 0.95))
        for m, cand in lists.items():
            ks = {s_ for c, s_ in hits.get(m, {}).values() if c >= tau}
            if ks:
                heads[m] = next((i for i, c in enumerate(cand) if c not in ks), len(cand))
    view = args.get("view", "hidden")
    grid = [dict(p=float(args.get("p", 4)), k=int(args.get("k", 50)), noself=args.get("noself") == "1", agg=args.get("agg", "max"),
                 src=args.get("src", "all"), self_tau=float(args.get("self_tau", 9.0)))]
    if args.get("grid") == "tau":                      # noself with own spectra kept above a cosine threshold
        grid = [dict(p=4.0, k=50, noself=True, agg="max", src=args.get("src", "all"), self_tau=tau)
                for tau in (0.7, 0.8, 0.85, 0.9, 0.95, 9.0)]
    if args.get("grid") == "1":
        grid = [dict(p=p, k=k, noself=ns, agg=ag, src=args.get("src", "all")) for ns in (False, True) for ag in ("max", "top3")
                for p in (2.0, 4.0, 8.0) for k in (10, 25, 50, 90)]
    if args.get("grid") == "src":                      # MassSpecGym only vs train-library references only vs both
        grid = [dict(p=4.0, k=50, noself=False, agg="max", src=s_) for s_ in ("msg", "train", "all")]
    print(f"{Path(sys.argv[2]).name} | view {view} | lists {len(lists)} | protected leads {len(heads)} | lambdas {lams}")
    if "candswitch" in args:                           # per candidate: own spectra kept only for the query's library hits
        hits = pickle.load(open(args["keep"], "rb"))["hits"]
        for tau in [float(x) for x in args["candswitch"].split(",")]:
            ok = {m: {std_ik(s_) for c, s_ in hits.get(m, {}).values() if c >= tau} for m in lists}
            sups = supports(nbd, lists, view=view, noself=True, selfok=ok, src=args.get("src", "all"))
            ms, res = evaluate(lists, sups, heads, lams, keys)
            print(f"  candidate switch {tau:<5g} (exempt candidates {sum(len(v) for v in ok.values())}) | n {len(ms)} | "
                  + " | ".join(f"{lam:g}: {r:.4f}" for lam, r in zip(lams, res)), flush=True)
        return
    if "libswitch" in args:                            # per query: own spectra excluded only without a library match
        hits = pickle.load(open(args["keep"], "rb"))["hits"]
        best = {m: max((c for c, _ in hits.get(m, {}).values()), default=0.0) for m in lists}
        s0 = supports(nbd, lists, view=view, noself=False)
        s1 = supports(nbd, lists, view=view, noself=True)
        for tau in [float(x) for x in args["libswitch"].split(",")]:
            sups = {m: (s0[m] if best.get(m, 0.0) >= tau else s1[m]) for m in s0}
            ms, res = evaluate(lists, sups, heads, lams, keys)
            n_ns = sum(best.get(m, 0.0) < tau for m in s0)
            print(f"  library switch {tau:<5g} (noself for {n_ns} lists) | n {len(ms)} | "
                  + " | ".join(f"{lam:g}: {r:.4f}" for lam, r in zip(lams, res)), flush=True)
        return
    for g in grid:
        sups = supports(nbd, lists, view=view, **g)
        ms, res = evaluate(lists, sups, heads, lams, keys)
        print(f"  src {g['src']:5s} p {g['p']:g} k {g['k']:3d} noself {int(g['noself'])} tau {g.get('self_tau', 9.0):<4g} agg {g['agg']:4s} | n {len(ms)} | "
              + " | ".join(f"{lam:g}: {r:.4f}" for lam, r in zip(lams, res)), flush=True)


if __name__ == "__main__":
    main()
