"""Offline re-ranking rules for the DreaMS analogue support (tools/build_dreams_support_dump.py output).

For each list file (real-pipeline panel lists, molecule_id = the answer's metric key) and each weight lambda, every
list is re-ordered exactly as casmi/dreams_overlay.py does: 1/(3 + rank) / sd + lambda x z(support), z within the list.
`--keep=DUMP.pkl` protects the library-first leads the pipeline put on top (hits >= `--lib_first` in the dump's library
hits, while they still head the list), as the notebooks do. The support view is `hidden` (answer's own structure
removed from the neighbours) unless `--view=visible`.

usage: dreams_rules.py SUPPORT.json.gz LISTS.csv [--keep=DUMP.pkl] [--lib_first=0.95] [--view=hidden|visible]
       [--lams=0,0.25,0.5,0.75,1,1.25,1.5]
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casmi.metric import ik14  # noqa: E402


def rerank(cand: list[str], sup: np.ndarray, lam: float, n_head: int = 0) -> list[str]:
    base = 1.0 / (3.0 + np.arange(1, len(cand) + 1))
    z = (sup - sup.mean()) / (sup.std() + 1e-9)
    new = [cand[i] for i in np.argsort(-(base / base.std() + lam * z), kind="stable")]
    if n_head:
        head = cand[:n_head]
        new = head + [c for c in new if c not in set(head)]
    return new


def main() -> None:
    support = json.loads(gzip.decompress(Path(sys.argv[1]).read_bytes()))
    sub = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)
    args = dict(a[2:].split("=", 1) for a in sys.argv[3:] if a.startswith("--") and "=" in a)
    view = 1 if args.get("view") == "visible" else 0
    lams = [float(x) for x in args.get("lams", "0,0.25,0.5,0.75,1,1.25,1.5").split(",")]
    leads = {}
    if "keep" in args:                                 # library-first leads, as c1_pipeline writes them
        hits = pickle.load(open(args["keep"], "rb"))["hits"]
        tau = float(args.get("lib_first", 0.95))
        for m, h in hits.items():
            leads[m] = {s_ for c, s_ in h.values() if c >= tau}
    keys: dict[str, str] = {}
    key = lambda s_: keys[s_] if s_ in keys else keys.setdefault(s_, ik14(s_) or "")
    rows = []
    for m, smi in zip(sub.molecule_id, sub.smiles):
        cand = smi.split(";")
        sup_m = support.get(m)
        if sup_m is None or len(cand) < 2:
            continue
        sup = np.array([sup_m.get(c, [0.0, 0.0])[view] for c in cand])
        ks = set(leads.get(m, ()))
        n_head = next((i for i, c in enumerate(cand) if c not in ks), len(cand)) if ks else 0
        rows.append((m, cand, sup, n_head))
    print(f"{Path(sys.argv[2]).name}: {len(rows)} lists with support ({view and 'visible' or 'hidden'} view); "
          f"{sum(r[3] > 0 for r in rows)} with protected leads")

    def rr(m, lst):
        ks = [key(s_) for s_ in lst[:25]]
        return 1.0 / (ks.index(m) + 1) if m in ks else 0.0
    for guard in ((False, True) if leads else (False,)):
        res = []
        for lam in lams:
            v = [rr(m, rerank(cand, sup, lam, n_head if guard else 0)) for m, cand, sup, n_head in rows]
            res.append(f"{lam:g}: {np.mean(v):.4f}")
        print(("  with lead guard | " if guard else "  no guard       | ") + " | ".join(res))
    base_rr = np.array([rr(m, cand) for m, cand, _, _ in rows])
    for lam in lams[1:]:
        new_rr = np.array([rr(m, rerank(cand, sup, lam, n_head)) for m, cand, sup, n_head in rows])
        print(f"  lambda {lam:g}: helped {int((new_rr > base_rr).sum())}, hurt {int((new_rr < base_rr).sum())}, "
              f"rank-1 lost {int(((base_rr == 1) & (new_rr < 1)).sum())}, rank-1 gained {int(((base_rr < 1) & (new_rr == 1)).sum())}")


if __name__ == "__main__":
    main()
