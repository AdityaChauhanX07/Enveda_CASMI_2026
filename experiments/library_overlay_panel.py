"""Panel check of casmi/library_overlay.py: candidates whose same-adduct reference spectra all mismatch the query.

Lists = the panel's fork-like order (fold-safe FPNet + 0.4 * documentation, top 25, as in edit_overlay_panel.py).
Class 2: the answer's own reference spectra are hidden (hide_self), so only wrong candidates can be flagged.
Class-1 risk: the answer's spectra from other libraries are visible; reported as the share of answers flagged
(a class-1 list usually has its answer first, so each flagged answer costs ~1 - 1/(1 + down) of a molecule).
A grid over tau / down from one load of the reference spectra, then apply() end to end for the chosen settings.

usage: library_overlay_panel.py [--apply=tau:down,...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi import library_overlay as lo  # noqa: E402
from casmi.data import DATA  # noqa: E402
from casmi.metric import ik14, score  # noqa: E402
from casmi.panel import load_candidates, read_library, tier_smiles, zscore  # noqa: E402
from network_panel import shortlist  # noqa: E402


def panel_lists():
    art = ROOT / "artifacts"; art.mkdir(exist_ok=True)
    t = read_library("enveda-np-examples")
    t["molecule_id"] = [ik14(x) for x in t.normalized_smiles]
    t.to_parquet(art / "np_test.parquet")
    truth = t.drop_duplicates("molecule_id").set_index("molecule_id").normalized_smiles
    c = load_candidates()
    d = shortlist(c, zscore(c, "fold0_full") + 0.4 * c.doc.to_numpy())
    d = d[d.rank0 < 26].reset_index(drop=True)
    d["smiles"] = tier_smiles(d.tier_row.to_numpy())
    lists = {m: g.smiles.tolist()[:25] for m, g in d.groupby("molecule", sort=False)}
    sol = pd.DataFrame({"molecule_id": truth.index, "smiles": truth.values})
    sub = pd.DataFrame({"molecule_id": sol.molecule_id, "smiles": [";".join(lists.get(m, ["CCO"])) for m in sol.molecule_id]})
    return art, sol, sub


def rr(cand_keys: list, m: str) -> float:
    for i, k in enumerate(cand_keys[:25]):
        if k == m:
            return 1.0 / (i + 1)
    return 0.0


if __name__ == "__main__":
    t0 = time.time()
    art, sol, sub = panel_lists()
    sub.to_csv(art / "np_sub_c2.csv", index=False)
    p = dict(lo.PARAMS, tau=2.0)                       # tau 2: never stops early -> best cosine of every candidate
    queries = lo.load_queries(art / "np_test.parquet", p)
    lists = {m: s.split(";") for m, s in zip(sub.molecule_id, sub.smiles)}
    keys = {s: ik14(s) for c in lists.values() for s in c}      # metric keys (the panel id is the answer's key)
    from rdkit import Chem
    std = {s: (Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] if Chem.MolFromSmiles(s) is not None else "") for s in keys}
    need = {std[s] for m, c in lists.items() if m in queries for s in c} - {""}
    refs = lo.load_refs(DATA / "train.parquet", need, {a for q in queries.values() for a in q}, p)
    # class-1 view of the answer: its reference spectra from other libraries only (a class-1 test molecule is a new
    # measurement; its own np-examples spectra are the queries here)
    refs1 = lo.load_refs(DATA / "train.parquet", set(lists), {a for q in queries.values() for a in q}, p,
                         exclude_libs={"enveda-np-examples"})
    print(f"loaded queries for {len(queries)} molecules, references for {len({k for k, _ in refs})} structures in {time.time() - t0:.0f} s")
    info = {}                                          # (molecule, position) -> (n same-adduct refs, best cosine)
    for m, c in lists.items():
        for i, s in enumerate(c):
            if std[s]:
                _, n, best = lo.contradicted(queries.get(m, {}), refs1 if std[s] == m else refs, std[s], p)
                info[(m, i)] = (n, best)
    ans = [lo.contradicted(queries.get(m, {}), refs1, m, p)[1:] for m in lists]
    print("answers with same-adduct spectra in other libraries:", sum(n > 0 for n, _ in ans), "of", len(ans))
    print(f"cosines done in {time.time() - t0:.0f} s")
    mols = list(lists); halves = [set(mols[0::2]), set(mols[1::2])]
    N = len(sol)
    print("tau  down min_ref | class 2: delta (helped)  halves          | answers flagged (class-1 risk) | wrong flagged")
    for tau in (0.4, 0.5, 0.6, 0.7):
        for down in (3, 5, -1):
            for min_ref in (1, 3):
                d2, fl_ans, fl_wrong, helped = {}, 0, 0, 0
                for m, c in lists.items():
                    ks = [keys[s] for s in c]
                    flags, flags1 = [], []
                    for i, s in enumerate(c):
                        n, best = info.get((m, i), (0, 0.0))
                        f = n >= min_ref and best < tau
                        flags1.append(f)                       # class-1 view: the answer may be flagged
                        flags.append(f and ks[i] != m)         # class 2: the answer has no public spectra
                    pass
                    fl_wrong += int(sum(flags))
                    o = lo.reorder(flags, down)
                    d2[m] = rr([ks[j] for j in o], m) - rr(ks, m)
                    helped += d2[m] > 0
                fl_ans = sum(n >= min_ref and best < tau for n, best in ans)
                tot = sum(d2.values()) / N
                hv = [sum(v for k, v in d2.items() if k in h) / (N / 2) for h in halves]
                cost = (1 - 1 / (1 + down)) if down >= 0 else 1.0
                print(f"{tau:.1f}  {down:>3d}  {min_ref:>3d}    | {tot:+.4f} ({helped:2d})  {hv[0]:+.4f} / {hv[1]:+.4f} | "
                      f"{fl_ans:3d} = {fl_ans / N:.3f} -> ~{fl_ans / N * cost:.4f} per class-1 molecule | {fl_wrong}")
    for a in [x for x in sys.argv[1:] if x.startswith("--apply=")]:
        for spec in a.split("=", 1)[1].split(","):
            tau, down = spec.split(":")
            for hide in (1, 0):
                rep = lo.apply(art / "np_test.parquet", art / "np_sub_c2.csv", DATA / "train.parquet", art / "np_sub_lib.csv",
                               None, dict(tau=float(tau), down=int(down), hide_self=hide))
                after = pd.read_csv(art / "np_sub_lib.csv", dtype=str, keep_default_na=False)
                print(f"apply tau {tau} down {down} hide_self {hide}: MRR {score(sol, sub):.4f} -> {score(sol, after):.4f} | "
                      f"{ {k: rep[k] for k in ('with_queries', 'candidates', 'with_reference', 'flagged', 'molecules_changed', 'rank1_changed', 'errors', 'load_seconds', 'seconds')} }")
