"""Why are panel answers never generated? For each, find its true one-edit relatives in the searched windows
(the 23-window set) and report the stage that loses the best one: not in the 500 best-documented rows of any
window / outside the doc pre-filter / not among the fully scored rows (n_full best by the ECFP4 pass + n_docrank
best-documented) / not among the top k picked. Then, for answers that are generated, where they lose at insertion.

usage: generation_funnel.py [--state=t3] [--n_doc=150] [--n_full=40] [--n_docrank=0] [--k=3]
(defaults reproduce the a5 numbers on the task-3 dumps; a7: --state=t7 --n_full=20 --n_docrank=20 --k=4)"""
from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.edit_overlay import REL_SKIP, Chemistry, Tier  # noqa: E402
from new_edits_coverage import NEW_STRIP  # noqa: E402

def args() -> dict:
    a = dict(state="t3", n_doc=150, n_full=40, n_docrank=0, k=3)
    for x in sys.argv[1:]:
        k, v = x.lstrip("-").split("=", 1)
        a[k] = v if k == "state" else int(v)
    return a


def insertion_funnel(c3: dict) -> Counter:
    """For generated answers: not among the top 3 products / top-3 product below every top-3 list member / inserted."""
    import insertion_replay as ir
    from casmi.edit_overlay import merge_sources
    out = Counter()
    for m, st in c3.items():
        merged = sorted(merge_sources([(g, None, k, meta) for g, k, meta in st["scored"]], 0.0), key=lambda x: -x[0])
        keys = [k for _, _, k, _ in merged]
        if m not in keys:
            continue
        i = keys.index(m)
        if i >= 3:
            out["generated, not among the top 3 products"] += 1
        elif not (st["top"] and merged[i][0] <= min(st["top"])) or len(st["top"]) < 3:
            r = ir.rr(ir.insert(st, {}), m)
            out[f"inserted at rank {int(round(1 / r))}" if r else "top-3 product, not inserted"] += 1
        else:
            out["top-3 product below every top-3 list member"] += 1
    return out


if __name__ == "__main__":
    A = args()
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    Chem = chem.Chem
    tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    ik = lambda smi: (Chem.MolToInchiKey(Chem.MolFromSmiles(smi))[:14].encode() if Chem.MolFromSmiles(smi) is not None else b"")
    c3 = pickle.load(open(ROOT / "artifacts" / f"state_{A['state']}_c3.pkl", "rb"))
    generated = {m for m, st in c3.items() if any(k == m for _, k, _ in st["scored"])}
    t = pq.read_table(ROOT.parents[0] / "casmi_data" / "raw" / "train.parquet", columns=["ingest_lib", "normalized_smiles"],
                      filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas().drop_duplicates("normalized_smiles")
    t["key"] = [chem.key(s) for s in t.normalized_smiles]
    smiles = dict(zip(t.key, t.normalized_smiles))
    missing = sorted(set(c3) - generated)
    # relatives of answer A through each searched window: lighter (A minus group: strip on A) / heavier (A plus group)
    new_add = {"sulfate": "[OX2H1:1]>>[O:1]S(=O)(=O)O", "coumaroyl": "[OX2H1:1]>>[O:1]C(=O)C=Cc1ccc(O)cc1",
               "caffeoyl": "[OX2H1:1]>>[O:1]C(=O)C=Cc1ccc(O)c(O)c1", "galloyl": "[OX2H1:1]>>[O:1]C(=O)c1cc(O)c(O)c(O)c1"}
    windows = [(e, -1, chem.strip[e][1]) for e in chem.fwd if (e, -1) not in REL_SKIP] + \
              [(e, +1, chem.fwd[e][1]) for e in chem.strip if (e, +1) not in REL_SKIP] + \
              [(e, +1, AllChem.ReactionFromSmarts(new_add[e])) for e in NEW_STRIP if e in new_add]
    stages = Counter()
    for m in missing:
        s = smiles.get(m)
        if s is None:
            stages["answer SMILES not found"] += 1; continue
        cache_old = pickle.load(open(ROOT / "artifacts" / "relatives_cache" / f"{m}.pkl", "rb")) if (ROOT / "artifacts" / "relatives_cache" / f"{m}.pkl").is_file() else {}
        cache_new = pickle.load(open(ROOT / "artifacts" / "relatives_cache_new" / f"{m}.pkl", "rb")) if (ROOT / "artifacts" / "relatives_cache_new" / f"{m}.pkl").is_file() else {}
        best = None                                      # furthest stage reached by any true relative
        order = ["not in the 500 best-documented rows", f"outside the {A['n_doc']}-row pre-filter", "not fully scored",
                 f"fully scored, not in the top {A['k']} picked", "picked"]
        for e, sign, rx in windows:
            w = (cache_new if e in NEW_STRIP else cache_old).get((e, sign))
            if w is None:
                continue
            rel_keys = {ik(p) for k, p in chem.products(s, rx, cap=50).items() if k != m} - {b""}
            if not rel_keys:
                continue
            wkeys = np.asarray(tier.ik[np.asarray(w["rows"], np.int64)])          # standard InChIKey14 of the cached rows
            hits = [i for i, k in enumerate(wkeys) if k in rel_keys]
            if not hits:
                stage = 0
            else:
                i = min(hits)
                n = min(A["n_doc"], len(wkeys))
                if i >= n:
                    stage = 1
                else:
                    part = w["part"][:n]
                    sel = list(np.argsort(-part, kind="stable")[:A["n_full"]])
                    sel += [j for j in range(min(A["n_docrank"], n)) if j not in sel]
                    if not any(h in sel for h in hits if h < n):
                        stage = 2
                    else:
                        cand = [j for j in sel if np.isfinite(w["full"][j])]
                        fv = np.array([w["full"][j] for j in cand]); fz = (fv - fv.mean()) / (fv.std() + 1e-9)
                        picked = [cand[j] for j in np.argsort(-(fz + 0.3 * w["doc"][cand]), kind="stable")[:A["k"]]]
                        stage = 4 if any(h in picked for h in hits) else 3
            best = stage if best is None else max(best, stage)
        stages[order[best] if best is not None else "no relative through any searched edit"] += 1
    print(f"setting {A}: answers generated {len(generated)} of {len(c3)}, never generated {len(missing)}")
    for k, v in stages.most_common():
        print(f"  {k:46s} {v:4d}")
    print("generated answers at insertion:")
    for k, v in sorted(insertion_funnel(c3).items()):
        print(f"  {k:46s} {v:4d}")
