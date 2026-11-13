"""Why are 102 panel answers never generated? For each, find its true one-edit relatives in the searched windows
(the 23-window set) and report the stage that loses the best one: not in the 500 best-documented rows of any
window / outside the 150-row pre-filter / not among the 40 best by the ECFP4 pass / not among the top 3 picked."""
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

if __name__ == "__main__":
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    Chem = chem.Chem
    tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    ik = lambda smi: (Chem.MolToInchiKey(Chem.MolFromSmiles(smi))[:14].encode() if Chem.MolFromSmiles(smi) is not None else b"")
    c3 = pickle.load(open(ROOT / "artifacts" / "state_t3_c3.pkl", "rb"))
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
        order = ["not in the 500 best-documented rows", "outside the 150-row pre-filter", "not in the 40 best by ECFP4", "not in the top 3 picked", "picked"]
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
                if i >= 150:
                    stage = 1
                else:
                    part = w["part"][:150]
                    top40 = list(np.argsort(-part, kind="stable")[:40])
                    if not any(h in top40 for h in hits if h < 150):
                        stage = 2
                    else:
                        cand = [j for j in top40 if np.isfinite(w["full"][j])]
                        fv = np.array([w["full"][j] for j in cand]); fz = (fv - fv.mean()) / (fv.std() + 1e-9)
                        picked = [cand[j] for j in np.argsort(-(fz + 0.3 * w["doc"][cand]), kind="stable")[:3]]
                        stage = 4 if any(h in picked for h in hits) else 3
            best = stage if best is None else max(best, stage)
        stages[order[best] if best is not None else "no relative through any searched edit"] += 1
    print(f"answers never generated: {len(missing)}")
    for k, v in stages.most_common():
        print(f"  {k:46s} {v:4d}")
