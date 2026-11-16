"""Where does C1 (documentation-free settings) lose the panel's class-2 answers?

From a dump of the real pipeline (experiments/c1_window_lowdoc.py --dump=PATH: PubChem / COCONUT shortlists, library
hits, and PATH.sub.csv with the final lists), each molecule's answer is placed in one of:
  rank 1 | rank 2-25 | in the shortlist but below 25 | in the PubChem window but cut before the shortlist |
  not in the window (index / mass coverage)
Matching uses the standard InChIKey first block. For answers ranked 2-25 it reports whether the answer and the
candidates above it are natural-product members and whether they share the answer's formula (isomers).

usage: c1_failures.py DUMP NATURAL_PRODUCTS_PARQUET COCONUT_CSV
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT))
from casmi import edit_overlay as eo  # noqa: E402
from casmi.c1_pipeline import CocoExtra  # noqa: E402
from casmi.metric import ik14  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def std14(s_: str) -> str:
    m = Chem.MolFromSmiles(s_)
    return Chem.MolToInchiKey(m)[:14] if m is not None else ""


def formula(s_: str) -> str:
    m = Chem.MolFromSmiles(s_)
    return CalcMolFormula(m) if m is not None else ""


if __name__ == "__main__":
    dump_path, np_path, coco_path = sys.argv[1:4]
    d = pickle.load(open(dump_path, "rb"))
    sub = pd.read_csv(dump_path + ".sub.csv", dtype=str, keep_default_na=False).set_index("molecule_id").smiles
    art = ROOT / "artifacts"
    test = pd.read_parquet(art / "np_test.parquet")
    truth = test.drop_duplicates("molecule_id").set_index("molecule_id").normalized_smiles
    import importlib.util
    spec = importlib.util.spec_from_file_location("v4chem", EXT / "v4g" / "casmi26-v4b-models" / "code" / "casmi" / "chem.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    _, target = eo.molecule_spectra(test, {"chem": mod})
    tier = eo.Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    coco_npz = Path(dump_path).with_name("coconut_extra.npz")      # written by the pipeline next to its output
    coco = CocoExtra(str(coco_npz)) if coco_npz.is_file() else None
    members = set(pd.read_parquet(np_path, columns=["ik14"]).ik14.dropna().astype(str).str.slice(0, 14))
    members |= set(pd.read_csv(coco_path, usecols=["standard_inchi_key"]).standard_inchi_key.dropna().str.slice(0, 14))
    cats, ranked_info = Counter(), []
    for m in d["ids"]:
        ans_smi = truth[m]
        a14, a_form = std14(ans_smi), formula(ans_smi)
        final = sub.get(m, "").split(";")
        fk = [std14(s_) for s_ in final]
        rows = d["pub"].get(m, [])
        short14 = set()
        for r, _ in rows:
            short14.add(bytes(tier.ik[r]).decode() if r >= 0 else (bytes(coco.ik[-1 - r]).decode() if coco else ""))
        short14 |= {str(k)[:14] for k in d["hits"].get(m, {})}
        if a14 in fk:
            rank = fk.index(a14) + 1
            cat = "rank 1" if rank == 1 else "rank 2-25"
            if rank > 1:
                above = final[: rank - 1]
                ranked_info.append(dict(rank=rank, answer_member=a14 in members,
                                        above_members=np.mean([std14(s_) in members for s_ in above]),
                                        above_isomers=np.mean([formula(s_) == a_form for s_ in above])))
        elif a14 in short14:
            cat = "in shortlist, below 25"
        else:
            lo, hi = tier.window(target[m], 10.0) if m in target else (0, 0)
            in_win = hi > lo and a14.encode() in set(np.asarray(tier.ik[lo:hi]).tolist())
            cat = "in window, cut before the shortlist" if in_win else "not in the window"
        cats[cat] += 1
    n = len(d["ids"])
    print(f"class-2 answers (n {n}):")
    for c in ("rank 1", "rank 2-25", "in shortlist, below 25", "in window, cut before the shortlist", "not in the window"):
        print(f"  {c:38s} {cats[c]:4d} ({cats[c] / n:.1%})")
    if ranked_info:
        r = pd.DataFrame(ranked_info)
        print(f"answers ranked 2-25 (n {len(r)}): median rank {r['rank'].median():.0f}; answer a member {r.answer_member.mean():.1%}; "
              f"candidates above it: members {r.above_members.mean():.1%}, isomers of the answer {r.above_isomers.mean():.1%}")
