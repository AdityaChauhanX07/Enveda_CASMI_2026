"""Which natural-product flag: LOTUS / NPAtlas (natural_products.parquet), COCONUT (CSV-lite export), or either?

Membership of the panel answers and of the decoys in C1's top 25 (by documentation), then C1s (spec2mol pooled 0.75)
+ beta x flag on the class-2 and class-1-view lists with the answer's documentation capped (as
experiments/np_prior_lowdoc.py). LOTUS / NPAtlas joins on the scorer key; COCONUT on the standard InChIKey's first
block (PubChem rows: the index's pc_ik14; library hits: from their SMILES).

usage: np_flags_compare.py NATURAL_PRODUCTS_PARQUET COCONUT_CSV
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import spec2mol_lowdoc as s2  # noqa: E402
from casmi.edit_overlay import Tier  # noqa: E402
from casmi.metric import ik14  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def std14(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(m)[:14] if m is not None else ""


if __name__ == "__main__":
    art = ROOT / "artifacts"
    ln = set(pd.read_parquet(sys.argv[1], columns=["scorer_key"]).scorer_key)
    co = set(pd.read_csv(sys.argv[2], usecols=["standard_inchi_key"]).standard_inchi_key.dropna().str.slice(0, 14))
    print(f"LOTUS / NPAtlas keys {len(ln):,}; COCONUT connectivity blocks {len(co):,}")
    data = s2.split_rows(art)
    ids = data["ids"]
    tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    d2 = pickle.load(open(art / "c1_dump_c2_v1.pkl", "rb"))
    dumps = {tag: pickle.load(open(art / f"c1_dump_{tag}_v1.pkl", "rb")) for tag in ("c2", "c1")}
    std = {}                                          # metric key -> standard InChIKey14s seen for it
    for m in ids:
        for (k, _, _), (r, _) in zip(data["rows"][m], d2["pub"].get(m, [])):
            if k:
                std.setdefault(k, set()).add(bytes(tier.ik[r]).decode())
        for tag, d in dumps.items():                # same order as split_rows built them
            for (key, _), (_, (c, s_)) in zip(data["hits"][tag][m], d["hits"].get(m, {}).items()):
                if key:
                    std.setdefault(key, set()).add(std14(s_))
    t = pd.read_parquet(art / "np_test.parquet", columns=["normalized_smiles"]).drop_duplicates()
    for s_ in t.normalized_smiles:                  # the answers' own standard keys
        std.setdefault(ik14(s_), set()).add(std14(s_))
    flags = {"LOTUS/NPAtlas": lambda k: k in ln, "COCONUT": lambda k: bool(std.get(k, set()) & co),
             "either": lambda k: k in ln or bool(std.get(k, set()) & co)}
    for name, f in flags.items():
        ans = np.mean([f(m) for m in ids])
        dec = []
        for m in ids:
            sc = s2.c1_scores(m, data["rows"][m], data["hits"]["c2"][m], None)
            docs = {k: doc for k, _, doc in data["rows"][m]}
            dec += [(f(k), docs.get(k, np.nan)) for k in sorted(sc, key=lambda k: -sc[k])[:25] if k != m]
        dd = pd.DataFrame(dec, columns=["member", "doc"]).dropna()
        bins = dd.groupby(pd.cut(dd.doc, [-1, 4, 6, 8, 12, 100]), observed=True).member.mean().round(3).tolist()
        print(f"{name:14s} answers {ans:.3f} | decoys in C1's top 25 by doc (<=4, 4-6, 6-8, 8-12, >12): {bins}")
    scores = json.load(open(art / "spec2mol_scores.json"))
    caps = (None, 8, 6, 4)
    for tag in ("c2", "c1"):
        sc0 = {c: {m: s2.c1_scores(m, data["rows"][m], data["hits"][tag][m], c) for m in ids} for c in caps}
        scale = s2.pooled_scale({m: list(sc0[None][m]) for m in ids}, scores)
        print(f"\n{tag}: C1s + beta x flag, answer doc as is / <= 8 / <= 6 / <= 4")
        for name, f in [("none", lambda k: False)] + list(flags.items()):
            for beta in ((0.0,) if name == "none" else (0.5, 1.0, 1.5)):
                cells = []
                for c in caps:
                    rr = 0.0
                    for m in ids:
                        sc = {k: v + (beta if f(k) else 0.0) for k, v in sc0[c][m].items()}
                        rr += s2.rr(s2.fuse_scores(sc, scores.get(m), 0.75, scale)[:25], m)
                    cells.append(rr / len(ids))
                print(f"  {name:14s} beta {beta:.1f} | " + " | ".join(f"{v:.4f}" for v in cells), flush=True)
