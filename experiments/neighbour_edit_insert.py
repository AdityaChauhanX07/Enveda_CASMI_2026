"""Where to insert neighbour-edit products: gain on class-3 simulations vs harm when the answer is in the list."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.data import DATA  # noqa: E402
from casmi.metric import ik14  # noqa: E402
from casmi.panel import load_candidates, tier_smiles, zscore  # noqa: E402
from neighbour_edit_oracle import EDITS, products  # noqa: E402
from network_panel import shortlist  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def rr(keys, truth):
    for i, k in enumerate(keys[:25]):
        if k == truth:
            return 1.0 / (i + 1)
    return 0.0


if __name__ == "__main__":
    c = load_candidates()
    d = shortlist(c, zscore(c, "fold0_full") + 0.4 * c.doc.to_numpy())
    d = d[d.rank0 < 26].reset_index(drop=True)
    d["smiles"] = tier_smiles(d.tier_row.to_numpy())
    d["key"] = [ik14(s) if Chem.MolFromSmiles(s) is not None else "" for s in d.smiles]
    lists = {m: g for m, g in d.groupby("molecule", sort=False)}
    t = pq.read_table(DATA / "train.parquet", columns=["ingest_lib", "normalized_smiles", "inchikey14"],
                      filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas().drop_duplicates("inchikey14")
    truth = {k: ik14(s) for k, s in zip(t.inchikey14, t.normalized_smiles)}
    mass = {k: rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(s)) for k, s in zip(t.inchikey14, t.normalized_smiles)}
    g = pd.read_parquet(ROOT / "artifacts" / "np_graph.parquet")
    e = g[(g.cosine >= 0.5) & (g.matches >= 4)]
    prods = {}
    for a, b in list(zip(e.a, e.b)) + list(zip(e.b, e.a)):
        if a not in lists or b not in mass or a not in mass:
            continue
        dm = mass[b] - mass[a]
        for name, (delta, rxn) in EDITS.items():
            if abs(dm - delta) > 0.003:
                continue
            for r, smi in enumerate(lists[a].smiles.tolist()[:3]):
                for k in sorted(products(Chem.MolFromSmiles(smi), rxn)):
                    prods.setdefault(b, []).append((r, k))
    print(f"molecules receiving products: {len(prods)}")
    for k_ins in (0, 1, 2, 4):
        for max_p in (1, 2, 3):
            gain2, gain3 = 0.0, 0.0
            for m, g_ in lists.items():
                base = g_.key.tolist()
                if m not in prods:
                    continue
                p = [k for _, k in sorted(prods[m])]
                p = list(dict.fromkeys(p))[:max_p]
                def insert(lst):
                    rest = [x for x in lst if x not in p]
                    return rest[:k_ins] + p + rest[k_ins:]
                gain2 += rr(insert(base), truth[m]) - rr(base, truth[m])          # answer in the PubChem window
                no_t = [x for x in base if x != truth[m]]
                gain3 += rr(insert(no_t), truth[m]) - rr(no_t, truth[m])          # class-3 simulation
            print(f"insert at rank {k_ins + 1}, up to {max_p} products: class-2 change {gain2 / 250:+.4f}, "
                  f"class-3 gain {gain3 / 250:+.4f} | net at 25% class 3: {0.75 * gain2 / 250 + 0.25 * gain3 / 250:+.4f}")
