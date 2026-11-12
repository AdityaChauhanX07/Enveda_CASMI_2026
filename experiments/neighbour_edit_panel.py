"""Neighbour-edit generation in a class-3 simulation on the panel.

B's own structure is removed from its candidates. Each spectral neighbour A (graph edge) whose neutral mass differs
from B's by a known edit contributes the edit products of A's top-3 candidates (fold-safe FPNet + doc order).
Reports how often B's structure is regenerated and the rank it would get if products are listed in A-rank order.
"""
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
REVERSE = {"O-methyl": "[O:1][CH3]>>[O:1]", "hexose": None}


if __name__ == "__main__":
    c = load_candidates()
    d = shortlist(c, zscore(c, "fold0_full") + 0.4 * c.doc.to_numpy())
    d = d[d.rank0 < 3].reset_index(drop=True)
    d["smiles"] = tier_smiles(d.tier_row.to_numpy())
    top = {m: g.smiles.tolist() for m, g in d.groupby("molecule", sort=False)}
    t = pq.read_table(DATA / "train.parquet", columns=["ingest_lib", "normalized_smiles", "inchikey14"],
                      filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas().drop_duplicates("inchikey14")
    truth_key = {k: ik14(s) for k, s in zip(t.inchikey14, t.normalized_smiles)}
    mass = {k: rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(s)) for k, s in zip(t.inchikey14, t.normalized_smiles)}
    g = pd.read_parquet(ROOT / "artifacts" / "np_graph.parquet")
    for cos_min in (0.3, 0.5, 0.7):
        e = g[(g.cosine >= cos_min) & (g.matches >= 4)]
        found, tried, nprod = {}, set(), []
        for a, b in list(zip(e.a, e.b)) + list(zip(e.b, e.a)):  # generate B from neighbour A
            if a not in top or b not in mass or a not in mass:
                continue
            dm = mass[b] - mass[a]
            for name, (delta, rxn) in EDITS.items():
                if abs(dm - delta) > 0.003:
                    continue
                tried.add(b)
                for r, smi in enumerate(top[a]):
                    prods = products(Chem.MolFromSmiles(smi), rxn)
                    nprod.append(len(prods))
                    if truth_key[b] in prods:
                        found[b] = min(found.get(b, 99), r)
        rr = sum(1.0 / (r + 1) for r in found.values())
        print(f"cos>={cos_min}: molecules with an edit-mass neighbour {len(tried)}, structure regenerated for "
              f"{len(found)} (from neighbour rank 1: {sum(v == 0 for v in found.values())}); "
              f"median products per candidate {np.median(nprod) if nprod else 0:.0f}; "
              f"MRR gain if listed by neighbour rank, over 250: +{rr / 250:.4f}")
