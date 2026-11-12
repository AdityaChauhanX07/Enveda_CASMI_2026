"""Upper bound for neighbour-driven generation: apply one modification to a related molecule's TRUE structure.

For every np-examples pair whose neutral masses differ by a known modification, apply that modification (or its
reverse) at every site of the lighter (heavier) structure and check whether the partner is among the products.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casmi.data import DATA  # noqa: E402
from casmi.metric import ik14  # noqa: E402

RDLogger.DisableLog("rdApp.*")
GLC = "C1OC(CO)C(O)C(O)C1O"
EDITS = {  # name: (mass delta, forward reaction SMARTS on the lighter structure)
    "O-methyl": (14.01565, "[OX2H1:1]>>[O:1]C"),
    "N-methyl": (14.01565, "[NX3;H1,H2;!$(NC=O):1]>>[N:1]C"),
    "C-methyl(ar)": (14.01565, "[cH1:1]>>[c:1]C"),
    "OH(ar)": (15.99491, "[cH1:1]>>[c:1]O"),
    "OH(aliph)": (15.99491, "[CX4;!H0:1]>>[C:1]O"),
    "hexose": (162.05282, "[OX2H1:1]>>[O:1]" + GLC.replace("C1O", "C1O", 1)),
    "deoxyhexose": (146.05791, "[OX2H1:1]>>[O:1]C1OC(C)C(O)C(O)C1O"),
    "pentose": (132.04226, "[OX2H1:1]>>[O:1]C1OCC(O)C(O)C1O"),
    "glucuronide": (176.03209, "[OX2H1:1]>>[O:1]C1OC(C(=O)O)C(O)C(O)C1O"),
    "acetyl": (42.01057, "[OX2H1:1]>>[O:1]C(C)=O"),
    "H2": (2.01565, "[C:1]=[C:2]>>[C:1][C:2]"),
    "prenyl(ar)": (68.06260, "[cH1:1]>>[c:1]CC=C(C)C"),
    "malonyl": (86.00039, "[OX2H1:1]>>[O:1]C(=O)CC(=O)O"),
}
for k, (dm, sm) in list(EDITS.items()):
    EDITS[k] = (dm, AllChem.ReactionFromSmarts(sm))


def products(mol, rxn, cap=400) -> set[str]:
    out = set()
    for ps in rxn.RunReactants((mol,), maxProducts=cap):
        p = ps[0]
        try:
            Chem.SanitizeMol(p)
            out.add(ik14(Chem.MolToSmiles(p)))
        except Exception:
            continue
    return out


if __name__ == "__main__":
    t = pq.read_table(DATA / "train.parquet", columns=["ingest_lib", "normalized_smiles", "inchikey14"],
                      filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas().drop_duplicates("inchikey14")
    mols = [Chem.MolFromSmiles(s) for s in t.normalized_smiles]
    t["mass"] = [rdMolDescriptors.CalcExactMolWt(m) for m in mols]
    t["key"] = [ik14(s) for s in t.normalized_smiles]
    t = t.reset_index(drop=True)
    m = t.mass.to_numpy()
    rows = []
    for name, (dm, rxn) in EDITS.items():
        for i in range(len(t)):
            for j in np.flatnonzero(np.abs(m - m[i] - dm) < 0.003):
                prods = products(mols[i], rxn)
                rows.append((name, t.inchikey14[i], t.inchikey14[j], t.key[j] in prods, len(prods)))
    r = pd.DataFrame(rows, columns=["edit", "light", "heavy", "hit", "n_products"])
    print(r.groupby("edit").agg(pairs=("hit", "size"), hits=("hit", "sum"), products=("n_products", "median")).to_string())
    hit_mols = set(r[r.hit].light) | set(r[r.hit].heavy)
    print(f"pairs {len(r)}, hits {int(r.hit.sum())}; molecules recoverable from a partner: {len(hit_mols)} / {len(t)}")
