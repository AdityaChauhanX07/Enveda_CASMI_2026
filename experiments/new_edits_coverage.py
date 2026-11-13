"""Task 2, step 1: could more natural-product edits reach panel answers the generator does not rebuild today?

For each candidate edit: answers with a lighter relative (the group removed) or a heavier relative (the group added)
in the PubChem tier, over all answers and over those the current package does not rebuild (from the dumped
class-3 state, artifacts/state_c3.pkl)."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT))
from casmi.edit_overlay import Chemistry, Tier  # noqa: E402

NEW_FORWARD = {     # lighter relative (OH or aromatic CH) -> product
    "sulfate": "[OX2H1:1]>>[O:1]S(=O)(=O)O",
    "coumaroyl": "[OX2H1:1]>>[O:1]C(=O)C=Cc1ccc(O)cc1",
    "caffeoyl": "[OX2H1:1]>>[O:1]C(=O)C=Cc1ccc(O)c(O)c1",
    "feruloyl": "[OX2H1:1]>>[O:1]C(=O)C=Cc1ccc(O)c(OC)c1",
    "sinapoyl": "[OX2H1:1]>>[O:1]C(=O)C=Cc1cc(OC)c(O)c(OC)c1",
    "galloyl": "[OX2H1:1]>>[O:1]C(=O)c1cc(O)c(O)c(O)c1",
    "O-prenyl": "[OX2H1:1]>>[O:1]CC=C(C)C",
    "C-hexose(ar)": "[cH1:1]>>[c:1]C1OC(CO)C(O)C(O)C1O",
}
NEW_STRIP = {       # heavier relative -> product (the group taken off again)
    "sulfate": "[#6:2][O:1]S(=O)(=O)[OX2H1,OX1-]>>[#6:2][O:1]",
    "coumaroyl": "[#6:2][O:1]C(=O)C=Cc1ccc([OH])cc1>>[#6:2][O:1]",
    "caffeoyl": "[#6:2][O:1]C(=O)C=Cc1ccc([OH])c([OH])c1>>[#6:2][O:1]",
    "feruloyl": "[#6:2][O:1]C(=O)C=Cc1ccc([OH])c(O[CH3])c1>>[#6:2][O:1]",
    "sinapoyl": "[#6:2][O:1]C(=O)C=Cc1cc(O[CH3])c([OH])c(O[CH3])c1>>[#6:2][O:1]",
    "galloyl": "[#6:2][O:1]C(=O)c1cc([OH])c([OH])c([OH])c1>>[#6:2][O:1]",
    "O-prenyl": "[#6:2][O:1][CH2][CH]=C([CH3])[CH3]>>[#6:2][O:1]",
    "C-hexose(ar)": "[c:1]C1OC(CO)C(O)C(O)C1O>>[c:1]",
}

if __name__ == "__main__":
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    Chem = chem.Chem
    c3 = pickle.load(open(ROOT / "artifacts" / "state_c3.pkl", "rb"))
    rebuilt = {m for m, st in c3.items() if any(k == m for _, k, _ in st["scored"])}
    t = pq.read_table(ROOT.parents[0] / "casmi_data" / "raw" / "train.parquet", columns=["ingest_lib", "normalized_smiles"],
                      filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas().drop_duplicates("normalized_smiles")
    t["key"] = [chem.key(s) for s in t.normalized_smiles]
    t = t.drop_duplicates("key")
    print(f"answers {len(t)}; rebuilt by the current package {len(rebuilt & set(t.key))}")
    fwd = {e: AllChem.ReactionFromSmarts(s) for e, s in NEW_FORWARD.items()}
    strip = {e: AllChem.ReactionFromSmarts(s) for e, s in NEW_STRIP.items()}

    def in_pubchem(smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False
        conn, mass = Chem.MolToInchiKey(m)[:14].encode(), ExactMolWt(m)
        lo, hi = int(np.searchsorted(tier.mass, mass - 5e-4)), int(np.searchsorted(tier.mass, mass + 5e-4, "right"))
        return hi > lo and bool((np.asarray(tier.ik[lo:hi]) == conn).any())

    rows = {e: [0, 0, 0, 0] for e in NEW_FORWARD}          # lighter all, heavier all, either all, either not-rebuilt
    new_any = set()
    for s, key in zip(t.normalized_smiles, t.key):
        for e in NEW_FORWARD:
            light = any(in_pubchem(p) for k, p in chem.products(s, strip[e], cap=50).items() if k != key)
            heavy = any(in_pubchem(p) for k, p in chem.products(s, fwd[e], cap=50).items() if k != key)
            rows[e][0] += light; rows[e][1] += heavy; rows[e][2] += light or heavy
            if (light or heavy) and key not in rebuilt:
                rows[e][3] += 1; new_any.add(key)
    print(f"{'edit':14s} {'lighter':>8s} {'heavier':>8s} {'either':>7s} {'either, not rebuilt now':>24s}")
    for e, (a, b, c, d) in rows.items():
        print(f"{e:14s} {a:8d} {b:8d} {c:7d} {d:24d}")
    print(f"answers not rebuilt now that some new edit could reach: {len(new_any)} of {len(set(t.key) - rebuilt)}")
