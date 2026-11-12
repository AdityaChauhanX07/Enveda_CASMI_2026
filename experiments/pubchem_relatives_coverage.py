"""Class-3 generator, step 0: how many panel answers have a one-edit relative in the PubChem tier?

For an answer A: lighter relatives = A with one modification removed (sugar, methyl, hydroxyl, acetyl, ...),
heavier relatives = A with one modification added. If a lighter relative R is in PubChem, the forward edit applied
to R can regenerate A; if a heavier one is, the reverse edit can. Presence is checked by standard InChIKey14 within
+-0.5 mDa of the relative's mass in the 105.9 M-row tier.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT))
from casmi.edit_overlay import FORWARD, Chemistry, Tier  # noqa: E402

# lighter relatives: the modification of each forward edit taken off again
STRIP = {
    "O-methyl": "[CH3][OX2:1][#6:2]>>[O:1][#6:2]",
    "N-methyl": "[CH3][NX3;!$(NC=O):1]>>[N:1]",
    "C-methyl(ar)": "[CH3][c:1]>>[c:1]",
    "OH(ar)": "[OX2H1][c:1]>>[c:1]",
    "OH(aliph)": "[OX2H1][CX4:1]>>[C:1]",
    "hexose": "[O:1]C1OC(CO)C(O)C(O)C1O>>[O:1]",
    "deoxyhexose": "[O:1]C1OC(C)C(O)C(O)C1O>>[O:1]",
    "pentose": "[O:1]C1OCC(O)C(O)C1O>>[O:1]",
    "glucuronide": "[O:1]C1OC(C(=O)O)C(O)C(O)C1O>>[O:1]",
    "acetyl": "[O:1]C(C)=O>>[O:1]",
    "H2": "[CX4;!H0:1][CX4;!H0:2]>>[C:1]=[C:2]",
    "prenyl(ar)": "[c:1][CH2][CH]=C([CH3])[CH3]>>[c:1]",
    "malonyl": "[O:1]C(=O)CC(=O)O>>[O:1]",
}

if __name__ == "__main__":
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    t = pq.read_table(ROOT.parents[0] / "casmi_data" / "raw" / "train.parquet", columns=["ingest_lib", "normalized_smiles"],
                      filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas().drop_duplicates("normalized_smiles")
    strip = {k: AllChem.ReactionFromSmarts(s) for k, s in STRIP.items()}
    Chem = chem.Chem

    def in_pubchem(smiles: str) -> bool:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False
        conn = Chem.MolToInchiKey(m)[:14].encode()
        mass = chem.mass(smiles)
        lo, hi = int(np.searchsorted(tier.mass, mass - 5e-4, "left")), int(np.searchsorted(tier.mass, mass + 5e-4, "right"))
        return hi > lo and bool((np.asarray(tier.ik[lo:hi]) == conn).any())

    light, heavy, any_rel = Counter(), Counter(), 0
    n_light, n_heavy = [], []
    for s in t.normalized_smiles:
        a_key = chem.key(s)
        found_l = {e for e, rx in strip.items() if any(in_pubchem(p) for k, p in chem.products(s, rx, cap=50).items() if k != a_key)}
        found_h = {e for e, (_, rx) in chem.fwd.items() if any(in_pubchem(p) for k, p in chem.products(s, rx, cap=50).items() if k != a_key)}
        light.update(found_l); heavy.update(found_h)
        n_light.append(len(found_l)); n_heavy.append(len(found_h))
        any_rel += bool(found_l or found_h)
    n = len(t)
    print(f"answers {n}: with a lighter relative in PubChem {np.mean(np.array(n_light) > 0):.2f}, "
          f"heavier {np.mean(np.array(n_heavy) > 0):.2f}, either {any_rel / n:.2f}")
    print("lighter relative by edit:", dict(light.most_common()))
    print("heavier relative by edit:", dict(heavy.most_common()))
