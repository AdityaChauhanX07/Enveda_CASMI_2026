"""Where does the relative search lose the answer? For each panel answer, find its true one-edit relatives in the
PubChem tier and report their documentation rank inside the edit window (the generator keeps the best 500)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT))
from casmi.edit_overlay import Chemistry, Tier  # noqa: E402

if __name__ == "__main__":
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    t = pq.read_table(ROOT.parents[0] / "casmi_data" / "raw" / "train.parquet", columns=["ingest_lib", "normalized_smiles"],
                      filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas().drop_duplicates("normalized_smiles")
    Chem = chem.Chem
    best, window_sizes = [], []
    for s in t.normalized_smiles:
        a_key, M = chem.key(s), chem.mass(s)
        ranks = []
        for e, (d, rx) in list(chem.strip.items()) + list(chem.fwd.items()):
            sign = -1 if e in chem.strip and rx is chem.strip[e][1] else +1      # strip -> lighter relative
            for k, p in chem.products(s, rx, cap=50).items():
                if k == a_key:
                    continue
                m = Chem.MolFromSmiles(p)
                if m is None:
                    continue
                conn = Chem.MolToInchiKey(m)[:14].encode()
                lo, hi = tier.window(M + sign * d, 10.0)
                if hi <= lo:
                    continue
                ik = np.asarray(tier.ik[lo:hi])
                hit = np.flatnonzero(ik == conn)
                if not len(hit):
                    continue
                doc = np.asarray(tier.lsid[lo:hi], float) + np.asarray(tier.lpmid[lo:hi], float)
                r = int((doc > doc[hit[0]]).sum())
                ranks.append(r); window_sizes.append(hi - lo)
        best.append(min(ranks) if ranks else None)
    b = np.array([x for x in best if x is not None])
    print(f"answers with a relative found: {len(b)}/{len(best)}")
    for k in (50, 100, 200, 500, 1000, 2000, 5000):
        print(f"  best relative within the top {k:5d} by documentation: {np.mean(b < k):.2f}")
    print("median window size", int(np.median(window_sizes)))
