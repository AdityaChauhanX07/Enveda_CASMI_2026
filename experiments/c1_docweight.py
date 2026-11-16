"""How robust is C1's documentation weight when answers are less documented than on the panel (the a7 lesson)?

From the C1 class-2 dump (artifacts/c1_dump_c2_v1.pkl: each list's best 100 PubChem rows at w_doc 0.25, library hits
with the answer's spectra hidden), each row's score is split back into its fingerprint part (score - 0.25 x doc) and
its documentation (from the index), then re-scored as z + w x doc with the answer's documentation capped at d.
Library hits use C1's rule (+ 4 x (cos - 0.8)). Rows outside the dumped 100 count as absent (they would rank low).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT))
from casmi.edit_overlay import Chemistry, Tier  # noqa: E402

if __name__ == "__main__":
    d = pickle.load(open(ROOT / "artifacts" / "c1_dump_c2_v1.pkl", "rb"))
    tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    rows = {}
    for m in d["ids"]:
        out = []
        for r, total in d["pub"].get(m, []):
            doc = float(tier.lsid[r]) + float(tier.lpmid[r])
            out.append((chem.key(tier.smiles(r)), total - 0.25 * doc, doc))
        rows[m] = out
    hits = {m: [(chem.key(s), c) for _, (c, s) in h.items()] for m, h in d["hits"].items()}
    print("w_doc | answer doc as is | <= 8 | <= 6 | <= 4   (class 2, C1's library-hit rule)")
    for w in (0.0, 0.1, 0.15, 0.25, 0.4):
        cells = []
        for cap in (None, 8, 6, 4):
            tot = 0.0
            for m in d["ids"]:
                sc = {}
                for k, z, doc in rows[m]:
                    if not k or k in sc:
                        continue
                    dd = min(doc, cap) if (cap is not None and k == m) else doc
                    sc[k] = z + w * dd
                base = min(sc.values()) if sc else 0.0
                for k, c in hits.get(m, []):
                    if k:
                        sc[k] = sc.get(k, base) + 4.0 * max(0.0, c - 0.8)
                ks = [k for k, _ in sorted(sc.items(), key=lambda x: -x[1])][:25]
                tot += 1.0 / (ks.index(m) + 1) if m in ks else 0.0
            cells.append(tot / 250)
        print(f"{w:5.2f} | " + " | ".join(f"{c:.4f}" for c in cells))
