"""Library overlay when answers are less documented than on the panel (the a7 lesson).

Class-2 fork-like lists are rebuilt with the answer's documentation capped at d (fold-safe FPNet z + 0.4 * doc),
so the answer sinks below documented wrong candidates as it would on a less documented test; then candidates
whose same-adduct reference spectra all mismatch (best cosine < tau, >= 3 spectra; flags from the top-26 analysis
in artifacts/lib_discrim_26.parquet from library_spectra_discrimination.py) move down `down` places (or to the end). The answer itself has no public spectra (class 2).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi import library_overlay as lo
from casmi.panel import load_candidates, zscore
from network_panel import shortlist

S = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts"   # lib_discrim_26.parquet (library_spectra_discrimination.py)
R = pd.read_parquet(S / "lib_discrim_26.parquet")
info = {(r.molecule, r.cand): (0 if pd.isna(r.n_add) else r.n_add, 0.0 if pd.isna(r.sim_add) else r.sim_add) for r in R.itertuples()}
c = load_candidates()
z = zscore(c, "fold0_full")
for d in (None, 8, 6, 4, 2):
    doc = c.doc.to_numpy().copy()
    if d is not None:
        doc = np.where(c.is_truth.to_numpy(), np.minimum(doc, d), doc)
    L = shortlist(c, z + 0.4 * doc)
    L = L[L.rank0 < 25].merge(c[["molecule", "tier_row", "connectivity"]].drop_duplicates(["molecule", "tier_row"]), on=["molecule", "tier_row"])
    L = L.sort_values(["molecule", "rank0"])
    out = {}
    for tau, down in ((0.5, 3), (0.5, 6), (0.5, 10), (0.5, -1), (0.6, 3), (0.6, -1)):
        before, after, helped = 0.0, 0.0, 0
        for m, g in L.groupby("molecule", sort=False):
            truth = g.truth_id.to_numpy()
            flags = [(not t) and info.get((m, k), (0, 0.0))[0] >= 3 and info.get((m, k), (0, 0.0))[1] < tau for k, t in zip(g.connectivity, truth)]
            o = lo.reorder(flags, down)
            rb = 1 / (np.flatnonzero(truth)[0] + 1) if truth.any() else 0.0
            ra = 1 / (np.flatnonzero(truth[o])[0] + 1) if truth.any() else 0.0
            before += rb; after += ra; helped += ra > rb
        out[(tau, down)] = (before / 250, after / 250, helped)
    cells = " | ".join(f"{t} {'end' if dn < 0 else 'down ' + str(dn)}: {v[1] - v[0]:+.4f}" for (t, dn), v in out.items())
    print(f"answer doc {'as is' if d is None else '<= ' + str(d):>7}: class-2 MRR {out[(0.5, 3)][0]:.4f} | {cells}")
