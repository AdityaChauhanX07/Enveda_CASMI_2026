"""Library spectra of panel list candidates: who has same-adduct reference spectra, and how well they match.

Step 1 (headroom): the panel's fork-like class-2 lists (fold-safe FPNet + 0.4 * doc, top 26) with the number of
train.parquet spectra per candidate (the answer's own spectra count as hidden).
Step 2 (discrimination): best cosine (top 60 peaks, sqrt, 0.01 Da) between a molecule's np-examples spectra and
each candidate's spectra in the same ion mode / same adduct; the answer's spectra from other libraries only.
Writes artifacts/panel_lists_lib.parquet and artifacts/lib_discrim_26.parquet (read by library_overlay_lowdoc.py).
"""
import sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.data import DATA
from casmi.network import prepare
from casmi.panel import load_candidates, zscore
from network_panel import shortlist
ART = ROOT / "artifacts"

lib = pq.read_table(DATA / "train.parquet", columns=["ingest_lib", "inchikey14", "ionization_mode", "adduct"]).to_pandas()
print("spectra", len(lib), "structures", lib.inchikey14.nunique())
has = lib.groupby("inchikey14").size()
c = load_candidates()
print(c.columns.tolist()[:30])
d = shortlist(c, zscore(c, "fold0_full") + 0.4 * c.doc.to_numpy())
d = d[d.rank0 < 26].reset_index(drop=True)
d = d.merge(c[["molecule", "tier_row", "connectivity", "doc"]].drop_duplicates(["molecule", "tier_row"]), on=["molecule", "tier_row"], how="left")
d["is_truth"] = d.truth_id
print("list rows", len(d), "molecules", d.molecule.nunique(), "conn sample", d.connectivity.head(3).tolist())
d["lib_n"] = d.connectivity.map(has).fillna(0).astype(int)
d.loc[d.is_truth, "lib_n"] = 0                     # class 2/3: the answer has no public spectra
cnt = Counter()
for m, g in d.groupby("molecule", sort=False):
    g = g.sort_values("rank0")
    tr = np.flatnonzero(g.is_truth.to_numpy())
    r = tr[0] if len(tr) else None
    top3 = g.head(3)
    cnt["molecules"] += 1
    cnt["wrong top-3 members with library spectra"] += int(((top3.lib_n > 0) & ~top3.is_truth).sum())
    cnt["wrong top-3 members"] += int((~top3.is_truth).sum())
    cnt["leader wrong"] += int(not g.is_truth.iloc[0])
    cnt["leader wrong and has library spectra"] += int((not g.is_truth.iloc[0]) and g.lib_n.iloc[0] > 0)
    if r is not None and r > 0:
        above = g.iloc[:r]
        cnt["answer below rank 1 (in list)"] += 1
        cnt["... every candidate above it has library spectra"] += int((above.lib_n > 0).all())
        cnt["... at least one above has library spectra"] += int((above.lib_n > 0).any())
    if r is None:
        cnt["answer not in top 25"] += 1
for k, v in cnt.items():
    print(f"{k:55s} {v}")
d[["molecule", "rank0", "is_truth", "connectivity", "lib_n", "doc"]].to_parquet(ART / "panel_lists_lib.parquet")

# ---- step 2: discrimination ----
d = pd.read_parquet(ART / "panel_lists_lib.parquet")
d = d[d.rank0 < 26]
need = set(d.connectivity) | set(d.molecule)
cols = ["ingest_lib", "inchikey14", "adduct", "ionization_mode", "instrument_type", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]
t0 = time.time()
import pyarrow as pa, pyarrow.compute as pc
pf = pq.ParquetFile(DATA / "train.parquet"); kset = pa.array(sorted(need)); parts = []
for g in range(pf.num_row_groups):          # row group by row group: a filtered read_table peaks at ~8 GB
    m_ = pc.is_in(pf.read_row_group(g, columns=["inchikey14"])["inchikey14"], value_set=kset)
    idx = np.flatnonzero(np.asarray(m_.to_numpy(zero_copy_only=False), dtype=bool))
    if len(idx):
        parts.append(pf.read_row_group(g, columns=cols).take(pa.array(idx)))
lib = pa.concat_tables(parts).to_pandas()
print("rows", len(lib), "structures", lib.inchikey14.nunique(), f"{time.time()-t0:.0f}s")
def prep(r):
    return prepare(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
lib["spec"] = [prep(r) for r in lib.itertuples()]
def cos(a, b, tol=0.01):
    (ma, wa), (mb, wb) = a, b
    if len(ma) == 0 or len(mb) == 0:
        return 0.0
    ii, jj = np.nonzero(np.abs(ma[:, None] - mb[None, :]) <= tol)
    if len(ii) == 0:
        return 0.0
    prod = wa[ii] * wb[jj]; order = np.argsort(-prod, kind="stable")
    ua, ub, tot = set(), set(), 0.0
    for k in order:
        i, j = ii[k], jj[k]
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); tot += prod[k]
    return float(tot)
by = defaultdict(list)
for r in lib.itertuples():
    by[r.inchikey14].append(r)
rng = np.random.default_rng(0)
rows = []
for m, g in d.groupby("molecule", sort=False):
    Q = [r for r in by.get(m, []) if r.ingest_lib == "enveda-np-examples"]
    for c in g.itertuples():
        refs = [r for r in by.get(c.connectivity, []) if r.ingest_lib != "enveda-np-examples" or c.connectivity != m]
        if c.connectivity == m:
            refs = [r for r in refs if r.ingest_lib != "enveda-np-examples"]
        if not refs:
            rows.append(dict(molecule=m, cand=c.connectivity, rank0=c.rank0, truth=bool(c.is_truth), n_ref=0)); continue
        if len(refs) > 300:
            refs = [refs[i] for i in rng.choice(len(refs), 300, replace=False)]
        best_mode, best_add, best_add_e180, n_add = 0.0, 0.0, 0.0, 0
        for q in Q:
            for r in refs:
                if r.ionization_mode != q.ionization_mode:
                    continue
                s = cos(q.spec, r.spec)
                best_mode = max(best_mode, s)
                if r.adduct == q.adduct:
                    n_add += 1; best_add = max(best_add, s)
                    if r.ingest_lib == "enveda-180":
                        best_add_e180 = max(best_add_e180, s)
        rows.append(dict(molecule=m, cand=c.connectivity, rank0=c.rank0, truth=bool(c.is_truth), n_ref=len(refs), n_add=n_add,
                         libs=",".join(sorted({r.ingest_lib for r in refs})), sim_mode=best_mode, sim_add=best_add, sim_e180=best_add_e180))
R = pd.DataFrame(rows)
R.to_parquet(ART / "lib_discrim_26.parquet")
print(f"done {time.time()-t0:.0f}s")
