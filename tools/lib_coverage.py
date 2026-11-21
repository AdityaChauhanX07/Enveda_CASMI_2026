"""New-structure coverage of candidate spectral libraries vs the 0.495 notebook's sources (train + extlib-gnps-full + MassBank h)."""
import json, re, sys
import pandas as pd, pyarrow.parquet as pq
S, U = sys.argv[1], sys.argv[2]   # S: folder with train_ik14.parquet; U: folder with the downloaded libraries
M = dict(C=12.0, H=1.0078250319, N=14.0030740052, O=15.9949146221, S=31.97207069, P=30.97376151, F=18.99840320,
         Cl=34.96885271, Br=78.9183376, I=126.904468, Na=22.98976966, K=38.9637069, Si=27.9769265, B=11.0093055, Se=79.9165218)
def mass(f):
    try:
        return sum(M[e] * (int(n) if n else 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", str(f)))
    except KeyError:
        return float("nan")
tr = set(pd.read_parquet(f"{S}/train_ik14.parquet").inchikey14.astype(str))
A = pd.read_parquet(f"{U}/extgnps/ext_meta.parquet", columns=["inchikey14", "source_library"])
mb = pq.read_table("/home/user/casmi_data/ext/massbank_h/spectra.parquet", columns=["inchikey", "is_casmi_subset"]).to_pandas()
B = set(mb.inchikey[~mb.is_casmi_subset.astype(bool)].astype(str).str[:14])
cur = tr | set(A.inchikey14.astype(str)) | B
print(f"current sources: train {len(tr)} | extlib-gnps-full {A.inchikey14.nunique()} | MassBank h {len(B)} | union {len(cur)}")
out = {}
def report(name, df, kcol, lcol, fcol):
    k = df[kcol].astype(str).str[:14]
    m = df[fcol].map(mass) if fcol else pd.Series(float("nan"), index=df.index)
    new = ~k.isin(cur); inrange = m.between(157, 1159) | m.isna()
    r = dict(spectra=len(df), structures=int(k.nunique()), not_in_train=int(k[~k.isin(tr)].nunique()),
             new_vs_current=int(k[new].nunique()), new_in_test_mass_range=int(k[new & inrange].nunique()),
             spectra_of_new=int(new.sum()))
    if lcol:
        g = pd.DataFrame(dict(k=k[new], lib=df.loc[new, lcol].astype(str))).groupby("lib").k.nunique().sort_values(ascending=False)
        r["new_structures_by_library"] = {a: int(b) for a, b in g.head(15).items()}
    out[name] = r
    print(f"\n== {name}: {json.dumps({x: y for x, y in r.items() if x != 'new_structures_by_library'})}")
    for a, b in r.get("new_structures_by_library", {}).items():
        print(f"     {b:6d}  {a}")
g = pq.read_table(f"{U}/gnpscc0/gnps_cc0_trainschema.parquet", columns=["inchikey14", "library", "molecular_formula"]).to_pandas()
report("gnps-community-cc0", g, "inchikey14", "library", "molecular_formula")
o = pq.read_table(f"{U}/openmsms/public_spectra.parquet", columns=["inchikey14", "source", "library", "formula", "clean"]).to_pandas()
o["lib"] = o.source.astype(str) + ":" + o.library.astype(str)
report("open-msms (all)", o, "inchikey14", "lib", "formula")
report("open-msms (clean)", o[o.clean.astype(bool)], "inchikey14", "lib", "formula")
ms = pd.read_csv(f"{U}/msnlib/ms-pred/data/spec_datasets/msnlib/labels.tsv", sep="\t")
ms["lib"] = ms.spec.astype(str).str.split("_").str[0]
report("msnlib (ms-pred labels)", ms, "inchikey", None, "formula")
# overlaps among candidates' new structures
sets = {}
for name, df, kc in (("gnps-cc0", g, "inchikey14"), ("open-msms", o, "inchikey14"), ("msnlib", ms, "inchikey")):
    k = df[kc].astype(str).str[:14]; sets[name] = set(k[~k.isin(cur)])
print("\nunion of new structures over all three:", len(set().union(*sets.values())),
      "| gnps-cc0 & open-msms:", len(sets["gnps-cc0"] & sets["open-msms"]), "| msnlib & gnps-cc0:", len(sets["msnlib"] & sets["gnps-cc0"]))
json.dump(out, open(f"{S}/analysis/lib_coverage.json", "w"), indent=1)
