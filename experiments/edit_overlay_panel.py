"""End-to-end check of casmi/edit_overlay.py on CPU: the 250 np-examples as the test set.

Lists = the panel's fold-safe FPNet + 0.4 * documentation order (top 25, SMILES from the PubChem tier), once as is
(class-2 situation) and once with each molecule's own structure removed (class-3 simulation). Both are passed
through the overlay exactly as in the notebook and scored with the official metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi import edit_overlay  # noqa: E402
from casmi.data import DATA  # noqa: E402
from casmi.metric import ik14, score  # noqa: E402
from casmi.panel import load_candidates, tier_smiles, zscore  # noqa: E402
from network_panel import shortlist  # noqa: E402

if __name__ == "__main__":
    art = ROOT / "artifacts"; art.mkdir(exist_ok=True)
    t = pq.read_table(DATA / "train.parquet", filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas()
    t["molecule_id"] = t.inchikey14
    t.to_parquet(art / "np_test.parquet")
    truth = t.drop_duplicates("molecule_id").set_index("molecule_id").normalized_smiles
    c = load_candidates()
    d = shortlist(c, zscore(c, "fold0_full") + 0.4 * c.doc.to_numpy())
    d = d[d.rank0 < 26].reset_index(drop=True)
    d["smiles"] = tier_smiles(d.tier_row.to_numpy())
    lists = {m: g.smiles.tolist() for m, g in d.groupby("molecule", sort=False)}
    sol = pd.DataFrame({"molecule_id": truth.index, "smiles": truth.values})
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("_v4chem", EXT / "v4g" / "casmi26-v4b-models" / "code" / "casmi" / "chem.py")
    v4chem = importlib.util.module_from_spec(spec); spec.loader.exec_module(v4chem)
    zl = np.load(EXT / "panel" / "panel_logits.npz")
    cached = {k: zl["fold0"][i].astype(np.float64) for i, k in enumerate(zl["keys"])}
    kw = dict(p={"device": "cpu"}, v4={"chem": v4chem}, logits_fn=cached.get,
              fold_model=EXT / "foldsafe" / "fpnet_fold0.pt", tier_root=EXT / "pubchem_prior" / "casmi26-pubchem-tier",
              fp_bits=EXT / "foldsafe" / "fp_bits.npy")
    mixed = [a for a in sys.argv[1:] if a.startswith("--mixed=")]
    runs = [(f"mixed seed {sd}", int(sd)) for sd in mixed[0].split("=")[1].split(",")] if mixed else \
        [("class-2 lists", None), ("class-3 simulation", -1)]
    for name, seed in runs:
        rng = np.random.default_rng(seed if seed is not None and seed >= 0 else 0)
        dropped = set(sol.molecule_id) if seed == -1 else set()
        if seed is not None and seed >= 0:
            dropped = set(rng.choice(sol.molecule_id.to_numpy(), len(sol) // 4, replace=False))
        rows = []
        for m in sol.molecule_id:
            cand = [s for s in lists.get(m, ["CCO"]) if not (m in dropped and ik14(s) == ik14(truth[m]))][:25] or ["CCO"]
            rows.append((m, ";".join(cand)))
        sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
        sub.to_csv(art / "np_sub.csv", index=False)
        rep = edit_overlay.apply(art / "np_test.parquet", art / "np_sub.csv", art / "np_sub_edit.csv", None, **kw)
        after = pd.read_csv(art / "np_sub_edit.csv", dtype=str, keep_default_na=False)
        if dropped and len(dropped) < len(sol):
            ds, ks = sol[sol.molecule_id.isin(dropped)], sol[~sol.molecule_id.isin(dropped)]
            print(f"  {name}: dropped 25 %: {score(ds, sub[sub.molecule_id.isin(dropped)]):.4f} -> "
                  f"{score(ds, after[after.molecule_id.isin(dropped)]):.4f}; kept: "
                  f"{score(ks, sub[~sub.molecule_id.isin(dropped)]):.4f} -> {score(ks, after[~after.molecule_id.isin(dropped)]):.4f}")
        print(f"{name}: MRR {score(sol, sub):.4f} -> {score(sol, after):.4f} | "
              f"{ {k: rep[k] for k in ('with_neighbour', 'with_products', 'products', 'inserted', 'molecules_changed', 'seconds')} }")
