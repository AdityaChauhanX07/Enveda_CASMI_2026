"""Edge-case harness for the two overlay cells: a deliberately awkward copy of the visible test and a submission with
odd candidate lists, run through casmi/network_overlay.py and casmi/edit_overlay.py exactly as the cells do.

usage: python experiments/edge_cases.py [--real]   (--real: the real FPNet models, needs torch; default random logits)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "edge"


def awkward_test(test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mutate a few molecules of the visible test into the odd cases the hidden test could contain."""
    t = test.copy()
    t["ms2_mzs"] = t.ms2_mzs.map(lambda a: np.asarray(a, np.float64))
    t["ms2_normalized_intensities"] = t.ms2_normalized_intensities.map(lambda a: np.asarray(a, np.float64))
    t["collision_energy_ev"] = t.collision_energy_ev.map(lambda a: None if a is None else np.asarray(a, np.float64))
    mols = t.molecule_id.drop_duplicates().tolist()
    order = t.groupby("molecule_id").precursor_mz.median().sort_values()
    case = {"empty_peaks": mols[0], "peaks_above_precursor": mols[1], "ce_none": mols[2], "ce_empty": mols[3],
            "ce_nan": mols[4], "single_peak": mols[5], "nan_intensities": mols[6], "many_spectra": mols[7],
            "lightest": order.index[0], "heaviest": order.index[-1]}
    t = t.reset_index(drop=True)
    idx = lambda m: list(np.flatnonzero((t.molecule_id == m).to_numpy()))

    def put(col, rows, fn):                           # build the column as a list (pandas .at stores 0-d arrays)
        vals = t[col].tolist()
        for r in rows:
            vals[r] = fn(r, vals[r])
        t[col] = vals

    put("ms2_mzs", idx(case["empty_peaks"]), lambda r, v: np.zeros(0))
    put("ms2_normalized_intensities", idx(case["empty_peaks"]), lambda r, v: np.zeros(0))
    put("precursor_mz", idx(case["peaks_above_precursor"]),
        lambda r, v: float(np.min(t.ms2_mzs[r])) - 5.0 if len(t.ms2_mzs[r]) else 100.0)
    put("collision_energy_ev", idx(case["ce_none"]), lambda r, v: None)
    put("collision_energy_ev", idx(case["ce_empty"]), lambda r, v: np.zeros(0))
    put("collision_energy_ev", idx(case["ce_nan"]), lambda r, v: np.array([np.nan]))
    put("ms2_mzs", idx(case["single_peak"]), lambda r, v: np.asarray(v)[:1])
    put("ms2_normalized_intensities", idx(case["single_peak"]), lambda r, v: np.ones(1))
    put("ms2_normalized_intensities", idx(case["nan_intensities"]),
        lambda r, v: np.where(np.arange(len(v)) % 2 == 0, np.nan, np.asarray(v, float)))
    extra = t[t.molecule_id == case["many_spectra"]]
    reps = [extra.assign(spectrum_id=extra.spectrum_id.astype(str) + f"_dup{r}") for r in range(16 // max(1, len(extra)) + 1)]
    t = pd.concat([t] + reps, ignore_index=True)
    keep = t[t.molecule_id == case["many_spectra"]].index[16:]
    t = t.drop(index=keep).reset_index(drop=True)
    return t, case


def awkward_lists(base: pd.DataFrame, case: dict) -> tuple[pd.DataFrame, dict]:
    s = base.set_index("molecule_id").smiles.to_dict()
    mols = list(s)
    odd = {"invalid_smiles": mols[10], "single_candidate": mols[11], "duplicates": mols[12], "stereo_charged_salt": mols[13],
           "empty_element": mols[14], "all_invalid": mols[15], "huge_molecule": mols[16]}
    first = lambda m: s[m].split(";")
    s[odd["invalid_smiles"]] = ";".join(["C1CC", "xyz"] + first(odd["invalid_smiles"])[:20])
    s[odd["single_candidate"]] = first(odd["single_candidate"])[0]
    s[odd["duplicates"]] = ";".join([first(odd["duplicates"])[0]] * 3 + first(odd["duplicates"])[1:10])
    s[odd["stereo_charged_salt"]] = ";".join(["C[C@H](N)C(=O)O", "CC(=O)[O-].[Na+]", "C[N+](C)(C)C"] + first(odd["stereo_charged_salt"])[:5])
    s[odd["empty_element"]] = ";".join(first(odd["empty_element"])[:3] + [""] + first(odd["empty_element"])[3:8])
    s[odd["all_invalid"]] = "C1CC;xyz;(("
    s[odd["huge_molecule"]] = ";".join(["CC(C)C" + "C(=O)NC(C)" * 40 + "C(=O)O"] + first(odd["huge_molecule"])[:5])
    return pd.DataFrame({"molecule_id": list(s), "smiles": list(s.values())}), odd


def check(path: Path, ids: list) -> str:
    f = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert f.molecule_id.tolist() == ids, "molecule order changed"
    n = f.smiles.str.split(";").map(len)
    assert n.between(1, 25).all(), f"list sizes out of range: {n.min()}-{n.max()}"
    return f"ok ({len(f)} rows, sizes {n.min()}-{n.max()})"


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from casmi import edit_overlay, network_overlay
    real = "--real" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    data = Path(sys.argv[sys.argv.index("--data") + 1]) if "--data" in sys.argv else ROOT.parents[0] / "casmi_data" / "raw"
    test, case = awkward_test(pd.read_parquet(data / "test.parquet"))
    test.to_parquet(OUT / "test.parquet")
    ids = test.molecule_id.drop_duplicates().tolist()
    # realistic lists: the best-documented PubChem structures of each molecule's mass window
    kw = {} if real else dict(tier_root=Path(sys.argv[sys.argv.index("--tier") + 1]) if "--tier" in sys.argv else
                              ROOT.parents[0] / "casmi_data" / "ext" / "pubchem_prior" / "casmi26-pubchem-tier")
    if real:
        doc = edit_overlay._find("pc_ik14.npy").parent
        tier = edit_overlay.Tier(doc / "casmi26-pubchem-tier" if (doc / "casmi26-pubchem-tier" / "pc_mass.npy").is_file()
                                 else edit_overlay._find("casmi26-pubchem-tier/pc_mass.npy").parent, doc)
        v4 = edit_overlay._load_v4(edit_overlay._find("casmi26-v4b-models/code/casmi/fpnet.py").parent)
    else:
        import importlib.util
        tier = edit_overlay.Tier(kw["tier_root"])
        code = ROOT.parents[0] / "casmi_data" / "ext" / "v4g" / "casmi26-v4b-models" / "code" / "casmi" / "chem.py"
        spec = importlib.util.spec_from_file_location("_v4chem", code); v4chem = importlib.util.module_from_spec(spec); spec.loader.exec_module(v4chem)
        v4 = {"chem": v4chem}
    spectra, target = edit_overlay.molecule_spectra(test, v4)
    rows = []
    for m in ids:
        lo, hi = tier.window(target[m], 10.0) if m in target else (0, 0)
        if hi > lo:
            d = np.asarray(tier.lsid[lo:hi], float) + np.asarray(tier.lpmid[lo:hi], float)
            rows.append((m, ";".join(tier.smiles(int(r)) for r in lo + np.argsort(-d, kind="stable")[:25])))
        else:
            rows.append((m, "CCO"))
    sub, odd = awkward_lists(pd.DataFrame(rows, columns=["molecule_id", "smiles"]), case)
    sub.to_csv(OUT / "sub.csv", index=False)
    report = {"cases": case, "list_cases": odd}
    t0 = time.time()
    rep_net = network_overlay.apply(OUT / "test.parquet", OUT / "sub.csv", OUT / "sub_net.csv", None)
    report["network"] = dict(rep_net, params=None, check=check(OUT / "sub_net.csv", ids), seconds=round(time.time() - t0, 1))
    print("NETWORK", report["network"], flush=True)
    rng = np.random.default_rng(0)
    fake = {m: rng.normal(-1.6, 1.6, 10226) for m in ids}
    p = dict(device="cuda" if real else "cpu", relatives=1, max_seconds=1500, rel_seconds=900)
    t0 = time.time()
    if real:
        rep = edit_overlay.apply(OUT / "test.parquet", OUT / "sub_net.csv", OUT / "sub_edit.csv", None, p)
    else:
        rep = edit_overlay.apply(OUT / "test.parquet", OUT / "sub_net.csv", OUT / "sub_edit.csv", None, p, v4=v4,
                                 logits_fn=fake.get, fold_model=ROOT.parents[0] / "casmi_data" / "ext" / "foldsafe" / "fpnet_fold0.pt",
                                 fp_bits=ROOT.parents[0] / "casmi_data" / "ext" / "foldsafe" / "fp_bits.npy", **kw)
    report["edit"] = dict({k: v for k, v in rep.items() if k != "params"}, check=check(OUT / "sub_edit.csv", ids),
                          seconds=round(time.time() - t0, 1))
    print("EDIT", report["edit"], flush=True)
    out = pd.read_csv(OUT / "sub_edit.csv", dtype=str, keep_default_na=False).set_index("molecule_id").smiles
    for name, m in list(case.items()) + list(odd.items()):
        print(f"  {name:24s} {m}: {len(out[m].split(';'))} candidates", flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
