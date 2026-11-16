"""Panel check of the compliant pipeline (casmi/c1_pipeline.py) on the 250 np-examples.

Class 2: the answer's own spectra are hidden from the library search, so only the FPNet + documentation ranking can
find it. Class-1 view: only the queries' own library (enveda-np-examples) is left out, so the answer's spectra from
other libraries can produce a library hit. Logits: the cached five-model ensemble (as edit_overlay_panel.py
--ensemble). Index: the shared tier until our own build lands (--index=DIR uses that one, docs in the same folder).

usage: c1_panel.py [--w_doc=0.25] [--lib_hit=0.75] [--any pipeline parameter] [--index=DIR] [--only=c2|c1] [--n=10]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi import edit_overlay  # noqa: E402
sys.modules["edit_overlay"] = edit_overlay
from casmi import c1_pipeline  # noqa: E402
from casmi.data import DATA  # noqa: E402
from casmi.metric import score  # noqa: E402
from casmi.panel import read_library  # noqa: E402

if __name__ == "__main__":
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    art = ROOT / "artifacts"
    t = read_library("enveda-np-examples")
    from casmi.metric import ik14
    t["molecule_id"] = [ik14(x) for x in t.normalized_smiles]
    if "n" in args:                                   # smoke test on the first n molecules
        t = t[t.molecule_id.isin(t.molecule_id.drop_duplicates().head(int(args["n"])))]
    test_path = art / ("np_test.parquet" if "n" not in args else "np_test_small.parquet")
    t.to_parquet(test_path)
    truth = t.drop_duplicates("molecule_id").set_index("molecule_id").normalized_smiles
    sol = pd.DataFrame({"molecule_id": truth.index, "smiles": truth.values})
    from rdkit import Chem
    std = {m: Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for m, s in truth.items()}
    zl = np.load(art / "panel_logits_foldsafe.npz")
    ens = np.mean([0.5 * (zl[f"single{k}"].astype(np.float64) + zl[f"merged{k}"].astype(np.float64)) for k in range(5)], axis=0)
    cached = {str(k): ens[i] for i, k in enumerate(zl["keys"])}
    if "index" in args:
        index_root = doc_root = Path(args["index"])
    else:
        index_root, doc_root = EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior"
    p = {}
    for k, v in args.items():                         # any pipeline parameter: --name=value, cast to the default's type
        if k in c1_pipeline.PARAMS and not isinstance(c1_pipeline.PARAMS[k], (tuple, list)):
            p[k] = type(c1_pipeline.PARAMS[k])(float(v)) if isinstance(c1_pipeline.PARAMS[k], (int, float)) else v
    for name, extra, hide in (("class 2 (answer's spectra hidden)", {}, {m: {std[m]} for m in std}),
                              ("class-1 view (queries' own library left out)", {"exclude_libs": ("enveda-np-examples",)}, None)):
        if args.get("only") == "c2" and hide is None or args.get("only") == "c1" and hide is not None:
            continue
        out = art / "c1_panel_sub.csv"
        rep = c1_pipeline.run(test_path, DATA / "train.parquet", index_root, EXT / "foldsafe",
                              EXT / "v4g" / "casmi26-v4b-models" / "code" / "casmi", out, None, dict(p, **extra),
                              hide_lib=hide, logits_fn=cached.get, doc_root=doc_root)
        sub = pd.read_csv(out, dtype=str, keep_default_na=False)
        print(f"{name}: MRR {score(sol, sub):.4f} | " + str({k: rep.get(k) for k in ("with_spectra", "pubchem_lists", "library_molecules", "library_hits",
                                                                                    "errors", "error_samples", "stopped_early", "pubchem_seconds", "seconds")}), flush=True)
