"""C1's window stage when the answer is poorly documented: are natural-product answers lost to the documentation cuts?

The replays (experiments/np_prior_lowdoc.py, np_flags_compare.py) cap the answer's documentation only after C1's window
stage, whose shortlists were cut with the answer's true (panel: high) documentation: the best-documented max_window rows
of large windows, the n_docpre best-documented rows fingerprinted beside the n_partial best by the ECFP4-block score,
and the 4 x top_k best by score + w_doc x doc. Here the answer's documentation is capped inside the window stage
(c1_pipeline stress_ik / stress_cap), so every cut sees it, and C1 + natural-product bonus is compared with and without
np_window (members never cut, all fully scored, the np_keep best carried into the shortlist).

Panel class 2 (the answer's spectra hidden from the library search); cached five-model logits; shared tier for
candidates (as experiments/c1_panel.py); no spec2mol (its saved scores cover only the earlier shortlists).

usage: c1_window_lowdoc.py NATURAL_PRODUCTS_PARQUET COCONUT_CSV [--caps=none,4,0] [--beta=3] [--w_doc=0.25] [--window_only=1]
       [--coconut_extra=1] [--lib_beta=4] [--frag=LAMBDA:TOL] [--npl=ETA]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT))
from casmi import edit_overlay  # noqa: E402
sys.modules["edit_overlay"] = edit_overlay
from casmi import c1_pipeline  # noqa: E402
from casmi.data import DATA  # noqa: E402
from casmi.metric import score  # noqa: E402

if __name__ == "__main__":
    args = dict(a[2:].split("=", 1) for a in sys.argv[3:] if a.startswith("--") and "=" in a)
    caps = [None if c == "none" else float(c) for c in args.get("caps", "none,4,0").split(",")]
    beta = float(args.get("beta", 3.0))
    w_doc = float(args.get("w_doc", 0.25))
    art = ROOT / "artifacts"
    test_path = art / "np_test.parquet"
    t = pd.read_parquet(test_path)
    truth = t.drop_duplicates("molecule_id").set_index("molecule_id").normalized_smiles
    sol = pd.DataFrame({"molecule_id": truth.index, "smiles": truth.values})
    from rdkit import Chem
    std = {m: Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for m, s in truth.items()}
    zl = np.load(art / "panel_logits_foldsafe.npz")
    ens = np.mean([0.5 * (zl[f"single{k}"].astype(np.float64) + zl[f"merged{k}"].astype(np.float64)) for k in range(5)], axis=0)
    cached = {str(k): ens[i] for i, k in enumerate(zl["keys"])}
    # memoise the slow parts that do not change between runs: metric keys and the library search
    key0, memo = edit_overlay.Chemistry.key, {}
    edit_overlay.Chemistry.key = lambda self, s_: memo[s_] if s_ in memo else memo.setdefault(s_, key0(self, s_))
    hits0, hit_memo = c1_pipeline.library_hits, {}
    c1_pipeline.library_hits = lambda test, train_path, p, hide=None: hit_memo.setdefault("h", hits0(test, train_path, p, hide))
    base = dict(np_table=sys.argv[1], np_coconut=sys.argv[2], np_beta=beta, stress_ik=std, w_doc=w_doc,
                np_coconut_extra=int(args.get("coconut_extra", 0)), lib_beta=float(args.get("lib_beta", 4.0)))
    if "npl" in args:
        base.update(npl_eta=float(args["npl"]))
    if "frag" in args:
        lam, _, tol = args["frag"].partition(":")
        base.update(frag_lambda=float(lam), frag_tol=float(tol or 0.01))
    print(f"class 2, C1 (w_doc {w_doc}) + {beta} x natural-product membership; answer documentation capped inside the window stage")
    variants = (("C1 + bonus", dict(np_window=0)), ("C1 + bonus + NP-aware window", dict(np_window=1)))
    for cap in caps:
        for name, extra in variants[1:] if args.get("window_only") == "1" else variants:
            out = art / "c1_window_lowdoc_sub.csv"
            rep = c1_pipeline.run(test_path, DATA / "train.parquet", EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "foldsafe",
                                  EXT / "v4g" / "casmi26-v4b-models" / "code" / "casmi", out, None, dict(base, stress_cap=cap, **extra),
                                  hide_lib={m: {std[m]} for m in std}, logits_fn=cached.get, doc_root=EXT / "pubchem_prior")
            sub = pd.read_csv(out, dtype=str, keep_default_na=False)
            print(f"  doc cap {str(cap):5s} | {name:28s} | MRR {score(sol, sub):.4f} | "
                  + str({k: rep.get(k) for k in ("np_members", "coconut_candidates", "coconut_error", "frag_sd", "frag_molecules", "frag_error", "frag_seconds", "npl_candidates", "npl_error",
                                                 "errors", "error_samples", "np_error", "pubchem_seconds", "seconds")}), flush=True)
