# Run log

| Date | What | NP-Bench (C1 / C2 / C3) | Public LB | Notes |
|---|---|---|---|---|
| 2026-11-05 | Repo foundation: official metric replica, adducts, loaders, submission writer | — | — | 5 tests pass |
| 2026-11-11 | Rewrote commit author to the GitHub noreply identity; pinned RDKit 2026.03.3 locally; Kaggle CLI routed via tools/kaggle_www.py | — | — | 7 tests pass on 2026.03.3 |
| 2026-11-11 | Fork A submitted: casmi26-fork-ozertuu-v21 v1 (claims 0.500+), submission 58253021 | — | pending | original pinned image; fast commit run clean |
| 2026-11-11 | Fork B submitted: casmi26-fork-alexchilton-0487 v1 (control, claims 0.487), submission 58253022 | — | pending | original pinned image; fast commit run clean |
| 2026-11-11 | Panel benchmark harness: stack order on np-examples | 0.955 / 0.971 / 0.361 | — | own spectra hidden / all references hidden / answer removed from pool; 99/250 answers still recovered by analogue generation |
| 2026-11-12 | Fork A (58253021) scored nothing: "unhandled error" in the hidden rerun | — | error | two upstream bugs: its pinned image `gcr.io/kaggle-images/python@sha256:0717…` has no GPU driver (the pipeline's preflight calls nvidia-smi and asserts), and its model swap never writes `config.json`; the author's latest version has both |
| 2026-11-12 | Fork A patched as `casmi26-a2-ozertuu-fixed`: Fork B's model-swap cell and Fork B's image, `COMMIT_MODE='smoke'` | — | smoke run | smoke commits run every stage on 12 visible molecules before a submission is spent |
| 2026-11-12 | Panel harness (`casmi/panel.py`), PubChem channel, fold-safe FPNet | fold-safe 0.208, +0.4·doc 0.826, doc alone 0.773 | — | leaky A+B FPNet 0.406: leakage doubles the spectral score |
| 2026-11-12 | Molecular networking (`casmi/network.py`, `experiments/network_panel.py`): modified-cosine graph over the panel's own spectra | 0.826 → 0.834–0.839 (all settings up) | — | 32 % of molecules have a neighbour at cosine ≥ 0.8; as a rank-only top-25 overlay +0.006–0.011 with more churn |
| 2026-11-12 | Smoke run of `casmi26-a2-ozertuu-fixed` v1: every stage passed on 12 molecules in 28 min (pipeline 20 min, neural ensemble, outsiders, consensus, physical, v8 mass + RR8, doc-prior merge, apex extlib) | — | ok | — |
| 2026-11-12 | Submitted `casmi26-a2-ozertuu-fixed` v1, submission 58257995 | — | pending | A/B base for the networking variant |
| 2026-11-12 | Neighbour-edit generation, class-3 simulation (answer removed from its own list; edits applied to the neighbour's top 3 from the fold-safe order) | answer regenerated for 29-30/250 molecules, ~2 products each | — | up to +0.10 MRR when every molecule is class 3 |
| 2026-11-12 | Same products inserted without a gate | net -0.008 to -0.056 at 25 % class 3 | — | they push correct answers down in class-2 lists; needs a gate that scores products with the fold-safe FPNet (fingerprint = `chem.raw_fingerprint` in the v4 code) |
