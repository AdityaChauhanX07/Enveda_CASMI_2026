# CASMI 2026 — facts, landscape, strategy, schedule

Last updated: 2026-11-11. Final deadline 2026-12-14 23:59 UTC (entry/merge 2026-12-07).

## 0. Status (2026-11-11)

- Leaderboard: #1 0.540, #2 0.539, #3 0.534, #10 0.503, #20 0.495; 5,152 teams. Public notebooks now 0.487-0.500.
- First two submissions in flight: private forks `adityachauhanx/casmi26-fork-ozertuu-v21` (claims 0.500+, all
  inputs public) and `adityachauhanx/casmi26-fork-alexchilton-0487` (control). heliosli's 0.488 hides 6 inputs.
- Local: RDKit 2026.03.3 installed (matches the scorer); Kaggle CLI works via `tools/kaggle_www.py`.
- Panel benchmark downloaded to `../casmi_data/ext/panel`. Caveat: 249/250 answers are well-documented COCONUT
  natural products and the ranking models saw them, so the documentation prior looks far stronger there
  (0.51 -> 0.95) than it can be on hidden class-2 molecules. Correct for this before trusting any re-ranker.

## 1. Verified facts

**Rules (Kaggle API + rules tab).** Featured code competition, notebook-only submissions, internet off,
CPU or GPU notebook <= 9 h, output `/kaggle/working/submission.csv`. 5 submissions/day, 2 finals,
max team 5, $50k (1st $16k, top 5 paid). External data and pretrained models allowed if free and
public. Data license CC BY-NC 4.0. Winners must release code (MIT listed). No private sharing,
no hand-labelling. Host allows newer releases of train.parquet's source libraries (discussion #750684, Oct 28).

**Metric (official notebook `metric/casmi-mean-reciprocal-rank`, replicated in `casmi/metric.py`).**
MRR@25. Key = RDKit 2026.03.3: RemoveStereochemistry -> TautomerEnumerator.Canonicalize -> InChIKey[:14].
Duplicates and unparseable SMILES still consume a rank position -> dedupe on the key before writing.
>= 1 guess per molecule, <= 25. Answer cell may list several acceptable isomers (comma-separated).
Install RDKit from the official wheel dataset `metric/rdkit-2026-3-3-wheel` (not in the Kaggle image).

**Data.** train.parquet: 2,539,608 spectra, 275,810 structures, 11 libraries, 121 adduct strings.
Visible test.parquet is a dummy (400 molecules, 1,213 spectra copied from enveda-180); the hidden set
(~400 molecules, ~1,500 spectra, 157–1,159 Da, 10 adducts) replaces it at scoring time.

- enveda-np-examples (the test-like set): 1,184 spectra, 250 structures, all 10 test adducts
  ([M+H]+ 569, [M-H]- 276, [M+CH2O2-H]- 105, [M+NH4]+ 98, [M+Na]+ 80, [M-H2O+H]+ 24, [M+K]+ 16,
  [M-2H2O+H]+ 8, [M+Cl]- 7, [M-H2O-H]- 1); CE 20/35/40/60/80 and merged [20,50], [20,40,60], [20,40,60,80];
  precursor m/z 132–1154 (median 388); all 250 structures also have spectra in other libraries.
- timsTOF precursor |ppm| error: np-examples p50 1.65, p90 3.19, p99 6.62, p99.9 11.0, max 21.1;
  enveda-180 p50 0.98, p99 4.4, max 7.1.
- Bruker-like spectra outside Enveda libraries: only 8,125 spectra / 3,808 structures (MoNA, RIKEN, GNPS).
- Dummy test noise: median 230 peaks/spectrum (max 3,259); half of all peaks < 0.02 % of base peak.

## 2. Landscape (2026-11-05)

Leaderboard: #1 0.536, #2 0.521, #3 0.519, #5 0.504, #10 0.490, #20 0.486; 4,775 teams.
Public LB ~ 1/3 of molecules (~125–133); private ~ 2/3.
Best public notebooks: heliosli `casmi26-0-482` (0.482; two inputs look private),
alexchilton `casmi26-lb-0-480-neural-ensemble` (0.480; all inputs public -> fork target),
zz3t3rn41 `casmi26-peaks-to-structures-v2`, flexonafft `casmi26-closing-the-evidence-gap`.

All are one shared stack:
- Candidates: train + COCONUT (0.71 M) within 10 ppm + analogue generation (engine 1);
  ChEBI, LIPID MAPS, PubChemLite (engine 2); PubChem tier of 105.9 M screened by fingerprint models.
- Scorers: FPNets, LightGBM rankers, ICEBERG + GLACIER forward models, MetFrag-lite, neural
  spectrum<->structure model (sanpier), same-formula isomer reranker, PubChem documentation prior,
  newer library releases (MSnLib 2026-07, GNPS 2026-10, MassBank 2026.03).
- Final order: hand-set RRF weights / promotion thresholds, many tuned on the public split (fragile).
- Runtime: time guard at 7 h -> little headroom inside the 9 h limit.

Public datasets they use (all reusable): `dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets`
(20 GB, whole stack), `casmi26-pubchem-popularity-prior` (80 GB), `casmi26-natural-product-knowledge-table`,
`casmi26-natural-product-panel-benchmark` (250 np-examples with every candidate + score),
`casmi26-fold-safe-fpnet` (5 identity-disjoint folds, models never saw their fold or np-examples),
`sanpier/casmi-spectrum-model-v1`, `sanpier/catalogue-experiment-saner`,
`qwertykeypadapple/casmi26-lotus-candidates`, `hengck23/hengck23-dreams-enveda-casmi26` (DreaMS),
`metric/rdkit-2026-3-3-wheel`.

Lessons from competitors' public logs:
- Forks that drop the author's pinned Docker image silently lose ~0.04.
- Validation on random enveda-180 structures ranked ideas in the wrong order; np-examples under class masks
  predicted the LB. Leakage is huge: FPNet top-1 97 % on seen vs 68 % on unseen molecules.
- Ranker training rows (how test-like) mattered more than a better fingerprint model.
- Candidate pruning silently dropped true answers; a fragment channel was dead above 34 bonds.
- Library-only pipelines: ~0.93 on class-1 validation but 0.12–0.15 on the LB -> class 1 is ~15–25 % of test.

## 3. Strategy

Borrow the engines, own the measurement and the final decision, add evidence nobody public has.
1. Learned final ranker (LambdaMART) trained on leak-free, test-like simulations, replacing LB-tuned fusion.
2. New evidence: more public libraries, test-set networking, class-3 generator.
3. Better isomer discrimination only where the benchmark shows it is the bottleneck.

## 4. Schedule (revised 2026-11-11, compressed after a 6-day slip)

| Phase | Dates | Goal | Gate |
|---|---|---|---|
| 0 | Nov 11-12 | On the board: two public forks (pinned images), record hidden-run time | fork reproduces ~0.49-0.50 |
| 1 | Nov 12-18 | Re-ranker prototype on the panel + audit of the extra libraries already in the stack | beats the stack's own order on the panel (doc-prior bias corrected) |
| 2 | Nov 19-28 | Differentiators: test-set networking, class-3 generator, leak-free re-ranker rows | each shows a gain on the benchmark |
| 3 | Nov 29-Dec 9 | One hardened notebook, stage time guards, fallback chain; <= 3 LB checks/day | <= 7.5 h, no failed reruns |
| 4 | Dec 10-14 | Freeze Dec 10, clean-fork rerun, pick 2 finals | best validated + one higher-upside variant |

Additions are overlays on the forked stack's final candidate lists (as the public overlays do), so the stack's
runtime and candidate set stay intact and every overlay can fall back to the stack's own answer.

**NP-Bench (phase 1).** Queries: 250 np-examples (fold-safe models only) + newly released public spectra of
structures absent from train.parquet (leak-free for every public model, natural class-2 behaviour).
Masks: C1 (hide own spectra), C1-sparse (keep 1–2 references from other instruments), C2 (hide all
spectra, keep in pools), C2-low-doc (poorly documented structures; np-examples inflate the documentation
prior: 0.51 -> 0.95), C3 (remove from every pool incl. PubChem). Run expensive engines once per
query x mask, cache per-candidate evidence, iterate rankers on CPU in seconds.

**Ranker features (phase 2).** Engine scores/ranks, FP likelihoods (several models), neural scores,
ICEBERG/GLACIER similarity, fragment explanation, library similarity per adduct, timsTOF mass-error
likelihood, formula plausibility, documentation/NP priors, within-list margins, best-library-hit strength
(learned class gating). Grouped CV by structure, class-mixture rows.

**New evidence (phase 3, by gain per effort).**
1. External libraries: newest MSnLib, GNPS, MassBank, MoNA, MS-DIAL, RIKEN; dedupe vs train; match
   thresholds from timsTOF false-hit rates. CPU.
2. Test-set networking: compare the ~400 test molecules with each other, propagate confident IDs to
   relatives (aglycone + sugar, +/- CH2, +/- O ...), pseudo-labels within the run. CPU.
3. Class-3 generator: biotransformation edits of top candidates to hit the exact formula (hexose,
   deoxyhexose, pentose, glucuronide, acetyl, malonyl, coumaroyl, caffeoyl, feruloyl, galloyl, methyl,
   hydroxyl, prenyl, sulfate, +/- H2), pruned by fragment consistency, gated by a class-3 detector.
4. Formula inference from fragment subformulas with the timsTOF error model (ranker features).
5. Own neural scorers (contrastive, isomer hard negatives, fold-safe) only if NP-Bench shows headroom.

Parked: full de novo generators (DiffMS/MSNovelist-style, ~0–2 % exact match); retraining ICEBERG on timsTOF (stretch).

## 5. Compute (no budget)

Kaggle CPU sessions (no weekly quota, 12 h, several in parallel) for bulk evidence generation;
Kaggle GPU 30 h/week and TPU 20 h/week (confirmed on the account) for submissions and training;
any dev box for ranker training. Big artifacts go to private Kaggle datasets, code to this repo.

## 6. Operating rules

- Every change gets an NP-Bench number per class before any LB submission; one idea per submission.
- Log every run (date, config, NP-Bench, LB) in `docs/log.md`.
- Pin Docker image + RDKit 2026.03.3; dedupe with `casmi.metric.ik14`; always 1–25 guesses; fallbacks.
- Choose finals by NP-Bench + LB agreement, never by public LB alone.

## 7. Immediate next steps

1. Pull alexchilton `casmi26-lb-0-480-neural-ensemble` (`kaggle kernels pull -m`), re-id to own account,
   keep docker image + inputs, push private, commit run, submit; record hidden-run time.
2. Download `casmi26-natural-product-panel-benchmark`, `casmi26-fold-safe-fpnet` and the identity-disjoint
   folds; reproduce the panel numbers locally; start NP-Bench builder (`casmi/bench.py`).
3. Library-only probe submission to measure the class-1 share of the public split.
