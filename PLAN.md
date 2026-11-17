# CASMI 2026 — facts, landscape, strategy, schedule

Last updated: 2026-11-12. Final deadline 2026-12-14 23:59 UTC (entry/merge 2026-12-07).

## 0. Status (2026-11-12, 08:30 UTC)

**Leaderboard.** #1 0.540, #2 0.539, #3 0.534, #4 0.527, #5 0.510, #10 ~0.503, #20 0.497. Public notebooks top out at
~0.500. Ours: Fork B 0.487 (58253022), fixed Fork A 0.486 (58257995), fixed Fork A + networking 0.486 (58258520).

**Budget.** Scoring reruns count against the weekly 30 GPU h (API `GetAcceleratorQuotaStatistics`, reset Saturday
00:00 UTC; 26 h used this week). A rerun takes ~4.5 h, a smoke commit ~0.5-1 h, so ~5-6 submissions per week. The LB
(~125 molecules, 31 % of the test) cannot resolve gains below ~0.005: it is a crash check and a coarse sanity check,
never a tuning tool. Only 2 batch-GPU sessions run at once. CPU-only Kaggle jobs are free and used for model runs
on the panel (`casmi26-fs-panel-logits`, `casmi26-fs-panel-spectra`, `casmi26-rel-cpu-check`).

**Rules.** Host (thread 752297): automated, unsupervised processing across test spectra at inference is allowed
(networking, neighbour edits). Open NC data allowed, NIST not. Class shares stay hidden; class 3 may be concentrated
in the private 69 % (speculation, thread 755039), so class-3 gains matter more than the public LB shows.

**Panel findings (250 np-examples, fold-safe models only, `docs/log.md` has every number).**
- Misses are close calls between same-formula isomers (61 of 65); formula or mass-error priors add nothing.
- Five fold-safe FPNets averaged: 0.243 alone / 0.840-0.845 with documentation vs 0.208 / 0.826 for fold 0.
  Shipping them inside the fork's PubChem channel is pointless (A+B logits are ~11x larger); used in our overlay.
- Second-opinion re-ranking of fork-like lists: <= +0.003. Spectrum weighting / pooling variants: no gain.
- Networking overlay: +0.006-0.011 on the panel, LB-neutral (0.486 vs 0.486). Kept.
- Neighbour-edit generator, products scored z + 0.2 x source doc + 0.5 x log(routes): +0.018 / +0.009 at 25 % class 3
  (class-2 -0.001), twice the gain of the first version (z + 3).
- PubChem-relative generator (new): 99 % of panel answers have a one-edit relative in PubChem. Per edit window
  (13 edits each way) the 150 best-documented rows, ECFP4 pass, 40 fully scored, top 3 relatives; products scored
  z + 0.15 x relative doc + 0.5 x log(routes): 138-141/250 answers rebuilt, class-2 lists -0.0002, class-3 lists
  +0.19 (cache grid, `experiments/relatives_cache.py`). Undocumented class-2 answers: net +0.048 (rebuilt from
  documented relatives). In `casmi/edit_overlay.py` (`relatives=1`), pooled over 4 processes with an in-process
  fallback and a deadline, weakest lists first, 40 min budget (~150-200 of 400 molecules on Kaggle's vCPUs).
- **Both generators end to end** (`edit_overlay_panel.py --relatives=1`): **+0.039 / +0.042 at 25 % class 3**
  (class-3 lists 0.175 / 0.186, class-2 lists -0.005); break-even class-3 share ~3 %.

**Notebooks.** a6 = Fork A + networking + both generators with the a6 setting (20 ECFP4 + 20 best-documented rows
fully scored per window; panel net +0.0182 / +0.0549 at 10 / 25 % class 3, class-3 lists 0.238, class-2 -0.0062) and
b6 = Fork B + the same; hardened against odd inputs (per-molecule isolation, edge-case harness clean locally and on
Kaggle with the real models). Relative search budget 60 min in the scoring run (~75 % coverage). a7 / b7 = a6 / b6 +
four relatives kept per window (panel: class-3 lists 0.258, net +0.0198 / +0.0596 at 10 / 25 % class 3, same time).
**a7 scored 0.412** (Fork A + networking 0.486): the edit overlay inserts generated products above correct leaders
that are less documented than the panel's answers (`experiments/lowdoc_leader.py`: doc <= 6 reproduces the -0.074).
Edit overlay dropped; b7 and a8 not run. **a8n** = Fork A + networking + library-consistency overlay (no added
candidates, no documentation term; panel class-2 lists +0.030, 0.8 % of class-1 answers flagged) is the next A/B
against a3. One CASMI GPU job at a time (another competition on the account needs the second slot).

**Next.** Goal: top ranks (0.487 is ~rank 948; top 10 0.510-0.549). Probe 0.141: class 1 (~13 % of the public split, found straight from train.parquet) is not the gap; class-2 ranking is (panel: 49 % of class-2 answers sit at ranks 2-25 behind same-formula natural-product isomers). Nov 17 on the LB: F6 (fork without documentation, natural-product-aware), F2 (without documentation, natural-product bonus off too), F7 (fork + spec2mol), C1j (compliant). Pre-registered Nov 18 rules (public LB noise ~+-0.013): (a) F6 >= 0.495 -> B6 (same patch on the 0.487 Fork B base) and F8 (F6 + spec2mol); (b) F7 >= 0.495 -> B8 / F5 (spectral on the documentation-free base); (c) F6 and F7 both within +-0.01 of 0.486 -> neither lever moves the fork; next: consensus leader weight (w_top) and the promotion gate (F4); (d) F2 > F6 by > 0.01 -> the PubChem natural-product bonus hurts; drop it. Finals: #1 = best fork variant whose gain is consistent across related variants (not a single lucky score), #2 = best compliant (C1d 0.320 / C1j). Compliant track keeps validating in the real pipeline (fold-model logit combinations, larger shortlists).

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
| 0 | Nov 11-13 | On the board: Fork B score, patched Fork A (smoke-tested) | a fork reproduces ~0.49-0.50 |
| 1 | Nov 12-18 | Networking overlay on the best fork (first LB A/B), neighbour-edit generator, re-ranker prototype on leak-free panel features | each beats the base on the panel |
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
