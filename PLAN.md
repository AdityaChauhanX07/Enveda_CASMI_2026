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

**Next.** Goal: top ranks (0.487 is ~rank 948; gold line ~0.503, top 10 0.510-0.549). Public frontier 0.495; the common public 0.492 = Fork B + the in-test network rerank N-1c at LAM 0.35. A public 59-arm study found no single knob above +0.006: the gap needs a new signal. Ours: **DreaMS analogue support**, now with (i) own-spectrum exclusion (a class-2 answer has no public spectra; wrong library compounds matched their own spectra), exempting the query's spectral-library matches (class 1), and (ii) a damped within-list z (eps 0.01, lambda 1.5). Panel gains at lambda 1.5: documentation-free class 2 +0.05-0.06, documentation-misleading +0.074, documentation-heavy +0.025, class 1 +0.008 / -0.012 (answer visible / hidden in the references); MassSpecGym references only (the train-library references, 96k spectra, are neutral on the panel). Nov 18: **C1p** (compliant, CPU, commit run done) and **N2D** (Fork B + N-1c 0.35 + overlay, GPU) - N2D only if the GPU quota left after the three Nov 17 fork reruns covers a ~4.5 h rerun without starving the account's other work; otherwise after the Sat Nov 21 reset. F6 / F2 / F7 tell whether documentation helps the fork on hidden answers (also decides whether C1 should bring documentation back). Later: stack heliosli's public overlays on the 0.492 base; newer public library releases for class 1 (host-allowed); keep a guarded class-3 option for the finals (the private split may hold more class 3). Finals (user, Nov 17: 'whatever it takes to win'): #1 = best fork variant whose gain is consistent across related variants, #2 = a second fork bet (e.g. + guarded class-3 generator); the compliant C1 track stays only as a free CPU testbed for signals. Compute (user, Nov 17): full Kaggle GPU quota from the Sat Nov 21 reset (~6 fork reruns / week); Colab, Azure for Students (~$96) and the user's RTX 4060 laptop for offline work (not attached yet).

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

**2026-11-19 — S2, a stronger second scorer for the fork (the user's request).** Data on Kaggle CPU (free):
`casmi26-s2-prep` (kit cache + novelty holdout + panel lists) and `casmi26-s2-pools-0..3` (PubChem isomer pools).
Then `casmi26-s2-train`: a CPU smoke run, then one GPU run (~1-1.75 h: kit recipe vs kit + PubChem negatives on the
same split, refit of the better one). Adopt only if the novelty-holdout PubChem MRR and the panel's fused MRR both
beat the kit-recipe baseline; LB test as N2 with `--ours=adityachauhanx/casmi26-s2-train` after N2 itself is scored.

**Result (GPU run, 1.14 h):** at 64 epochs each, PubChem negatives raise the novelty-holdout PubChem MRR 0.660 ->
**0.697** (top-1 0.545 -> 0.590; ahead at every evaluation), panel model alone 0.132 -> 0.140, panel fused 0.286 ->
0.284 (within the 0.01 guard); the kit's own diagnostic (training-structure decoys) drops 0.515 -> 0.479. Refit:
kit + PubChem, 80 epochs, all structures -> `casmi26-s2-train` `s2_model/`. Next: C1y (C1u + S2) vs C1z (C1u + v80)
vs C1x (C1u + public 16-epoch model) on the LB (free); if S2 >= v80 there, N2-S2 (`casmi26-n2s-net035-s2`, built:
N2's code with S2 in the second-scorer slot) after N2 next week.

**LB (2026-11-19):** C1x 0.336, C1z 0.334, **C1y 0.352** against their base C1u 0.347: the kit-style scorers hurt C1
(-0.011 / -0.013) and S2 helps (+0.005; +0.018 over v80). First LB-positive idea of ours since C1j.

**2026-11-19 — the public frontier moved to 0.495 / 0.496** (303 teams >= 0.495, 20 >= 0.507; we are at 0.487).
taeyangg4's 0.495 is our own 0.487 base + three cells (N-1c D-L1 network, FR 3-25 learned fragment re-ranker, P5 tail)
+ a GNPS extlib rebuild; heliosli's 0.496 is the 0.477 base + their isomer reranker, joint ranker with RT, M5 isomer
judgement and FRIGID de novo candidates (~8.5 h a run). N2 (0.492) is superseded. GPU this week: 30.73 of 30 h used.
Order after the Sat Nov 21 reset (each needs the user's OK; ~6-7 GPU h a scoring run):
1. **T1** `casmi26-t1-tae495` = taeyangg4's 0.495 verbatim on our account (built): the frontier + the control.
2. **T2** `casmi26-t2-tae495-s2` = T1 with S2 in place of v80 (built; notebook byte-identical to T1).
3. S2 copies (seed 2, ...) for an all-S2 neural blend (T3), trained on GPU (~27 min each) or on a TPU VM's host CPU
   if the smoke run shows it is fast enough (TPU quota is separate: 16.7 of 20 h left this week).
4. Later: heliosli's 0.496 as a second branch for the finals; porting its layers onto T1.

Earlier (2026-11-12):

1. Pull alexchilton `casmi26-lb-0-480-neural-ensemble` (`kaggle kernels pull -m`), re-id to own account,
   keep docker image + inputs, push private, commit run, submit; record hidden-run time.
2. Download `casmi26-natural-product-panel-benchmark`, `casmi26-fold-safe-fpnet` and the identity-disjoint
   folds; reproduce the panel numbers locally; start NP-Bench builder (`casmi/bench.py`).
3. Library-only probe submission to measure the class-1 share of the public split.
