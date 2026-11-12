# Enveda CASMI 2026 — Molecule ID from Mass Spectra

Solo entry for the Kaggle competition
[Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra):
identify the 2D structure of natural-product-like molecules from Bruker timsTOF MS/MS spectra.
Each molecule gets up to 25 ranked SMILES; the score is MRR@25 on the tautomer-canonical
InChIKey14 (stereo stripped first).

See [PLAN.md](PLAN.md) for the facts, the competitive landscape, the strategy and the schedule.

## Layout

```
casmi/metric.py       exact copy of the official canonicalization + MRR@25 scorer
casmi/adducts.py      adduct table and neutral-mass computation
casmi/data.py         parquet loaders (metadata-only, np-examples, peaks)
casmi/submission.py   dedupe by scorer key, fill to 25, validate the CSV
casmi/spectra.py      spectrum cleaning, spectral entropy similarity
casmi/panel.py        CPU harness on the 250-molecule natural-product panel (fold-safe scores, MRR helpers)
casmi/network.py      modified cosine and the spectral graph between test molecules
casmi/network_overlay.py  re-orders a finished submission by its spectral neighbours' likely structures
casmi/edit_overlay.py     class-3 generator for a finished submission: one-edit products of spectral neighbours'
                          candidates and of PubChem relatives, scored by the five fold-safe FPNets, source
                          documentation and route consensus, inserted into the top 3
casmi/fpscore.py      the FPNet's selected fingerprint bits for any structure
tools/add_overlay.py  appends the networking and edit cells to a forked notebook
tools/kaggle_www.py   Kaggle CLI routed through www.kaggle.com (when api.kaggle.com is blocked)
experiments/          panel experiments behind every number in docs/log.md
tests/                metric regression tests (official examples) and network tests
```

Results of every experiment and submission are in [docs/log.md](docs/log.md).

Data lives outside the repo (`../casmi_data/raw/`: train.parquet, test.parquet,
sample_submission.csv) and is never committed.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q
```
