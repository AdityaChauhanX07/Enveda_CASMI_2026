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
tools/kaggle_www.py   Kaggle CLI routed through www.kaggle.com (when api.kaggle.com is blocked)
tests/                metric regression tests (official examples)
```

Data lives outside the repo (`../casmi_data/raw/`: train.parquet, test.parquet,
sample_submission.csv) and is never committed.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q
```
