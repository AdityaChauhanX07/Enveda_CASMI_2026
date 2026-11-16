"""Build the compliant submission notebook (C1): CPU only, inputs with public generation code only.

Inputs: the competition data; our PubChem index (Kaggle kernel output of casmi26-pubchem-index-build, built from
PubChem's primary files by tools/build_pubchem_index.py); the five fold-safe FPNets (dmitriigluzdov/
casmi26-fold-safe-fpnet, training code public); the host's RDKit wheel. The FPNet architecture / inference code
(Ahmed Berat Özer, Apache-2.0) and our modules are embedded in the notebook.
Cells: setup -> C1 lists (casmi/c1_pipeline.py) -> networking -> library consistency -> guarded edit overlay.

usage: python tools/build_c1_notebook.py OUT_DIR [--support-gate | --support-rerank] [--neural]
(--neural: separate kernel casmi26-c1n-compliant-neural with sanpier's spectrum model fused at rank level)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import add_overlay  # noqa: E402

V4 = ROOT.parents[0] / "casmi_data" / "ext" / "v4g" / "casmi26-v4b-models" / "code" / "casmi"

INTRO = """# CASMI 2026: compliant pipeline (C1)

Every input of this notebook can be regenerated from public code and permitted data:

- **Candidates:** a mass-sorted PubChem index built from PubChem's primary FTP files (CID-Mass, CID-InChI-Key,
  CID-SMILES, CID-SID, CID-PMID) by a public build script; documentation counts from the same files.
- **Scoring:** the five fold-safe FPNets of `dmitriigluzdov/casmi26-fold-safe-fpnet` (training code in the
  dataset) on the fingerprint bits they were trained for, plus a documentation prior.
- **Library hits (class 1):** spectra of the competition's `train.parquet`.
- **Overlays:** test-set networking, library consistency (candidates whose own reference spectra contradict the
  query move down) and a guarded edit generator (never at rank 1, one candidate per list).

Credits: Enveda and the competition hosts for the data; the U.S. National Library of Medicine for PubChem
(Kim et al., PubChem 2025 update); Ahmed Berat Özer for the FPNet architecture and inference code (Apache-2.0);
Dmitrii Gluzdov for the fold-safe FPNet training kit and checkpoints; the RDKit developers.
"""

SETUP = '''# ── Setup: paths, RDKit 2026.03.3, embedded code ──
import os, sys, json, time, glob, subprocess, shutil, traceback
from pathlib import Path
import numpy as np, pandas as pd
NB_T0 = time.time()
INPUT = Path('/kaggle/input')
COMP = next(p for p in (INPUT / 'competitions/enveda-CASMI26-molecule-id-mass-spectra', INPUT / 'enveda-CASMI26-molecule-id-mass-spectra')
            if (p / 'test.parquet').is_file())
IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
COMMIT_MODE, PIPELINE_DONE = 'full', True        # commit runs rank every visible molecule (CPU only)
SAMPLE_IDS = pd.read_csv(COMP / 'sample_submission.csv', dtype=str).molecule_id.tolist()
_tag = f"cp{{sys.version_info.major}}{{sys.version_info.minor}}"
_whl = [w for w in glob.glob(str(INPUT / '**/rdkit-*.whl'), recursive=True) if _tag in w and '2026.3.3' in w]
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q', _whl[0]], check=True)
W = Path('/kaggle/working/c1code')
for _name, _src in {modules!r}.items():
    (W / _name).parent.mkdir(parents=True, exist_ok=True); (W / _name).write_text(_src)
sys.path.insert(0, str(W))
import importlib
edit_overlay = importlib.import_module('edit_overlay')
c1_pipeline = importlib.import_module('c1_pipeline')
INDEX = Path(sorted(glob.glob(str(INPUT / '**/pubchem_index/pc_mass.npy'), recursive=True), key=len)[0]).parent
FOLDSAFE = Path(sorted(glob.glob(str(INPUT / '**/fpnet_fold0.pt'), recursive=True), key=len)[0]).parent
EDIT_KW = dict(v4_code=str(W / 'v4'), fold_model=str(FOLDSAFE / 'fpnet_fold0.pt'), tier_root=str(INDEX))
print('C1 SETUP:', json.dumps(dict(comp=str(COMP), index=str(INDEX), foldsafe=str(FOLDSAFE), rerun=IS_RERUN,
                                   index_report=json.loads((INDEX / 'BUILD_REPORT.json').read_text()).get('rows'))))
'''

RUN = '''# ── C1 candidate lists: own PubChem index + fold-safe FPNets + documentation, library hits first ──
_left = 8.5 * 3600 - (time.time() - NB_T0)
try:
    _nb = sorted(glob.glob(str(INPUT / '**/casmi_spectrum_model_v1/inference.py'), recursive=True), key=len) if {neural!r} else []
    _rep = c1_pipeline.run(COMP / 'test.parquet', COMP / 'train.parquet', INDEX, FOLDSAFE, W / 'v4',
                           '/kaggle/working/submission.csv', '/kaggle/working/c1_report.json',
                           dict(device='cpu', workers=4, max_seconds=max(600, min(4 * 3600, _left - 3 * 3600)),
                                neural_bundle=str(Path(_nb[0]).parent) if _nb else ''))
    _f = pd.read_csv('/kaggle/working/submission.csv', dtype=str, keep_default_na=False)
    assert _f.molecule_id.tolist() == SAMPLE_IDS and _f.smiles.str.split(';').map(len).between(1, 25).all()
    print('C1:', json.dumps({{k: v for k, v in _rep.items() if k != 'params'}}))
except Exception:
    traceback.print_exc()
    pd.DataFrame({{'molecule_id': SAMPLE_IDS, 'smiles': ['CCO'] * len(SAMPLE_IDS)}}).to_csv('/kaggle/working/submission.csv', index=False)
    print('C1: failed, placeholder answers written')
'''


def cell(kind: str, text: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build(out_dir: Path, edit_params: dict, neural: bool = False) -> Path:
    modules = {f"v4/{n}.py": (V4 / f"{n}.py").read_text() for n in ("chem", "spectra", "fpnet")}
    for n in ("edit_overlay", "c1_pipeline"):
        modules[f"{n}.py"] = (ROOT / "casmi" / f"{n}.py").read_text()
    cells = [cell("markdown", INTRO), cell("code", SETUP.format(modules=modules)), cell("code", RUN.format(neural=neural))]
    cells.append(cell("code", add_overlay.CELL.format(src=(ROOT / "casmi" / "network_overlay.py").read_text(), params={"lam": 4.0})))
    cells.append(cell("code", add_overlay.LIB_CELL.format(src=(ROOT / "casmi" / "library_overlay.py").read_text(),
                                                          params=dict(tau=0.5, down=3, min_ref=3))))
    cells.append(cell("code", add_overlay.EDIT_CELL.format(src=(ROOT / "casmi" / "edit_overlay.py").read_text(), params=edit_params)))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = "casmi26-c1n-compliant-neural" if neural else "casmi26-c1-compliant"
    path = out_dir / f"{slug}.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = {"id": f"adityachauhanx/{slug}", "title": slug.replace("-", " "), "code_file": path.name,
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["dmitriigluzdov/casmi26-fold-safe-fpnet", "metric/rdkit-2026-3-3-wheel"]
                                                         + (["sanpier/casmi-spectrum-model-v1"] if neural else []),
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
            "kernel_sources": ["adityachauhanx/casmi26-pubchem-index-build"], "model_sources": []}
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return path


if __name__ == "__main__":
    params = dict(relatives=1, max_seconds=4800, rel_seconds=3600, rel_n_full=20, rel_n_docrank=20, rel_k=4, min_slot=1,
                  max_insert=1, device="cpu")
    if "--support-gate" in sys.argv:
        params.update(sup_gate=1, sup_sigma=0.6)
    if "--support-rerank" in sys.argv:
        params.update(sup_beta=1.0, sup_sigma=0.7)
    print("written", build(Path(sys.argv[1]), params, neural="--neural" in sys.argv))
