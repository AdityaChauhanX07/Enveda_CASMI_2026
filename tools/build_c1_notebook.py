"""Build the compliant submission notebook (C1): CPU only, inputs with public generation code only.

Inputs: the competition data; our PubChem index (Kaggle kernel output of casmi26-pubchem-index-build, built from
PubChem's primary files by tools/build_pubchem_index.py); the five fold-safe FPNets (dmitriigluzdov/
casmi26-fold-safe-fpnet, training code public); the host's RDKit wheel. The FPNet architecture / inference code
(Ahmed Berat Özer, Apache-2.0) and our modules are embedded in the notebook.
Cells: setup -> C1 lists (casmi/c1_pipeline.py) -> networking -> library consistency [-> guarded edit overlay with --edit].

usage: python tools/build_c1_notebook.py OUT_DIR [--edit [--support-gate | --support-rerank]] [--neural | --s2m] [--np [--npw]]
(--neural: separate kernel casmi26-c1n-compliant-neural with sanpier's spectrum model fused at rank level;
 --s2m: separate kernel casmi26-c1s-compliant-spec2mol with spec2mol fused into each whole shortlist, gamma S2M_GAMMA;
 --np: natural-product prior NP_BETA for COCONUT / LOTUS / NPAtlas members; with --s2m the kernel is casmi26-c1snp-compliant;
 --npw: also a natural-product-aware window stage (members never lost to documentation cuts): casmi26-c1snpw-compliant;
 --wdoc=W: documentation weight W (default 0.25; 0 with --s2m --np --npw: casmi26-c1d-compliant-docfree); --beta=B: bonus B;
 --coco: COCONUT structures missing from the PubChem windows join them; --libbeta=L: library-hit weight L (default 4);
 --frag=L[:TOL]: fragment explanation at weight L (match tolerance TOL Da, default 0.01); --slug=NAME: kernel name override;
 --library-only: probe kernel casmi26-c1lib-probe - lists of train.parquet library hits (cosine >= 0.5) only, no overlays;
 --san=G: sanpier's spectrum model fused at score level over each whole shortlist at weight G (pooled scale);
 --npl=E: natural-product likeness (RDKit Contrib NP_Score) at weight E;
 --libfirst=T: library hits with cosine >= T lead their lists;
 --set=name=value[,name=value...]: any other c1_pipeline parameter (numbers), e.g. --set=short_n=150,np_keep=150;
 --dreams=L[:k=v,...]: DreaMS analogue-support overlay (MassSpecGym neighbours) at weight L, after the library
           overlay; k=v sets overlay parameters (noself=1, self_lib=0.8, ...; trainrefs=1 adds the train-library references))
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import add_overlay  # noqa: E402

V4 = ROOT.parents[0] / "casmi_data" / "ext" / "v4g" / "casmi26-v4b-models" / "code" / "casmi"
# spec2mol weight over the pooled within-shortlist scale (experiments/spec2mol_lowdoc.py, panel class 2 / class-1 view:
# documentation as is 0.811 / 0.871 vs C1 0.815 / 0.874; answer documentation <= 6: 0.564 / 0.676 vs 0.522 / 0.652)
S2M_GAMMA = 0.75
# natural-product membership bonus, LOTUS / NPAtlas or COCONUT (experiments/np_prior_lowdoc.py, np_flags_compare.py;
# COCONUT members among answers 99.6 %, among decoys in C1's top 25 11 % at doc <= 4, 42 % at 4-6, 91 % above 12).
# C1s base, class 2 at answer documentation as is / <= 6 / <= 4 / 0: none 0.811 / 0.564 / 0.452 / 0.236; 1.0: 0.823 /
# 0.627 / 0.539 / 0.343; 3.0: 0.831 / 0.648 / 0.578 / 0.436 (5.0 saturates). C1 scored 0.238 on the LB, close to the
# panel at answer doc 0: the hidden answers behave as poorly documented, where membership matters most. No
# reference-count term: it lost 0.045 when the answer's LOTUS references were capped at 1
NP_BETA = 3.0

INTRO = """# CASMI 2026: compliant pipeline (C1)

Every input of this notebook can be regenerated from public code and permitted data:

- **Candidates:** a mass-sorted PubChem index built from PubChem's primary FTP files (CID-Mass, CID-InChI-Key,
  CID-SMILES, CID-SID, CID-PMID) by a public build script; documentation counts from the same files.
- **Scoring:** the five fold-safe FPNets of `dmitriigluzdov/casmi26-fold-safe-fpnet` (training code in the
  dataset) on the fingerprint bits they were trained for, plus a documentation prior.
- **Library hits (class 1):** spectra of the competition's `train.parquet`.
{s2m_line}- **Overlays:** test-set networking and library consistency (candidates whose own reference spectra contradict
  the query move down).

Credits: Enveda and the competition hosts for the data; the U.S. National Library of Medicine for PubChem
(Kim et al., PubChem 2025 update); Ahmed Berat Özer for the FPNet architecture and inference code (Apache-2.0);
Dmitrii Gluzdov for the fold-safe FPNet training kit and checkpoints{s2m_credit}; the RDKit developers.
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
    _s2m = sorted(glob.glob(str(INPUT / '**/casmi26-spec2mol/spec2mol.pt'), recursive=True), key=len) if {s2m_gamma!r} else []
    _npt = sorted(glob.glob(str(INPUT / '**/casmi26-natural-product-knowledge-table/natural_products.parquet'), recursive=True), key=len) if {np_beta!r} else []
    _coco = sorted(glob.glob(str(INPUT / '**/coconut_csv_lite*.csv'), recursive=True), key=len) if {np_beta!r} else []
    _rep = c1_pipeline.run(COMP / 'test.parquet', COMP / 'train.parquet', INDEX, FOLDSAFE, W / 'v4',
                           '/kaggle/working/submission.csv', '/kaggle/working/c1_report.json',
                           dict(device='cpu', workers=4, max_seconds=max(600, min(4 * 3600, _left - 3 * 3600)),
                                neural_bundle=str(Path(_nb[0]).parent) if _nb else '',
                                s2m_bundle=str(Path(_s2m[0]).parent) if _s2m else '', s2m_gamma={s2m_gamma!r},
                                np_table=_npt[0] if _npt else '', np_coconut=_coco[0] if _coco else '', np_beta={np_beta!r},
                                np_window={np_window!r}{run_extra}))
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


def build(out_dir: Path, edit_params: dict, neural: bool = False, s2m: bool = False, np_prior: bool = False,
          np_window: bool = False, w_doc: float | None = None, beta: float | None = None, coco: bool = False,
          lib_beta: float | None = None, slug: str | None = None, frag: str | None = None, library_only: bool = False,
          san: float | None = None, npl: float | None = None, lib_first: float | None = None, extra_set: str | None = None,
          dreams: str | None = None) -> Path:
    modules = {f"v4/{n}.py": (V4 / f"{n}.py").read_text() for n in ("chem", "spectra", "fpnet")}
    for n in ("edit_overlay", "c1_pipeline", "fragments"):
        modules[f"{n}.py"] = (ROOT / "casmi" / f"{n}.py").read_text()
    extra = ("- **spec2mol** (`casmi26-spec2mol`, training code `train.py` / `prep.py` in the dataset): spectra and\n"
             "  fingerprints in one embedding space; its score joins each whole shortlist at weight "
             f"{S2M_GAMMA} (pooled scale).\n" if s2m else "")
    nb_beta = NP_BETA if beta is None else beta
    extra += ("- **Natural-product prior:** candidates listed in COCONUT (the COCONUT CSV-lite export of September 2026)\n"
              "  or in LOTUS / NPAtlas (`natural_products.parquet` of `casmi26-natural-product-knowledge-table`, built\n"
              f"  from both by its public `build_np.py`), or matched in the spectral library, gain {nb_beta}"
              + ("; they are also never lost to the documentation-based cuts of the window stage" if np_window else "")
              + ".\n" if np_prior else "")
    if w_doc is not None:
        extra += f"- **Documentation weight:** {w_doc} (default 0.25){' - no documentation prior' if w_doc == 0 else ''}.\n"
    if coco and np_prior:
        extra += "- **COCONUT candidates:** COCONUT structures missing from a PubChem mass window join it (no documentation).\n"
    if lib_beta is not None:
        extra += f"- **Library-hit weight:** {lib_beta} x (cosine - 0.8) (default 4).\n"
    run_extra = "".join(f", {k}={v!r}" for k, v in (("w_doc", w_doc), ("lib_beta", lib_beta)) if v is not None)
    run_extra += ", np_coconut_extra=1" if (coco and np_prior) else ""
    if san:
        run_extra += f", san_gamma={float(san)!r}"
        extra += (f"- **Spectrum model (sanpier/casmi-spectrum-model-v1, training code public):** its score joins each whole "
                  f"shortlist at weight {float(san)} (pooled scale).\n")
    if npl:
        run_extra += f", npl_eta={float(npl)!r}"
        extra += (f"- **Natural-product likeness** (Ertl et al. 2008, RDKit Contrib NP_Score): + {float(npl)} x the "
                  "candidate's score.\n")
    dreams_params, dreams_kernels = add_overlay.parse_dreams(dreams) if dreams else ({}, [])
    if dreams:
        extra += (f"- **DreaMS analogue support** (public DreaMS model; MassSpecGym spectra"
                  f"{' + train.parquet library spectra' if dreams_kernels else ''}): each list is re-ranked by the "
                  f"structures of the query's nearest public spectra, weight {dreams_params['lam']}"
                  f"{' (a candidate never supported by its own spectra unless the query has a library match)' if dreams_params.get('noself') else ''}.\n")
    if lib_first:
        run_extra += f", lib_first={float(lib_first)!r}"
        extra += f"- **Strong library matches first:** hits with cosine >= {float(lib_first)} lead their lists.\n"
    if extra_set:
        for kv in extra_set.split(","):
            k, v = kv.split("=")
            run_extra += f", {k.strip()}={float(v) if '.' in v else int(v)!r}"
    if library_only:
        run_extra += ", library_only=1, lib_hit=0.5"
        extra += "- **Probe:** lists hold only spectral-library hits (cosine >= 0.5, best first); no PubChem stage, no overlays.\n"
    if frag:
        lam, _, tol = frag.partition(":")
        run_extra += f", frag_lambda={float(lam)!r}" + (f", frag_tol={float(tol)!r}" if tol else "")
        extra += (f"- **Fragment explanation:** each candidate's bond-cleavage fragments (one or two cuts, ring openings) are "
                  f"matched to the spectrum's top peaks; the explained share joins the score at weight {float(lam)}.\n")
    credit = " and spec2mol" if s2m else ""
    credit += ("; COCONUT (Chandrasekhar et al., NAR 2025; CC0), the LOTUS initiative (Rutz et al., eLife 2022) and NPAtlas "
               "(van Santen et al., NAR 2022)" if np_prior else "")
    intro = INTRO.format(s2m_line=extra, s2m_credit=credit)
    cells = [cell("markdown", intro), cell("code", SETUP.format(modules=modules)), cell("code", RUN.format(neural=bool(neural or san), s2m_gamma=S2M_GAMMA if s2m else 0.0, np_beta=nb_beta if np_prior else 0.0,
                                                                            np_window=int(np_prior and np_window),
                                                                            run_extra=run_extra))]
    if not library_only:
        cells.append(cell("code", add_overlay.CELL.format(src=(ROOT / "casmi" / "network_overlay.py").read_text(), params={"lam": 4.0})))
        cells.append(cell("code", add_overlay.LIB_CELL.format(src=(ROOT / "casmi" / "library_overlay.py").read_text(),
                                                              params=dict(tau=0.5, down=3, min_ref=3))))
    if dreams:                        # analogue support from public spectra (panel, documentation-free lists: +0.026)
        cells.append(cell("code", add_overlay.DREAMS_CELL.format(src=(ROOT / "casmi" / "dreams_overlay.py").read_text(),
                                                                 params=dreams_params)))
    if edit_params is not None:       # opt-in: the class-3 generator lost on the LB twice (a7 0.412, a9 0.481 vs 0.485)
        cells.append(cell("code", add_overlay.EDIT_CELL.format(src=(ROOT / "casmi" / "edit_overlay.py").read_text(), params=edit_params)))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slug or ("casmi26-c1lib-probe" if library_only else "casmi26-c1n-compliant-neural" if neural else
            "casmi26-c1d-compliant-docfree" if (s2m and np_prior and np_window and w_doc == 0) else
            "casmi26-c1snpw-compliant" if (s2m and np_prior and np_window) else
            "casmi26-c1snp-compliant" if (s2m and np_prior) else
            "casmi26-c1s-compliant-spec2mol" if s2m else "casmi26-c1p-compliant-np" if np_prior else "casmi26-c1-compliant")
    path = out_dir / f"{slug}.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = {"id": f"adityachauhanx/{slug}", "title": slug.replace("-", " "), "code_file": path.name,
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["dmitriigluzdov/casmi26-fold-safe-fpnet", "metric/rdkit-2026-3-3-wheel"]
                                                         + (["sanpier/casmi-spectrum-model-v1"] if (neural or san) else [])
                                                         # spec2mol (casmi26-spec2mol: train.py / prep.py inside); nothing
                                                         # else of this dataset is loaded
                                                         + (["dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets"] if s2m else [])
                                                         + (["dmitriigluzdov/casmi26-natural-product-knowledge-table",
                                                             "pri2si17/coconut-natural-products-2026-09"] if np_prior else [])
                                                         + (["hengck23/hengck23-dreams-enveda-casmi26"] if dreams else []),
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
            "kernel_sources": ["adityachauhanx/casmi26-pubchem-index-build"] + dreams_kernels, "model_sources": []}
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return path


if __name__ == "__main__":
    params = dict(relatives=1, max_seconds=4800, rel_seconds=3600, rel_n_full=20, rel_n_docrank=20, rel_k=4, min_slot=1,
                  max_insert=1, device="cpu")
    if "--support-gate" in sys.argv:
        params.update(sup_gate=1, sup_sigma=0.6)
    if "--support-rerank" in sys.argv:
        params.update(sup_beta=1.0, sup_sigma=0.7)
    assert not ("--neural" in sys.argv and "--s2m" in sys.argv), "one neural variant per kernel"
    print("written", build(Path(sys.argv[1]), params if "--edit" in sys.argv else None, neural="--neural" in sys.argv,
                           s2m="--s2m" in sys.argv, np_prior="--np" in sys.argv, np_window="--npw" in sys.argv,
                           w_doc=next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--wdoc=")), None),
                           beta=next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--beta=")), None),
                           coco="--coco" in sys.argv,
                           lib_beta=next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--libbeta=")), None),
                           slug=next((a.split("=")[1] for a in sys.argv if a.startswith("--slug=")), None),
                           frag=next((a.split("=")[1] for a in sys.argv if a.startswith("--frag=")), None),
                           library_only="--library-only" in sys.argv,
                           san=next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--san=")), None),
                           npl=next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--npl=")), None),
                           lib_first=next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--libfirst=")), None),
                           extra_set=next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--set=")), None),
                           dreams=next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dreams=")), None)))
