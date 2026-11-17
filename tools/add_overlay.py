"""Append the networking overlay as the last cell of a forked notebook.

Usage: python tools/add_overlay.py IN.ipynb OUT.ipynb [lam] [--library[=tau:down:min_ref]] [--edit [--relatives [--docrank] [--k4] [--safe] [--support-gate | --support-rerank]]]
(--library appends the library-consistency overlay after networking; --edit also appends the edit generator)
The cell runs after every layer of the fork, re-orders submission.csv in place and keeps the fork's file on any
error. It is skipped in a fast commit (dummy answers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELL = '''# ── Networking overlay: spectral neighbours inside the test set re-order ranks of linked molecules ──
import json, shutil, traceback
from pathlib import Path as _P
_NET_SRC = {src!r}
_NET_PARAMS = {params!r}
_fast = (not globals().get('PIPELINE_DONE', True)) and (not globals().get('IS_RERUN', False)) and globals().get('COMMIT_MODE') == 'fast'
if _fast:
    print('NETWORK OVERLAY: skipped in fast commit')
else:
    try:
        _p = _P('/kaggle/working/p2s_code/network_overlay.py'); _p.parent.mkdir(parents=True, exist_ok=True)
        _p.write_text(_NET_SRC)
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location('network_overlay', _p); _no = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_no)
        _params = dict(_no.PARAMS); _params.update(_NET_PARAMS)
        _sub = _P('/kaggle/working/submission.csv'); shutil.copy2(_sub, '/kaggle/working/submission_before_network.csv')
        _rep = _no.apply(COMP / 'test.parquet', _sub, '/kaggle/working/submission_network.csv',
                         '/kaggle/working/network_report.json', _params)
        _f = pd.read_csv('/kaggle/working/submission_network.csv', dtype=str, keep_default_na=False)
        assert _f.molecule_id.tolist() == SAMPLE_IDS and _f.smiles.str.split(';').map(len).between(1, 25).all()
        shutil.copy2('/kaggle/working/submission_network.csv', _sub)
        print('NETWORK OVERLAY:', json.dumps(_rep))
    except Exception:
        traceback.print_exc()
        if _P('/kaggle/working/submission_before_network.csv').is_file():
            shutil.copy2('/kaggle/working/submission_before_network.csv', '/kaggle/working/submission.csv')
        print('NETWORK OVERLAY: failed, fork answers kept')
'''


def build(src_nb: Path, out_nb: Path, params: dict) -> None:
    nb = json.loads(src_nb.read_text())
    code = (ROOT / "casmi" / "network_overlay.py").read_text()
    text = CELL.format(src=code, params=params)
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                        "source": text.splitlines(keepends=True)})
    out_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=1))




DREAMS_CELL = '''# ── DreaMS analogue support: lists re-ranked by the structures of the query's nearest public spectra (MassSpecGym) ──
import json, shutil, time, traceback, glob as _glob
from pathlib import Path as _P
_DR_SRC = {src!r}
_DR_PARAMS = {params!r}
_fast = (not globals().get('PIPELINE_DONE', True)) and (not globals().get('IS_RERUN', False)) and globals().get('COMMIT_MODE') == 'fast'
_left = 8.5 * 3600 - (time.time() - globals().get('NB_T0', time.time()))
if _fast or _left < 1500:
    print('DREAMS OVERLAY: skipped', 'fast commit' if _fast else 'time guard')
else:
    _DR_PARAMS.update(max_seconds=min(_DR_PARAMS.get('max_seconds', 2400), _left - 900))
    try:
        _p = _P('/kaggle/working/p2s_code/dreams_overlay.py'); _p.parent.mkdir(parents=True, exist_ok=True)
        _p.write_text(_DR_SRC)
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location('dreams_overlay', _p); _do = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_do)
        _dd = _P(sorted(_glob.glob('/kaggle/input/**/MassSpecGym_DreaMS_embeddings.npy', recursive=True), key=len)[0]).parent
        _sub = _P('/kaggle/working/submission.csv'); shutil.copy2(_sub, '/kaggle/working/submission_before_dreams.csv')
        _keep = {{}}                     # leads that stay on top: strong library matches (usually right already)
        try:
            _cur = pd.read_csv(_sub, dtype=str, keep_default_na=False)
            _lead = {{m: s.split(';')[0] for m, s in zip(_cur.molecule_id, _cur.smiles)}}
            _lf = _P('/kaggle/working/lib_first_leads.json')                 # C1: library hits >= lib_first lead
            if _lf.is_file():
                _keep.update({{str(k): list(v) for k, v in json.loads(_lf.read_text()).items()}})
            _ev = _P('/kaggle/working/frozen_candidate_evidence.parquet')    # fork: lists with a strong library match
            if _ev.is_file():
                _e = pd.read_parquet(_ev, columns=['molecule_id', 'library_max'])
                for _m, _v in _e.groupby('molecule_id', sort=False).library_max.max().items():
                    if _v == _v and float(_v) >= 0.9 and str(_m) in _lead:
                        _keep.setdefault(str(_m), [_lead[str(_m)]])
            _pre = _P('/kaggle/working/submission_pre_apex_extlib.csv')     # fork: leaders promoted by external-library hits
            if _pre.is_file():
                _p0 = pd.read_csv(_pre, dtype=str, keep_default_na=False)
                for _m, _s in zip(_p0.molecule_id, _p0.smiles):
                    if _m in _lead and _s.split(';')[0] != _lead[_m]:
                        _keep.setdefault(_m, [_lead[_m]])
        except Exception:
            traceback.print_exc()
        print('DREAMS OVERLAY: protected leads for', len(_keep), 'molecules', flush=True)
        _rep = _do.apply(COMP / 'test.parquet', '/kaggle/working/submission_before_dreams.csv', _dd,
                         '/kaggle/working/submission_dreams.csv', '/kaggle/working/dreams_report.json', _DR_PARAMS, keep=_keep)
        _f = pd.read_csv('/kaggle/working/submission_dreams.csv', dtype=str, keep_default_na=False)
        assert _f.molecule_id.tolist() == SAMPLE_IDS and _f.smiles.str.split(';').map(len).between(1, 25).all()
        shutil.copy2('/kaggle/working/submission_dreams.csv', _sub)
        print('DREAMS OVERLAY:', json.dumps(_rep))
    except Exception:
        traceback.print_exc()
        if _P('/kaggle/working/submission_before_dreams.csv').is_file():
            shutil.copy2('/kaggle/working/submission_before_dreams.csv', '/kaggle/working/submission.csv')
        print('DREAMS OVERLAY: failed, previous answers kept')
'''


LIB_CELL = '''# ── Library-consistency overlay: candidates whose public reference spectra contradict the query move down ──
import json, shutil, time, traceback
from pathlib import Path as _P
_LIB_SRC = {src!r}
_LIB_PARAMS = {params!r}
_fast = (not globals().get('PIPELINE_DONE', True)) and (not globals().get('IS_RERUN', False)) and globals().get('COMMIT_MODE') == 'fast'
_left = 8.5 * 3600 - (time.time() - globals().get('NB_T0', time.time()))
if _fast or _left < 1200:
    print('LIBRARY OVERLAY: skipped', 'fast commit' if _fast else 'time guard')
else:
    _LIB_PARAMS.update(max_seconds=min(_LIB_PARAMS.get('max_seconds', 1200), _left - 900))
    try:
        _p = _P('/kaggle/working/p2s_code/library_overlay.py'); _p.parent.mkdir(parents=True, exist_ok=True)
        _p.write_text(_LIB_SRC)
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location('library_overlay', _p); _lo = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_lo)
        _sub = _P('/kaggle/working/submission.csv'); shutil.copy2(_sub, '/kaggle/working/submission_before_library.csv')
        _rep = _lo.apply(COMP / 'test.parquet', _sub, COMP / 'train.parquet', '/kaggle/working/submission_library.csv',
                         '/kaggle/working/library_report.json', _LIB_PARAMS)
        _f = pd.read_csv('/kaggle/working/submission_library.csv', dtype=str, keep_default_na=False)
        assert _f.molecule_id.tolist() == SAMPLE_IDS and _f.smiles.str.split(';').map(len).between(1, 25).all()
        shutil.copy2('/kaggle/working/submission_library.csv', _sub)
        print('LIBRARY OVERLAY:', json.dumps(_rep))
    except Exception:
        traceback.print_exc()
        if _P('/kaggle/working/submission_before_library.csv').is_file():
            shutil.copy2('/kaggle/working/submission_before_library.csv', '/kaggle/working/submission.csv')
        print('LIBRARY OVERLAY: failed, previous answers kept')
'''


def add_library(nb_path: Path, params: dict | None = None) -> None:
    nb = json.loads(nb_path.read_text())
    text = LIB_CELL.format(src=(ROOT / "casmi" / "library_overlay.py").read_text(), params=dict(params or {}))
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                        "source": text.splitlines(keepends=True)})
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


EDIT_CELL = '''# ── Neighbour-edit generator: one-modification products of spectral neighbours' candidates, gated by the fold-safe FPNet ──
import json, shutil, time, traceback
from pathlib import Path as _P
_EDIT_SRC = {src!r}
_EDIT_PARAMS = {params!r}
_fast = (not globals().get('PIPELINE_DONE', True)) and (not globals().get('IS_RERUN', False)) and globals().get('COMMIT_MODE') == 'fast'
_left = 8.5 * 3600 - (time.time() - globals().get('NB_T0', time.time()))      # the fork's own stages stop at 8.55-8.65 h
if _fast or _left < 600:
    print('EDIT OVERLAY: skipped', 'fast commit' if _fast else 'time guard')
else:
    _EDIT_PARAMS.update(max_seconds=min(_EDIT_PARAMS.get('max_seconds', 3600), _left - 300),
                        rel_seconds=min(_EDIT_PARAMS.get('rel_seconds', 2400), _left - 900))
    if not globals().get('IS_RERUN', False):      # commit runs only validate the code path (most lists are placeholders);
        # the first pass takes ~5 min on the GPU machine, so the relative search gets ~10 min of its own here
        _EDIT_PARAMS.update(max_seconds=min(_EDIT_PARAMS['max_seconds'], 1500), rel_seconds=min(_EDIT_PARAMS['rel_seconds'], 900))
    if _EDIT_PARAMS.get('sup_beta', 0) > 0 or _EDIT_PARAMS.get('sup_gate', 0):   # product support reads train.parquet
        _EDIT_PARAMS.update(sup_train=str(COMP / 'train.parquet'),
                            sup_seconds=max(0, min(_EDIT_PARAMS.get('sup_seconds', 600), _left - _EDIT_PARAMS['rel_seconds'] - 600,
                                                   600 if globals().get('IS_RERUN', False) else 300)))
    try:
        _p = _P('/kaggle/working/p2s_code/edit_overlay.py'); _p.parent.mkdir(parents=True, exist_ok=True)
        _p.write_text(_EDIT_SRC)
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location('edit_overlay', _p); _eo = _ilu.module_from_spec(_spec)
        import sys as _sys; _sys.modules['edit_overlay'] = _eo     # worker processes find the module by name
        _spec.loader.exec_module(_eo)
        _sub = _P('/kaggle/working/submission.csv'); shutil.copy2(_sub, '/kaggle/working/submission_before_edit.csv')
        _rep = _eo.apply(COMP / 'test.parquet', _sub, '/kaggle/working/submission_edit.csv', '/kaggle/working/edit_report.json',
                         _EDIT_PARAMS, **globals().get('EDIT_KW', {{}}))   # EDIT_KW: the compliant notebook's model / index paths
        _f = pd.read_csv('/kaggle/working/submission_edit.csv', dtype=str, keep_default_na=False)
        assert _f.molecule_id.tolist() == SAMPLE_IDS and _f.smiles.str.split(';').map(len).between(1, 25).all()
        shutil.copy2('/kaggle/working/submission_edit.csv', _sub)
        print('EDIT OVERLAY:', json.dumps({{k: v for k, v in _rep.items() if k != 'params'}}))
    except Exception:
        traceback.print_exc()
        if _P('/kaggle/working/submission_before_edit.csv').is_file():
            shutil.copy2('/kaggle/working/submission_before_edit.csv', '/kaggle/working/submission.csv')
        print('EDIT OVERLAY: failed, previous answers kept')
'''


def add_edit(nb_path: Path, params: dict | None = None) -> None:
    nb = json.loads(nb_path.read_text())
    text = EDIT_CELL.format(src=(ROOT / "casmi" / "edit_overlay.py").read_text(), params=dict(params or {}))
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                        "source": text.splitlines(keepends=True)})
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lam = float(args[2]) if len(args) > 2 else 4.0
    build(Path(args[0]), Path(args[1]), {"lam": lam})
    lib = [a for a in sys.argv[1:] if a == "--library" or a.startswith("--library=")]
    if lib:                                    # a8: before the edit overlay, so products are compared with the re-ranked top 3
        spec = lib[0].split("=", 1)[1].split(":") if "=" in lib[0] else []
        lp = {}
        if spec:
            lp = dict(tau=float(spec[0]), down=int(spec[1]), min_ref=int(spec[2]))
        add_library(Path(args[1]), lp)
    if "--edit" in sys.argv:
        # --relatives also turns on the PubChem-relative products. Budget: 60 min for the relative search (task 4,
        # experiments/search_order.py: weakest lists first keeps 88 % of the full-coverage gain at 50 % coverage,
        # ~96 % at 75 %); the cell still caps everything at the time left before 8.5 h
        params = dict(relatives=1, max_seconds=4800, rel_seconds=3600) if "--relatives" in sys.argv else {}
        if "--docrank" in sys.argv:            # a6: 20 rows by the ECFP4 pass + 20 best-documented rows fully scored per window
            params.update(rel_n_full=20, rel_n_docrank=20)
        if "--k4" in sys.argv:                 # a7: four relatives kept per window (panel: class-3 0.238 -> 0.258, same time)
            params.update(rel_k=4)
        if "--safe" in sys.argv:               # a9: never above rank 2, one product per list (a7 lost 0.074 on the LB by
            params.update(min_slot=1, max_insert=1)   # displacing correct, poorly documented leaders)
        if "--support-gate" in sys.argv:       # insert only products whose relatives' spectra match the query (>= 0.6)
            params.update(sup_gate=1, sup_sigma=0.6)
        if "--support-rerank" in sys.argv:     # products ranked by score + 1.0 x support (>= 0.7)
            params.update(sup_beta=1.0, sup_sigma=0.7)
        add_edit(Path(args[1]), params)
    print("written", args[1])
