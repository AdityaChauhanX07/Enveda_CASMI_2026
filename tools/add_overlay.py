"""Append the networking overlay as the last cell of a forked notebook.

Usage: python tools/add_overlay.py IN.ipynb OUT.ipynb [lam] [--edit [--relatives]]   (--edit also appends the edit generator)
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
    try:
        _p = _P('/kaggle/working/p2s_code/edit_overlay.py'); _p.parent.mkdir(parents=True, exist_ok=True)
        _p.write_text(_EDIT_SRC)
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location('edit_overlay', _p); _eo = _ilu.module_from_spec(_spec)
        import sys as _sys; _sys.modules['edit_overlay'] = _eo     # worker processes find the module by name
        _spec.loader.exec_module(_eo)
        _sub = _P('/kaggle/working/submission.csv'); shutil.copy2(_sub, '/kaggle/working/submission_before_edit.csv')
        _rep = _eo.apply(COMP / 'test.parquet', _sub, '/kaggle/working/submission_edit.csv', '/kaggle/working/edit_report.json',
                         _EDIT_PARAMS)
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
    if "--edit" in sys.argv:
        # --relatives also turns on the PubChem-relative products (panel: class-2 0.0000, class-3 +0.053)
        add_edit(Path(args[1]), dict(relatives=1, max_seconds=3600, rel_seconds=2400) if "--relatives" in sys.argv else {})
    print("written", args[1])
