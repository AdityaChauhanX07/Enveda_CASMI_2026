"""Append the networking overlay as the last cell of a forked notebook.

Usage: python tools/add_overlay.py IN.ipynb OUT.ipynb [lam]
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


if __name__ == "__main__":
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    build(Path(sys.argv[1]), Path(sys.argv[2]), {"lam": lam})
    print("written", sys.argv[2])
