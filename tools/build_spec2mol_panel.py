"""Build the Kaggle CPU job that checks spec2mol on the panel (casmi26-spec2mol-panel).

spec2mol (Dmitrii Gluzdov; prep.py / train.py public) held out the panel's 250 natural products, so unlike sanpier's
model it can be measured on the panel. The job re-ranks C1's final panel lists (class 2: the answer's spectra hidden;
class-1 view: the queries' own library left out) by reciprocal-rank fusion 1/(3 + C1 rank) + w / (3 + spec2mol rank).
Lists come from the C1 dumps (artifacts/c1_dump_c2_v1.pkl, c1_dump_c1_v1.pkl) with the module's assembly rule.

usage: python tools/build_spec2mol_panel.py OUT_DIR
"""
import base64, json, pickle, sys, zlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT))
from casmi.edit_overlay import Chemistry, Tier  # noqa: E402

TEMPLATE = (ROOT / "tools" / "templates" / "spec2mol_panel.py.tmpl").read_text()   # %r slots: lists JSON, edit_overlay source


def dump_candidates():
    """molecule -> [[metric key, smiles], ...]: every row of the c2 dump's PubChem lists and every hit of both dumps."""
    tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    out = {}
    for tag in ("c2", "c1"):
        d = pickle.load(open(ROOT / "artifacts" / f"c1_dump_{tag}_v1.pkl", "rb"))
        for m in d["ids"]:
            seen = {k for k, _ in out.get(m, [])}
            rows = [tier.smiles(r) for r, _ in d["pub"].get(m, [])] if tag == "c2" else []
            for s_ in rows + [s_ for _, (c, s_) in d["hits"].get(m, {}).items()]:
                k = chem.key(s_)
                if k and k not in seen:
                    seen.add(k); out.setdefault(m, []).append([k, s_])
    return out


def panel_lists():
    tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    chem = Chemistry(np.load(EXT / "foldsafe" / "fp_bits.npy"))
    out = {}
    for tag in ("c2", "c1"):
        d = pickle.load(open(ROOT / "artifacts" / f"c1_dump_{tag}_v1.pkl", "rb"))
        lists = {}
        for m in d["ids"]:
            sc = {}
            for r, v in d["pub"].get(m, []):
                s_ = tier.smiles(r); k = chem.key(s_)
                if k and k not in sc:
                    sc[k] = (v, s_)
            base = min(v for v, _ in sc.values()) if sc else 0.0
            for _, (c, s_) in d["hits"].get(m, {}).items():
                if c < 0.8:
                    continue
                k = chem.key(s_)
                if k:
                    sc[k] = (sc[k][0] + 4.0 * (c - 0.8), sc[k][1]) if k in sc else (base + 4.0 * (c - 0.8), s_)
            lists[m] = [[k, s_] for k, (v, s_) in sorted(sc.items(), key=lambda x: -x[1][0])[:25]]
        out[tag] = lists
    return out


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    lists = panel_lists(); lists["all"] = dump_candidates()   # "all": every dumped candidate, scored and saved for replays
    pack = lambda text: base64.b64encode(zlib.compress(text.encode(), 9)).decode()   # Kaggle caps a script at 1 MB
    script = TEMPLATE % (pack(json.dumps(lists)), pack((ROOT / "casmi" / "edit_overlay.py").read_text()))
    (out / "spec2mol_panel.py").write_text(script)
    meta = {"id": "adityachauhanx/casmi26-spec2mol-panel", "title": "casmi26 spec2mol panel", "code_file": "spec2mol_panel.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets",
                                                          "dmitriigluzdov/casmi26-fold-safe-fpnet", "metric/rdkit-2026-3-3-wheel"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "spec2mol_panel.py")
