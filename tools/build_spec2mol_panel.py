"""Build the Kaggle CPU job that checks spec2mol on the panel (casmi26-spec2mol-panel).

spec2mol (Dmitrii Gluzdov; prep.py / train.py public) held out the panel's 250 natural products, so unlike sanpier's
model it can be measured on the panel. The job re-ranks C1's final panel lists (class 2: the answer's spectra hidden;
class-1 view: the queries' own library left out) by reciprocal-rank fusion 1/(3 + C1 rank) + w / (3 + spec2mol rank).
Lists come from the C1 dumps (artifacts/c1_dump_c2_v1.pkl, c1_dump_c1_v1.pkl) with the module's assembly rule.

usage: python tools/build_spec2mol_panel.py OUT_DIR
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT))
from casmi.edit_overlay import Chemistry, Tier  # noqa: E402

TEMPLATE = '# Panel check: spec2mol (held out the panel\'s molecules) re-ranking C1\'s class-2 and class-1-view lists (CPU).\nimport glob, json, os, subprocess, sys, time\nfrom pathlib import Path\nimport numpy as np, pandas as pd\n_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"\n_whl = [w for w in glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True) if _tag in w and "2026.3.3" in w]\nsubprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", _whl[0]], check=True)\nimport torch\nLISTS = json.loads(%r)\nW = Path("/kaggle/working/eo"); W.mkdir(parents=True, exist_ok=True); (W / "edit_overlay.py").write_text(%r)\nsys.path.insert(0, str(W))\nimport edit_overlay as eo\nINPUT = Path("/kaggle/input")\nS2M = Path(sorted(glob.glob(str(INPUT / "**/casmi26-spec2mol/spec2mol.pt"), recursive=True), key=len)[0]).parent\nsys.path.insert(0, str(S2M))\nfrom spec2mol_model import SpecMol\nfrom spec2mol_features import spectrum\nck = torch.load(S2M / "spec2mol.pt", map_location="cpu", weights_only=False)\nmodel = SpecMol(**ck["cfg"]); model.load_state_dict(ck["model"]); model.eval()\nbits = np.load(sorted(glob.glob(str(INPUT / "**/casmi26-fold-safe-fpnet/fp_bits.npy"), recursive=True) + glob.glob(str(INPUT / "**/fp_bits.npy"), recursive=True), key=len)[0])\nchem = eo.Chemistry(bits)\ncomp = next(Path(t).parent for t in glob.glob(str(INPUT / "**/train.parquet"), recursive=True) if "enveda" in t.lower())\nimport pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq\npf = pq.ParquetFile(comp / "train.parquet"); parts = []\nfor g in range(pf.num_row_groups):\n    msk = pc.equal(pf.read_row_group(g, columns=["ingest_lib"])["ingest_lib"], "enveda-np-examples")\n    idx = np.flatnonzero(np.asarray(msk.to_numpy(zero_copy_only=False), dtype=bool))\n    if len(idx): parts.append(pf.read_row_group(g).take(pa.array(idx)))\nt = pa.concat_tables(parts).to_pandas()\nt["m"] = [chem.key(s) for s in t.normalized_smiles]\nprint("np-examples spectra", len(t), "molecules", t.m.nunique(), "| spec2mol cfg", ck["cfg"], "val", ck.get("val"), flush=True)\nscale = float(model.scale())\nneural = {}\nwith torch.inference_mode():\n    for tagv in ("c2", "c1"):\n        for m, cand in LISTS[tagv].items():\n            if (tagv, m) in neural: continue\n            g = t[t.m == m]\n            feats = [spectrum(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz, r.adduct, r.ionization_mode, r.collision_energy_ev) for r in g.itertuples()]\n            feats = [f for f in feats if f is not None]\n            fps = [chem.fingerprint(s) for _, s in cand]\n            ok = [i for i, f in enumerate(fps) if f is not None]\n            if not feats or not ok:\n                neural[(tagv, m)] = None; continue\n            pk = torch.tensor(np.array([f[0] for f in feats], np.float32)); npk = torch.tensor([f[1] for f in feats]); meta = torch.tensor(np.array([f[2] for f in feats]))\n            ad = torch.tensor([f[3] for f in feats])\n            z, _ = model.encode_spectra(pk, npk, meta, ad)\n            me = model.encode_molecules(torch.tensor(np.array([fps[i] for i in ok], np.float32)))\n            sc = np.full(len(cand), -1e9); sc[ok] = ((z @ me.T).mean(0) * scale).numpy()\n            neural[(tagv, m)] = sc\ndef mrr(tagv, w):\n    tot = 0.0\n    for m, cand in LISTS[tagv].items():\n        keys = [k for k, _ in cand]\n        sc = neural.get((tagv, m))\n        if sc is not None and w > 0:\n            nr = np.empty(len(cand)); nr[np.argsort(-sc, kind="stable")] = np.arange(1, len(cand) + 1)\n            fused = 1.0 / (3.0 + nr) if w > 1e6 else 1.0 / (3.0 + np.arange(1, len(cand) + 1)) + w / (3.0 + nr)\n            order = np.argsort(-fused, kind="stable"); keys = [keys[i] for i in order]\n        tot += 1.0 / (keys.index(m) + 1) if m in keys else 0.0\n    return tot / 250\nprint("neural weight | class 2 (answer\'s spectra hidden) | class-1 view")\nfor w in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 1e9):\n    print(f"{\'neural only\' if w > 1e6 else w:>13} | {mrr(\'c2\', w):.4f} | {mrr(\'c1\', w):.4f}", flush=True)\n'


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
    script = TEMPLATE % (json.dumps(panel_lists()), (ROOT / "casmi" / "edit_overlay.py").read_text())
    (out / "spec2mol_panel.py").write_text(script)
    meta = {"id": "adityachauhanx/casmi26-spec2mol-panel", "title": "casmi26 spec2mol panel", "code_file": "spec2mol_panel.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets",
                                                          "dmitriigluzdov/casmi26-fold-safe-fpnet", "metric/rdkit-2026-3-3-wheel"],
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / "spec2mol_panel.py")
