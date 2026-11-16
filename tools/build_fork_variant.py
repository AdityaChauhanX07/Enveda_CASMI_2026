"""Build a variant of the fork notebook (Fork A + networking, a3) with changed ranking weights.

The fork re-ranks the final lists of a frozen copy of the public 0.465 pipeline (embedded as base64, SHA-256 pinned).
Documentation enters at four places: the frozen pipeline's PubChem-channel popularity prior (POP_LAM), its pool
popularity re-rank (POOLPOP_MU), its promotion gates (POP_TAU), and the fork's final merge (POP_W). The hidden test's
answers behave as poorly documented (the compliant pipeline gained +0.04 on the LB when documentation was dropped), so
these are the first knobs to try. Frozen-pipeline edits re-encode the copy and update its pinned SHA-256.

--s2m=G adds spec2mol (casmi26-spec2mol: public train.py / prep.py; in the attached reproduction-assets dataset) as a
third neural opinion: after the fork's 0.5 z(public) + 0.5 z(ours) blend, neural_score += G x z(spec2mol), standardised
within each molecule like the blend; any failure keeps the fork's own blend.

usage: python tools/build_fork_variant.py BASE_KERNEL_DIR OUT_DIR SLUG [--pop_w=X] [--np_w=X] [--spec_w=X]
       [--pop_lam=X] [--poolpop_mu=X] [--pop_tau=X] [--s2m=G] [--commit=fast|smoke]
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, what: str) -> str:
    assert text.count(old) == 1, f"{what}: expected one '{old[:60]}', found {text.count(old)}"
    return text.replace(old, new)


def edit_frozen(frozen: bytes, pop_lam=None, poolpop_mu=None, pop_tau=None) -> bytes:
    nb = json.loads(frozen)
    hits = {"pop_lam": pop_lam is None, "poolpop_mu": poolpop_mu is None, "pop_tau": pop_tau is None}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if pop_lam is not None and "POP_LAM, POP_UNION = 0.15, 200" in src:
            src = replace_once(src, "POP_LAM, POP_UNION = 0.15, 200", f"POP_LAM, POP_UNION = {pop_lam}, 200", "POP_LAM"); hits["pop_lam"] = True
        if poolpop_mu is not None and "POOLPOP_MU = 0.15" in src:
            src = replace_once(src, "POOLPOP_MU = 0.15", f"POOLPOP_MU = {poolpop_mu}", "POOLPOP_MU"); hits["poolpop_mu"] = True
        if pop_tau is not None and "S_TAU, POP_TAU = 6.0, 5.0" in src:
            src = replace_once(src, "S_TAU, POP_TAU = 6.0, 5.0", f"S_TAU, POP_TAU = 6.0, {pop_tau}", "POP_TAU"); hits["pop_tau"] = True
        c["source"] = src.splitlines(keepends=True)
    assert all(hits.values()), f"frozen edits not applied: {hits}"
    return json.dumps(nb, ensure_ascii=False, indent=1).encode()


S2M_CELL = """# ── spec2mol as a third neural opinion (casmi26-spec2mol: public train.py / prep.py) ──
S2M_GAMMA = __GAMMA__


def _s2m_fp_factory(bits):
    \"\"\"The fingerprint spec2mol (and the fold-safe FPNets) were trained on: six RDKit fingerprints + MACCS, bit-selected.\"\"\"
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator as g
    gens = [g.GetMorganGenerator(radius=2, fpSize=4096), g.GetMorganGenerator(radius=3, fpSize=4096),
            g.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=g.GetMorganFeatureAtomInvGen()),
            g.GetRDKitFPGenerator(fpSize=2048, maxPath=6), g.GetAtomPairGenerator(fpSize=2048),
            g.GetTopologicalTorsionGenerator(fpSize=2048)]

    def fp(smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        parts = [gen.GetFingerprintAsNumPy(m).astype(np.float32) for gen in gens]
        parts.append(np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32))
        return np.concatenate(parts)[bits]
    return fp


def _spec2mol_scores(features_path, queries_path):
    \"\"\"(molecule_id, smiles) -> spec2mol score: mean over the molecule's spectra of cosine x the model's scale.\"\"\"
    import glob as _glob, sys as _sys
    import torch as _torch
    _pt = sorted(_glob.glob('/kaggle/input/**/casmi26-spec2mol/spec2mol.pt', recursive=True), key=len)[0]
    _sys.path.insert(0, str(Path(_pt).parent))
    from spec2mol_model import SpecMol
    from spec2mol_features import spectrum
    _ck = _torch.load(_pt, map_location='cpu', weights_only=False)
    _model = SpecMol(**_ck['cfg']); _model.load_state_dict(_ck['model']); _model.eval()
    _scale = float(_model.scale().detach())
    _bits = np.load(sorted(_glob.glob('/kaggle/input/**/casmi26-fold-safe-fpnet/fp_bits.npy', recursive=True), key=len)[0])
    fp = _s2m_fp_factory(_bits)
    f = pd.read_parquet(features_path, columns=['molecule_id', 'smiles'])
    q = pd.read_parquet(queries_path)
    query = {str(m): g for m, g in q.groupby('molecule_id', sort=False)}
    out = {}
    with _torch.inference_mode():
        for mid, g in f.groupby('molecule_id', sort=False):
            qq = query.get(str(mid))
            if qq is None:
                continue
            feats = [spectrum(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz, r.adduct, r.ionization_mode,
                              r.collision_energy_ev) for r in qq.itertuples(index=False)]
            feats = [x for x in feats if x is not None]
            if not feats:
                continue
            z, _ = _model.encode_spectra(_torch.tensor(np.array([x[0] for x in feats], np.float32)), _torch.tensor([x[1] for x in feats]),
                                         _torch.tensor(np.array([x[2] for x in feats])), _torch.tensor([x[3] for x in feats]))
            smis = g.smiles.tolist()
            fps = [fp(s) for s in smis]
            ok = [i for i, x in enumerate(fps) if x is not None]
            if not ok:
                continue
            me = _model.encode_molecules(_torch.tensor(np.array([fps[i] for i in ok], np.float32)))
            for i, v in zip(ok, ((z @ me.T).mean(0) * _scale).numpy()):
                out[(mid, smis[i])] = float(v)
    s = pd.Series(list(out.values()), index=pd.MultiIndex.from_tuples(list(out.keys()), names=['molecule_id', 'smiles']), dtype=float)
    print('SPEC2MOL scored', len(s), 'candidates of', f.molecule_id.nunique(), 'molecules', flush=True)
    return s
"""

S2M_FUSE = """        print('ENSEMBLE neural = 0.5*z(public) + 0.5*z(ours)', flush=True)
        if globals().get('S2M_GAMMA', 0.0) > 0:      # spec2mol as a third neural opinion; failures keep the blend
            try:
                _s2m = _spec2mol_scores(ROOT / 'neural_candidate_features.parquet', ROOT / 'effective_query_spectra.parquet')
                _c = _pd.read_parquet(ROOT / 'neural_candidate_features.parquet').set_index(['molecule_id', 'smiles'])
                _sv = _s2m.reindex(_c.index)
                _sv = _sv.groupby(level=0).transform(lambda x: x.fillna(x.min()) if x.notna().any() else x.fillna(0.0))
                _c['neural_score'] = _c['neural_score'] + S2M_GAMMA * _z(_sv.fillna(0.0))
                _c.reset_index().to_parquet(ROOT / 'neural_candidate_features.parquet', index=False)
                print('SPEC2MOL fused into neural_score: gamma', S2M_GAMMA, '| scored', int(_s2m.reindex(_c.index).notna().sum()), 'of', len(_c), flush=True)
            except Exception as _exc:
                print('SPEC2MOL fusion skipped:', repr(_exc)[:300], flush=True)
"""


def main() -> None:
    base, out, slug = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    args = dict(a[2:].split("=", 1) for a in sys.argv[4:] if a.startswith("--") and "=" in a)
    nb_path = next(base.glob("*.ipynb"))
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]
    src_all = ["".join(c["source"]) for c in cells]
    joined = "\n".join(src_all)
    weights = re.search(r"POP_W, SPEC_W, NP_W = ([0-9.]+), ([0-9.]+), ([0-9.]+)", joined)
    pop_w, spec_w, np_w = (float(args.get(k, weights.group(i + 1))) for i, k in enumerate(("pop_w", "spec_w", "np_w")))
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        s = src_all[i]
        if weights.group(0) in s:
            s = replace_once(s, weights.group(0), f"POP_W, SPEC_W, NP_W = {pop_w}, {spec_w}, {np_w}", "fusion weights")
        if "commit" in args and "COMMIT_MODE = 'smoke'" in s:
            s = replace_once(s, "COMMIT_MODE = 'smoke'", f"COMMIT_MODE = '{args['commit']}'", "COMMIT_MODE")
        m = re.search(r"^FROZEN_B64\s*=\s*([^\n]+)", s, re.M)       # the assignment, not a use
        if any(k in args for k in ("pop_lam", "poolpop_mu", "pop_tau")) and m:
            old_raw = base64.b64decode(ast.literal_eval(m.group(1).strip()))
            new_raw = edit_frozen(old_raw, args.get("pop_lam"), args.get("poolpop_mu"), args.get("pop_tau"))
            s = s.replace(m.group(0), "FROZEN_B64 = " + repr(base64.b64encode(new_raw).decode()))
            old_sha, new_sha = hashlib.sha256(old_raw).hexdigest(), hashlib.sha256(new_raw).hexdigest()
            for j, cj in enumerate(cells):              # the pinned SHA-256 may sit in another cell
                if old_sha in src_all[j] and j != i:
                    src_all[j] = src_all[j].replace(old_sha, new_sha)
                    cj["source"] = src_all[j].splitlines(keepends=True)
            s = s.replace(old_sha, new_sha)
        c["source"] = s.splitlines(keepends=True)
        src_all[i] = s
    if "s2m" in args:                                    # spec2mol as a third neural opinion
        anchor = "        print('ENSEMBLE neural = 0.5*z(public) + 0.5*z(ours)', flush=True)\n"
        idx = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and anchor in "".join(c["source"])]
        assert len(idx) == 1, f"ensemble anchor found in {len(idx)} cells"
        s = replace_once("".join(cells[idx[0]]["source"]), anchor, S2M_FUSE, "spec2mol fusion")
        cells[idx[0]]["source"] = s.splitlines(keepends=True)
        new_cell = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                    "source": S2M_CELL.replace("__GAMMA__", repr(float(args["s2m"]))).splitlines(keepends=True)}
        cells.insert(idx[0], new_cell)
    out.mkdir(parents=True, exist_ok=True)
    (out / nb_path.name).write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = json.loads((base / "kernel-metadata.json").read_text())
    meta.update(id=f"adityachauhanx/{slug}", title=slug.replace("-", " "), code_file=nb_path.name)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / nb_path.name, "| fusion", (pop_w, spec_w, np_w),
          {k: args[k] for k in ("pop_lam", "poolpop_mu", "pop_tau", "s2m", "commit") if k in args})


if __name__ == "__main__":
    main()
