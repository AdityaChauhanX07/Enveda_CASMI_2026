"""Compliant pipeline (C1): candidate lists from inputs that all have public generation code.

1. PubChem candidates: every structure of our own PubChem index (tools/build_pubchem_index.py, PubChem's primary
   files) within `ppm` of the molecule's neutral mass (the `max_window` best-documented rows when the window is
   larger), scored by the five fold-safe FPNets (dmitriigluzdov/casmi26-fold-safe-fpnet: training code public)
   as z-standardised fingerprint . logits + w_doc * (log1p substances + log1p PubMed); full fingerprints only for
   the n_partial best rows by the ECFP4-block score plus the n_docpre best-documented rows.
2. Library hits (class 1): train.parquet spectra with the same adduct and a precursor within `lib_tol` Da of a
   query spectrum; a structure whose best cosine c reaches `lib_hit` gains lib_beta * (c - lib_c0) on its score
   (added at the list's lowest score if missing) - a strong match adds evidence instead of overriding the list.
3. Top 25 per molecule, one per metric key (stereo removed, canonical tautomer, InChIKey first block).
The overlays (networking, library consistency, guarded edit) run on the result as for the other notebooks.
Uses casmi/edit_overlay.py for spectra, logits, fingerprints and the index reader.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS = dict(ppm=10.0, w_doc=0.25, max_window=20000, n_partial=2000, n_docpre=400, window_sample=200, seed=0,
              top_k=25, lib_tol=0.01, lib_hit=0.8, lib_min_matches=3, lib_beta=4.0, lib_c0=0.8,
              lib_cap=200, workers=4, max_seconds=5400, device="cpu", exclude_libs=(),
              # optional neural re-rank of each list (sanpier/casmi-spectrum-model-v1: train.py and data audit public;
              # trained on all of train.parquet, so not testable on the panel): reciprocal-rank fusion
              # 1/(offset + C1 rank) + neural_weight / (offset + neural rank), Fork A's "Neural025" rule (0.25, 3)
              neural_bundle="", neural_weight=0.25, neural_offset=3.0, neural_top=25,
              # optional spec2mol fusion over each molecule's whole shortlist, before the cut to top_k (casmi26-spec2mol:
              # train.py / prep.py public; it held out the panel's molecules): score + s2m_gamma x spec2mol score /
              # scale, scale = s2m_scale or the median within-shortlist sd over molecules with >= s2m_min_n scores
              # (s2m_fallback below 20 such molecules; panel 5.5). Off unless s2m_bundle is set and s2m_gamma > 0
              s2m_bundle="", s2m_gamma=0.0, s2m_scale=0.0, s2m_min_n=5, s2m_fallback=5.5,
              # optional natural-product prior: + np_beta for a candidate listed in LOTUS / NPAtlas (np_table:
              # natural_products.parquet of casmi26-natural-product-knowledge-table, build_np.py public; scorer key) or
              # in COCONUT (np_coconut: the COCONUT CSV-lite export; standard InChIKey first block), + np_delta x
              # ln(1 + LOTUS references). Off unless np_beta or np_delta is set and a table is given
              np_table="", np_coconut="", np_beta=0.0, np_delta=0.0,
              # np_window 1: natural-product members of a window are never lost to C1's documentation-based cuts -
              # all kept when the window exceeds max_window, all (up to np_full, best by the ECFP4-block score) fully
              # fingerprinted, and the np_keep best by score carried into the shortlist beside the 4 x top_k best rows
              np_window=0, np_keep=50, np_full=1000,
              # np_hits 1: a candidate with a library hit (>= lib_hit) gains the bonus as if a member, so a strong
              # match is never outranked by membership alone (panel, class-1 view with the answer not a member, bonus 3:
              # 0.520 -> 0.814; with the answer a member and on class 2 within 0.008)
              np_hits=1,
              # panel only: the answer's documentation capped at stress_cap inside the window stage (stress_ik:
              # molecule -> the answer's standard InChIKey first block), the a7 lesson applied to every cut
              stress_ik=None, stress_cap=None)
# exclude_libs: panel only - train.parquet libraries left out of the library search (the queries' own library)

_W = {}


def _one_thread(store: dict) -> None:
    """Worker processes: one BLAS / OpenMP thread each. Forked workers otherwise each keep numpy's thread pool and
    four workers on four cores ran ~5x slower (37 vs 7 s per molecule in C1's window scoring)."""
    try:
        from threadpoolctl import threadpool_limits
        store["_threads"] = threadpool_limits(1)
    except Exception:
        pass


def h14(a) -> np.ndarray:
    """uint64 hashes of InChIKey first blocks (bytes or str, 14 characters)."""
    arr = np.ascontiguousarray(np.asarray(a, "S14")).view(np.uint8).reshape(-1, 14)
    hi = arr[:, :8].copy().view(np.uint64).ravel()
    lo = np.zeros((len(arr), 8), np.uint8); lo[:, :6] = arr[:, 8:]
    return hi * np.uint64(0x9E3779B97F4A7C15) + lo.view(np.uint64).ravel()


def is_member(hashes: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Membership of hashes in the sorted member hashes."""
    if not len(members) or not len(hashes):
        return np.zeros(len(hashes), bool)
    pos = np.minimum(np.searchsorted(members, hashes), len(members) - 1)
    return members[pos] == hashes


def load_np(p: dict) -> tuple[dict, set, np.ndarray]:
    """(scorer key -> LOTUS references, COCONUT InChIKey first blocks, sorted member hashes of both tables)."""
    refs, coco, blocks = {}, set(), []
    if p["np_table"]:
        npt = pd.read_parquet(p["np_table"], columns=["ik14", "scorer_key", "n_references"])
        npt["n_references"] = pd.to_numeric(npt.n_references, errors="coerce").fillna(0.0)
        refs = npt.groupby("scorer_key").n_references.max().to_dict()
        blocks.append(npt.ik14.dropna().astype(str).str.slice(0, 14).to_numpy())
    if p["np_coconut"]:
        coco = set(pd.read_csv(p["np_coconut"], usecols=["standard_inchi_key"]).standard_inchi_key.dropna().str.slice(0, 14))
        blocks.append(np.array(sorted(coco)))
    hashes = np.unique(h14(np.concatenate(blocks))) if blocks else np.zeros(0, np.uint64)
    return refs, coco, hashes


def _init_worker(index_root, doc_root, bits_path, member_path=""):
    import edit_overlay as eo
    _one_thread(_W)
    _W["tier"] = eo.Tier(Path(index_root), Path(doc_root))
    _W["chem"] = eo.Chemistry(np.load(bits_path))
    _W["members"] = np.load(member_path) if member_path else None


def _score_window(task):
    """(molecule, [(row, score)] best first) for one molecule: fingerprint . logits, standardised, + doc."""
    m, mass, z, p = task
    tier, chem = _W["tier"], _W["chem"]
    try:
        lo, hi = tier.window(mass, p["ppm"])
        if hi <= lo:
            return m, [], None
        rows = np.arange(lo, hi)
        doc = np.asarray(tier.lsid[lo:hi], np.float64) + np.asarray(tier.lpmid[lo:hi], np.float64)
        if p.get("stress_ik") and m in p["stress_ik"] and p.get("stress_cap") is not None:   # panel only
            ans = np.asarray(tier.ik[lo:hi]) == str(p["stress_ik"][m]).encode()
            doc[ans] = np.minimum(doc[ans], float(p["stress_cap"]))
        members = _W.get("members") if p.get("np_window") else None
        mem = is_member(h14(np.asarray(tier.ik[lo:hi])), members) if members is not None else None
        if len(rows) > p["max_window"]:
            keep = np.argsort(-doc, kind="stable")[: p["max_window"]]
            if mem is not None:                        # natural products are never cut for lack of documentation
                keep = np.union1d(keep, np.flatnonzero(mem))
            keep = np.sort(keep)
            rows, doc = rows[keep], doc[keep]
            mem = mem[keep] if mem is not None else None
        # stage 1: ECFP4-block score of every row; stage 2: full fingerprints for the n_partial best by it plus the
        # n_docpre best-documented (panel: all answers kept at 2,000 + 400 while fingerprinting 27 % of the rows)
        z_e = z[chem.sel_e]
        part = np.array([chem.partial(tier.smiles(int(r)), z_e) for r in rows])
        sel = set(np.argsort(-part, kind="stable")[: p["n_partial"]].tolist()) | set(np.argsort(-doc, kind="stable")[: p["n_docpre"]].tolist())
        if mem is not None and mem.any():              # every member (up to np_full, best by stage 1) fully scored
            idx = np.flatnonzero(mem)
            sel |= set(idx[np.argsort(-part[idx], kind="stable")[: p["np_full"]]].tolist())
        sel = np.array(sorted(sel), dtype=np.int64)
        sc = np.full(len(sel), np.nan)
        for i, j in enumerate(sel):
            f = chem.fingerprint(tier.smiles(int(rows[j])))
            if f is not None:
                sc[i] = float(f @ z)
        # the window's scale from a uniform sample of its rows (as the edit overlay's z), so doc keeps its weight
        rng = np.random.default_rng(p["seed"])
        samp = rows if len(rows) <= p["window_sample"] else rows[np.sort(rng.choice(len(rows), p["window_sample"], replace=False))]
        ref = [chem.fingerprint(tier.smiles(int(r))) for r in samp]
        ref = np.array([f @ z for f in ref if f is not None])
        if len(ref) < 2 or np.isfinite(sc).sum() < 1:
            return m, [], None
        zs = (sc - ref.mean()) / (ref.std() + 1e-9)
        total = np.where(np.isfinite(sc), zs + p["w_doc"] * doc[sel], -np.inf)
        ranked = np.argsort(-total, kind="stable")
        order = ranked[: 4 * p["top_k"]].tolist()
        if mem is not None:                            # the np_keep best members join the shortlist
            top = set(order)
            order += [int(i) for i in ranked if mem[sel[i]] and int(i) not in top][: p["np_keep"]]
        return m, [(int(rows[sel[i]]), float(total[i])) for i in order if np.isfinite(total[i])], None
    except Exception as e:  # one odd molecule keeps an empty PubChem part
        return m, [], f"{type(e).__name__}: {str(e)[:120]}"


def library_hits(test: pd.DataFrame, train_path, p, hide: dict | None = None) -> dict:
    """molecule -> {inchikey14: (best cosine, smiles)} for train.parquet structures whose spectra (same adduct,
    precursor within lib_tol) reach lib_hit against a query spectrum. Two passes, small memory."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import edit_overlay as eo
    pf = pq.ParquetFile(str(train_path))
    meta = pf.read(columns=["adduct", "precursor_mz", "inchikey14", "normalized_smiles", "ingest_lib"]).to_pandas()
    meta["row"] = np.arange(len(meta))
    if p.get("exclude_libs"):
        meta = meta[~meta.ingest_lib.isin(set(p["exclude_libs"]))]
    by_add = {a: g.sort_values("precursor_mz") for a, g in meta.groupby("adduct", sort=False)}
    ik = dict(zip(meta.row, meta.inchikey14)); smi = dict(zip(meta.row, meta.normalized_smiles))
    want = {}                                     # train row -> list of (molecule, query index)
    queries = []
    for r in test.itertuples(index=False):
        q = eo._prepare(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz), 60, 0.01)
        if not len(q[0]):
            continue
        g = by_add.get(str(r.adduct))
        if g is None:
            continue
        pm = g.precursor_mz.to_numpy()
        a, b = np.searchsorted(pm, float(r.precursor_mz) - p["lib_tol"]), np.searchsorted(pm, float(r.precursor_mz) + p["lib_tol"])
        if b <= a:
            continue
        qi = len(queries); queries.append((str(r.molecule_id), q))
        rows = g.row.to_numpy()[a:b]
        if len(rows) > p["lib_cap"]:
            rows = rows[np.linspace(0, len(rows) - 1, p["lib_cap"]).astype(int)]
        for t in rows:
            want.setdefault(int(t), []).append(qi)
    hits = {}
    if not want:
        return hits
    starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
    for gi in range(pf.num_row_groups):
        lo_, hi_ = starts[gi], starts[gi + 1]
        rows = sorted(t for t in want if lo_ <= t < hi_)
        if not rows:
            continue
        tab = pf.read_row_group(gi, columns=["precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]).take(pa.array([t - lo_ for t in rows])).to_pydict()
        for t, pm, mz, it in zip(rows, tab["precursor_mz"], tab["ms2_mzs"], tab["ms2_normalized_intensities"]):
            if mz is None or it is None or pm is None:
                continue
            ref = eo._prepare(mz, it, float(pm), 60, 0.01)
            if not len(ref[0]):
                continue
            for qi in want[t]:
                m, q = queries[qi]
                if hide and ik[t] in hide.get(m, ()):
                    continue
                c, n = eo._modcos((0.0, q[0], q[1]), (0.0, ref[0], ref[1]), 0.01)
                if n >= p["lib_min_matches"] and c >= p["lib_hit"]:
                    best = hits.setdefault(m, {}).get(ik[t], (0.0, ""))
                    if c > best[0]:
                        hits[m][ik[t]] = (float(c), smi[t])
    return hits


class Spec2Mol:
    """spec2mol: spectra and fingerprints (the fold-safe FPNets' bit selection) embedded in one space; a candidate's
    score is the mean over the molecule's spectra of cosine x the model's scale."""

    def __init__(self, bundle, bits_path, device="cpu"):
        import sys as _sys
        import torch
        import edit_overlay as eo
        _sys.path.insert(0, str(bundle))
        from spec2mol_features import spectrum
        from spec2mol_model import SpecMol
        ck = torch.load(Path(bundle) / "spec2mol.pt", map_location=device, weights_only=False)
        self.model = SpecMol(**ck["cfg"]); self.model.load_state_dict(ck["model"]); self.model.eval()
        self.torch, self.spectrum, self.scale = torch, spectrum, float(self.model.scale())
        self.chem = eo.Chemistry(np.load(bits_path))

    def encode(self, g: pd.DataFrame):
        """The molecule's spectra (rows of test.parquet) embedded, or None."""
        feats = [self.spectrum(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz, r.adduct, r.ionization_mode,
                               r.collision_energy_ev) for r in g.itertuples(index=False)]
        feats = [f for f in feats if f is not None]
        if not feats:
            return None
        t = self.torch
        with t.inference_mode():
            z, _ = self.model.encode_spectra(t.tensor(np.array([f[0] for f in feats], np.float32)), t.tensor([f[1] for f in feats]),
                                             t.tensor(np.array([f[2] for f in feats])), t.tensor([f[3] for f in feats]))
        return z

    def score(self, z, smiles: list) -> np.ndarray:
        """Scores of the candidates for embedded spectra z (NaN where a candidate has no fingerprint)."""
        out = np.full(len(smiles), np.nan)
        fps = [self.chem.fingerprint(s_) for s_ in smiles]
        ok = [i for i, f in enumerate(fps) if f is not None]
        if z is None or not ok:
            return out
        with self.torch.inference_mode():
            me = self.model.encode_molecules(self.torch.tensor(np.array([fps[i] for i in ok], np.float32)))
            out[ok] = ((z @ me.T).mean(0) * self.scale).numpy()
        return out


def s2m_pooled_scale(vs: dict, min_n: int, fallback: float) -> float:
    """Median within-shortlist sd of the spec2mol scores over molecules with >= min_n scores. One scale for every
    molecule: a shortlist's own sd inflates small differences on short lists (2-5 candidates), which cost the
    panel's class-1 view 0.018 against 0.003 with the pooled scale."""
    sds = [float(np.std(v[np.isfinite(v)])) for v in vs.values() if np.isfinite(v).sum() >= min_n]
    return float(np.median(sds)) if len(sds) >= 20 else float(fallback)


def s2m_fuse(sc: dict, v: np.ndarray, gamma: float, scale: float) -> dict:
    """{key: (score, smiles)} plus gamma x (v - mean) / scale for the spec2mol scores v (in sc's order); candidates
    without a score count as the lowest scored. Unchanged with fewer than two scores."""
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return sc
    mu, sd = v[ok].mean(), scale + 1e-9
    zz = np.where(ok, (v - mu) / sd, (v[ok].min() - mu) / sd)
    return {k: (val + gamma * float(a), s_) for (k, (val, s_)), a in zip(sc.items(), zz)}


def neural_rerank(rows_out: list, test: pd.DataFrame, p: dict, report: dict) -> list:
    """Re-order each list's first neural_top candidates by reciprocal-rank fusion with the neural scorer."""
    import sys as _sys
    t0 = time.time()
    try:
        _sys.path.insert(0, str(p["neural_bundle"]))
        from inference import Scorer
        scorer = Scorer(p["neural_bundle"], device=p["device"])
    except Exception as e:                    # without the scorer the lists stay as they are
        report["neural_error"] = f"{type(e).__name__}: {str(e)[:150]}"
        return rows_out
    by_mol = {str(m): g for m, g in test.groupby("molecule_id", sort=False)}
    out, changed = [], 0
    for m, smi in rows_out:
        try:
            cand = smi.split(";")
            g = by_mol.get(m)
            if g is None or len(cand) < 2:
                out.append((m, smi)); continue
            spectra = []
            for r in g.itertuples(index=False):
                ce = np.atleast_1d(np.asarray(r.collision_energy_ev, dtype=float)) if r.collision_energy_ev is not None else np.zeros(0)
                spectra.append(dict(mzs=np.asarray(r.ms2_mzs, float), intensities=np.asarray(r.ms2_normalized_intensities, float),
                                    precursor_mz=float(r.precursor_mz), adduct=str(r.adduct), ionization_mode=str(r.ionization_mode),
                                    collision_energy=float(np.nanmean(ce)) if len(ce) and np.isfinite(ce).any() else None))
            head, tail = cand[: p["neural_top"]], cand[p["neural_top"]:]
            sc = np.asarray(scorer.score_candidates(spectra, head), dtype=np.float64)
            if not np.isfinite(sc).any():
                out.append((m, smi)); continue
            nrank = np.empty(len(head)); nrank[np.argsort(-np.where(np.isfinite(sc), sc, -1e9), kind="stable")] = np.arange(1, len(head) + 1)
            fused = 1.0 / (p["neural_offset"] + np.arange(1, len(head) + 1)) + p["neural_weight"] / (p["neural_offset"] + nrank)
            order = np.argsort(-fused, kind="stable")
            changed += int((order != np.arange(len(head))).any())
            out.append((m, ";".join([head[i] for i in order] + tail)))
        except Exception as e:                # one odd molecule keeps its order
            report["errors"] += 1
            if len(report["error_samples"]) < 3:
                report["error_samples"].append(f"neural {m}: {type(e).__name__}: {str(e)[:100]}")
            out.append((m, smi))
    report["neural_changed"] = changed; report["neural_seconds"] = round(time.time() - t0, 1)
    return out


def run(test_path, train_path, index_root, foldsafe_dir, v4_code_dir, output_path, report_path=None, p=None,
        hide_lib: dict | None = None, logits_fn=None, doc_root=None, dump_path=None, pub_cache=None) -> dict:
    """Write a submission (molecule_id, smiles) from C1. logits_fn / hide_lib / dump_path / pub_cache: offline checks
    only (dump_path pickles the PubChem lists and library hits; pub_cache reuses a dump's PubChem lists)."""
    import edit_overlay as eo
    t0 = time.time()
    p = {**PARAMS, **(p or {})}
    test = pd.read_parquet(test_path)
    ids = test.molecule_id.drop_duplicates().astype(str).tolist()
    if logits_fn is None:
        v4 = eo._load_v4(Path(v4_code_dir))
    else:                                         # cached logits (offline checks): only the adduct table is needed
        import importlib.util
        spec = importlib.util.spec_from_file_location("_c1_v4chem", Path(v4_code_dir) / "chem.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        v4 = {"chem": mod}
    spectra, target = eo.molecule_spectra(test, v4)
    report = dict(molecules=len(ids), with_spectra=len(spectra), pubchem_lists=0, library_molecules=0, library_hits=0,
                  errors=0, error_samples=[], stopped_early=False)
    zs = {}
    if logits_fn is None:
        paths = sorted(str(x) for x in Path(foldsafe_dir).glob("fpnet_fold*.pt"))
        bank = v4["fpnet"].ModelBank(paths, device=p["device"])
        report["models"] = len(paths)
    for m in spectra:
        try:
            zs[m] = logits_fn(m) if logits_fn else eo.query_logits(bank, v4, spectra[m])
        except Exception as e:
            report["errors"] += 1
            if len(report["error_samples"]) < 3:
                report["error_samples"].append(f"logits {m}: {type(e).__name__}: {str(e)[:100]}")
    report["logits_seconds"] = round(time.time() - t0, 1)
    tasks = [(m, target[m], zs[m], p) for m in ids if m in zs and m in target]
    pub = dict(pub_cache) if pub_cache else {}
    tasks = [t for t in tasks if t[0] not in pub]
    bits = Path(foldsafe_dir) / "fp_bits.npy"
    deadline = t0 + p["max_seconds"]
    np_data, member_path = None, ""
    if (p["np_table"] or p["np_coconut"]) and (p["np_beta"] or p["np_delta"] or p["np_window"]):
        try:
            np_data = load_np(p)
            report["np_table_keys"], report["np_coconut_keys"], report["np_member_hashes"] = len(np_data[0]), len(np_data[1]), len(np_data[2])
            if p["np_window"]:
                member_path = str(Path(output_path).with_name("np_member_hashes.npy"))
                np.save(member_path, np_data[2])
        except Exception as e:                 # without the tables the lists are made as without the prior
            report["np_error"] = f"{type(e).__name__}: {str(e)[:150]}"
    import multiprocessing as mp
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(p["workers"], initializer=_init_worker, initargs=(str(index_root), str(doc_root or index_root), str(bits), member_path)) as pool:
            it = pool.imap_unordered(_score_window, tasks, chunksize=1)
            for _ in tasks:
                left = deadline - time.time()
                if left <= 0:
                    report["stopped_early"] = True
                    break
                try:
                    m, rows, err = it.next(timeout=left)
                except mp.TimeoutError:
                    report["stopped_early"] = True
                    break
                pub[m] = rows
                if err:
                    report["errors"] += 1
                    if len(report["error_samples"]) < 3:
                        report["error_samples"].append(f"{m}: {err}")
    except Exception as e:                     # pool unavailable: same work in this process
        report["pool_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        _init_worker(str(index_root), str(doc_root or index_root), str(bits), member_path)
        for t in tasks:
            if t[0] in pub:
                continue
            if time.time() > deadline:
                report["stopped_early"] = True
                break
            m, rows, err = _score_window(t)
            pub[m] = rows
    report["pubchem_seconds"] = round(time.time() - t0, 1)
    hits = {}
    try:
        hits = library_hits(test, train_path, p, hide_lib)
    except Exception as e:
        report["library_error"] = f"{type(e).__name__}: {str(e)[:150]}"
    report["library_seconds"] = round(time.time() - t0, 1)
    if dump_path:
        import pickle
        pickle.dump(dict(pub=pub, hits=hits, ids=ids), open(dump_path, "wb"))
    tier = eo.Tier(Path(index_root), Path(doc_root or index_root))
    chem = eo.Chemistry(np.load(bits))
    sample_fill = "CCO"
    s2m, by_mol = None, {}
    if p["s2m_bundle"] and p["s2m_gamma"] > 0:
        try:
            s2m = Spec2Mol(p["s2m_bundle"], bits, p["device"])
            by_mol = {str(m): g for m, g in test.groupby("molecule_id", sort=False)}
            report["s2m_molecules"] = 0
        except Exception as e:                 # without spec2mol the lists stay as C1 makes them
            report["s2m_error"] = f"{type(e).__name__}: {str(e)[:150]}"
    scs = {}                                      # molecule -> {metric key: (score, smiles)}
    std = {}                                      # metric key -> standard InChIKey first blocks seen for it
    hit_keys = {}                                 # molecule -> metric keys with a library hit
    for m in ids:
        try:
            # a library hit adds lib_beta * (cos - lib_c0) to its candidate's score (a hit missing from the PubChem
            # list starts at the list's lowest score). Panel (experiments/c1_replay.py): beta 4, c0 0.8 -> class 2
            # 0.825 -> 0.815 (false hits), class-1 view 0.825 -> 0.877; putting hits >= 0.75 first: 0.716 / 0.926
            sc = {}
            for r, v in pub.get(m, []):
                s_ = tier.smiles(r); k = chem.key(s_)
                if k:
                    std.setdefault(k, set()).add(bytes(tier.ik[r]).decode())
                if k and k not in sc:
                    sc[k] = (v, s_)
            base = min(v for v, _ in sc.values()) if sc else 0.0
            if m in hits:
                report["library_molecules"] += 1; report["library_hits"] += len(hits[m])
            for hik, (c, s_) in hits.get(m, {}).items():
                k = chem.key(s_)
                if k:
                    std.setdefault(k, set()).add(str(hik)[:14]); hit_keys.setdefault(m, set()).add(k)
                    bonus = p["lib_beta"] * max(0.0, c - p["lib_c0"])
                    sc[k] = (sc[k][0] + bonus, sc[k][1]) if k in sc else (base + bonus, s_)
            scs[m] = sc
            report["pubchem_lists"] += int(bool(pub.get(m)))
        except Exception as e:
            report["errors"] += 1
            if len(report["error_samples"]) < 3:
                report["error_samples"].append(f"assemble {m}: {type(e).__name__}: {str(e)[:100]}")
    if s2m is not None:                           # spec2mol over each whole shortlist (experiments/spec2mol_lowdoc.py)
        vs = {}
        for m, sc in scs.items():
            if len(sc) > 1 and m in by_mol:
                try:
                    vs[m] = s2m.score(s2m.encode(by_mol[m]), [s_ for _, s_ in sc.values()])
                except Exception as e:         # this molecule keeps C1's order
                    report["errors"] += 1
                    if len(report["error_samples"]) < 3:
                        report["error_samples"].append(f"s2m {m}: {type(e).__name__}: {str(e)[:100]}")
        scale = p["s2m_scale"] or s2m_pooled_scale(vs, p["s2m_min_n"], p["s2m_fallback"])
        report["s2m_scale"] = round(scale, 4)
        for m, v in vs.items():
            scs[m] = s2m_fuse(scs[m], v, p["s2m_gamma"], scale)
            report["s2m_molecules"] += 1
        report["s2m_seconds"] = round(time.time() - t0, 1)
    if np_data is not None and (p["np_beta"] or p["np_delta"]):
        try:
            refs, coco = np_data[0], np_data[1]
            report["np_members"] = 0
            for m, sc in scs.items():
                hk = hit_keys.get(m, set()) if p["np_hits"] else set()
                for k in sc:
                    if k in refs or std.get(k, set()) & coco or k in hk:
                        add = p["np_beta"] + p["np_delta"] * float(np.log1p(refs.get(k, 0.0)))
                        sc[k] = (sc[k][0] + add, sc[k][1]); report["np_members"] += 1
        except Exception as e:                 # without the tables the lists stay as they are
            report["np_error"] = f"{type(e).__name__}: {str(e)[:150]}"
    rows_out = []
    for m in ids:
        final = [s_ for _, (v, s_) in sorted(scs.get(m, {}).items(), key=lambda x: -x[1][0])][: p["top_k"]]
        rows_out.append((m, ";".join(final) if final else sample_fill))
    if p["neural_bundle"]:
        rows_out = neural_rerank(rows_out, test, p, report)
    out = pd.DataFrame(rows_out, columns=["molecule_id", "smiles"])
    assert out.molecule_id.tolist() == ids and out.smiles.str.split(";").map(len).between(1, 25).all()
    out.to_csv(output_path, index=False)
    report["seconds"] = round(time.time() - t0, 1)
    report["error_samples"] = report["error_samples"] or None
    if report_path:
        Path(report_path).write_text(json.dumps(dict(report, params=p), indent=2))
    return report
