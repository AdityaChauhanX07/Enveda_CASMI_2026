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
              neural_bundle="", neural_weight=0.25, neural_offset=3.0, neural_top=25)
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


def _init_worker(index_root, doc_root, bits_path):
    import edit_overlay as eo
    _one_thread(_W)
    _W["tier"] = eo.Tier(Path(index_root), Path(doc_root))
    _W["chem"] = eo.Chemistry(np.load(bits_path))


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
        if len(rows) > p["max_window"]:
            keep = np.sort(np.argsort(-doc, kind="stable")[: p["max_window"]])
            rows, doc = rows[keep], doc[keep]
        # stage 1: ECFP4-block score of every row; stage 2: full fingerprints for the n_partial best by it plus the
        # n_docpre best-documented (panel: all answers kept at 2,000 + 400 while fingerprinting 27 % of the rows)
        z_e = z[chem.sel_e]
        part = np.array([chem.partial(tier.smiles(int(r)), z_e) for r in rows])
        sel = set(np.argsort(-part, kind="stable")[: p["n_partial"]].tolist()) | set(np.argsort(-doc, kind="stable")[: p["n_docpre"]].tolist())
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
        order = np.argsort(-total, kind="stable")[: 4 * p["top_k"]]
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
    import multiprocessing as mp
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(p["workers"], initializer=_init_worker, initargs=(str(index_root), str(doc_root or index_root), str(bits))) as pool:
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
        _init_worker(str(index_root), str(doc_root or index_root), str(bits))
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
    rows_out = []
    for m in ids:
        try:
            # a library hit adds lib_beta * (cos - lib_c0) to its candidate's score (a hit missing from the PubChem
            # list starts at the list's lowest score). Panel (experiments/c1_replay.py): beta 4, c0 0.8 -> class 2
            # 0.825 -> 0.815 (false hits), class-1 view 0.825 -> 0.877; putting hits >= 0.75 first: 0.716 / 0.926
            sc = {}
            for r, v in pub.get(m, []):
                s_ = tier.smiles(r); k = chem.key(s_)
                if k and k not in sc:
                    sc[k] = (v, s_)
            base = min(v for v, _ in sc.values()) if sc else 0.0
            if m in hits:
                report["library_molecules"] += 1; report["library_hits"] += len(hits[m])
            for _, (c, s_) in hits.get(m, {}).items():
                k = chem.key(s_)
                if k:
                    bonus = p["lib_beta"] * max(0.0, c - p["lib_c0"])
                    sc[k] = (sc[k][0] + bonus, sc[k][1]) if k in sc else (base + bonus, s_)
            final = [s_ for _, (v, s_) in sorted(sc.items(), key=lambda x: -x[1][0])][: p["top_k"]]
            report["pubchem_lists"] += int(bool(pub.get(m)))
        except Exception as e:
            report["errors"] += 1
            final = []
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
