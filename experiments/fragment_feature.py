"""Does fragment explanation (casmi/fragments.py) separate the answer from its isomers in C1's shortlists?

Every candidate of the C1 dumps' shortlists (artifacts/c1_rows_v1.pkl keys; SMILES from the index rows and the library
hits) is fragmented, and its explained share of each panel spectrum's top peaks is averaged over the molecule's
spectra (cached in artifacts/frag_scores_v1.json). Then C1e's ranking (w_doc 0, bonus 5 for members or hits, library-hit
weight 12, spec2mol 0.75 pooled) is replayed with + lambda x explained share, and with lambda x share / pooled sd.

usage: fragment_feature.py NATURAL_PRODUCTS_PARQUET COCONUT_CSV
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parents[0] / "casmi_data" / "ext"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import spec2mol_lowdoc as s2  # noqa: E402
from casmi.edit_overlay import Tier  # noqa: E402
from casmi.fragments import Fragmenter, explained, peaks  # noqa: E402

_JOB = {}


def _score(m):
    frag = Fragmenter()
    specs, cands = _JOB["specs"][m], _JOB["cands"][m]
    out = {}
    for k, smi in cands.items():
        vals = []
        for mode, mz, w in specs:
            vals.append(explained(frag.ion_masses(smi, mode), mz, w))
        out[k] = float(np.mean(vals)) if vals else 0.0
    return m, out


if __name__ == "__main__":
    art = ROOT / "artifacts"
    data = s2.split_rows(art)
    ids = data["ids"]
    cache = art / "frag_scores_v1.json"
    if cache.is_file():
        frag_scores = json.load(open(cache))
    else:
        tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
        dumps = {t: pickle.load(open(art / f"c1_dump_{t}_v1.pkl", "rb")) for t in ("c2", "c1")}
        cands = {}
        for m in ids:
            c = {}
            for (k, _, _), (r, _) in zip(data["rows"][m], dumps["c2"]["pub"].get(m, [])):
                if k and k not in c:
                    c[k] = tier.smiles(r)
            for t, d in dumps.items():
                for (k, _), (_, (cc, s_)) in zip(data["hits"][t][m], d["hits"].get(m, {}).items()):
                    if k and k not in c:
                        c[k] = s_
            cands[m] = c
        test = pd.read_parquet(art / "np_test.parquet")
        from casmi.metric import ik14
        test["m"] = [ik14(s_) for s_ in test.normalized_smiles]
        specs = {}
        for r in test.itertuples():
            mz, w = peaks(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
            if len(mz):
                specs.setdefault(r.m, []).append((r.ionization_mode, mz, w))
        _JOB.update(specs=specs, cands=cands)
        with mp.get_context("fork").Pool(4) as pool:
            frag_scores = dict(pool.map(_score, [m for m in ids if m in specs], chunksize=2))
        json.dump(frag_scores, open(cache, "w"))
    # answer vs isomer-decoy shares
    a = [frag_scores[m][m] for m in ids if m in frag_scores and m in frag_scores[m]]
    dec = [v for m in ids if m in frag_scores for k, v in frag_scores[m].items() if k != m]
    print(f"explained share: answers mean {np.mean(a):.3f} (n {len(a)}), decoys mean {np.mean(dec):.3f}")
    best = [m for m in ids if m in frag_scores and m in frag_scores[m] and frag_scores[m][m] >= max(frag_scores[m].values()) - 1e-9]
    print(f"answer has the highest share in its shortlist: {len(best)} of {len(a)}")
    # replay C1e + fragments
    ln = set(pd.read_parquet(sys.argv[1], columns=["scorer_key"]).scorer_key)
    co = set(pd.read_csv(sys.argv[2], usecols=["standard_inchi_key"]).standard_inchi_key.dropna().str.slice(0, 14))
    tier = Tier(EXT / "pubchem_prior" / "casmi26-pubchem-tier", EXT / "pubchem_prior")
    dumps = {t: pickle.load(open(art / f"c1_dump_{t}_v1.pkl", "rb")) for t in ("c2", "c1")}
    std = {}
    for m in ids:
        for (k, _, _), (r, _) in zip(data["rows"][m], dumps["c2"]["pub"].get(m, [])):
            if k:
                std.setdefault(k, set()).add(bytes(tier.ik[r]).decode())
        for t, d in dumps.items():
            for (k, _), (hik, _) in zip(data["hits"][t][m], d["hits"].get(m, {}).items()):
                if k:
                    std.setdefault(k, set()).add(str(hik)[:14])
    member = lambda k: k in ln or bool(std.get(k, set()) & co)
    scores = json.load(open(art / "spec2mol_scores.json"))
    fsd = float(np.median([np.std(list(v.values())) for v in frag_scores.values() if len(v) >= 5]))
    print(f"pooled within-shortlist sd of the explained share: {fsd:.3f}")
    for tag in ("c2", "c1"):
        base = {}
        for m in ids:
            sc = {}
            for k, z, doc in data["rows"][m]:
                if k and k not in sc:
                    sc[k] = z
            b0 = min(sc.values()) if sc else 0.0
            hk = set()
            for k, c in data["hits"][tag][m]:
                if k and c >= 0.8:
                    sc[k] = sc.get(k, b0) + 12.0 * (c - 0.8); hk.add(k)
            base[m] = {k: v + (5.0 if (member(k) or k in hk) else 0.0) for k, v in sc.items()}
        scale = s2.pooled_scale({m: list(base[m]) for m in ids}, scores)
        out = []
        for lam in (0.0, 0.25, 0.5, 1.0, 2.0):          # in units of the pooled sd of the share
            rr = 0.0
            for m in ids:
                fs = frag_scores.get(m, {})
                sc = {k: v + lam * fs.get(k, 0.0) / fsd for k, v in base[m].items()}
                rr += s2.rr(s2.fuse_scores(sc, scores.get(m), 0.75, scale)[:25], m)
            out.append(f"{lam}: {rr / len(ids):.4f}")
        print(f"{tag} (C1e settings) + lambda x share / sd -> MRR | " + " | ".join(out), flush=True)
