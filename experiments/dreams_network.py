"""DreaMS in-test network: do structurally related test molecules (by DreaMS embedding) help rank each other's lists?

Like the public N-1c rerank (matched-peak cosine between test molecules), but the links come from DreaMS embedding
similarity, which reflects structure more than raw peak overlap. Each panel molecule (the panel stands in for the test
set) gets an embedding = mean of its positive-mode spectra's DreaMS embeddings (all spectra if it has no positive one).
Molecules with cosine >= tau are linked; a candidate's network support is the sum over linked molecules of cos^4 x
the 1/rank-weighted mean Tanimoto to that molecule's top-5 candidates. Lists are re-ranked by
1/(3 + rank) / sd + lambda x zd(MassSpecGym analogue support) + mu x zd(network support), zd = damped z
(as casmi/dreams_overlay.py; analogue support with own-spectrum exclusion, library-hit exemptions, positive mode).

usage: dreams_network.py PANEL_EMBED_DIR NEIGHBOURS.json.gz LISTS.csv DUMP.pkl [--lam=1.5] [--eps=0.01]
       [--taus=0.7,0.8,0.9] [--mus=0,0.25,0.5,1]
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dreams_local as dl  # noqa: E402


def zd(v, eps):
    return (v - v.mean()) / (v.std() + eps)


def main() -> None:
    emb_dir, nb_path, lists_path, dump_path = (Path(a) for a in sys.argv[1:5])
    args = dict(a[2:].split("=", 1) for a in sys.argv[5:] if a.startswith("--") and "=" in a)
    lam, eps = float(args.get("lam", 1.5)), float(args.get("eps", 0.01))
    taus = [float(x) for x in args.get("taus", "0.7,0.8,0.9").split(",")]
    mus = [float(x) for x in args.get("mus", "0,0.25,0.5,1").split(",")]
    Q = np.load(emb_dir / "panel_embeddings.npy").astype(np.float32)
    rows = pd.read_parquet(emb_dir / "panel_rows.parquet")
    sub = pd.read_csv(lists_path, dtype=str, keep_default_na=False)
    lists = {m: s.split(";") for m, s in zip(sub.molecule_id, sub.smiles)}
    # molecule embeddings (positive-mode spectra when available)
    mol_emb = {}
    for m, g in rows.groupby("m", sort=False):
        idx = g.index[g["mode"] == "pos"] if (g["mode"] == "pos").any() else g.index
        e = Q[idx].mean(0); mol_emb[m] = e / (np.linalg.norm(e) + 1e-9)
    ms = [m for m in lists if m in mol_emb]
    E = np.stack([mol_emb[m] for m in ms]); S = E @ E.T; np.fill_diagonal(S, -1.0)
    print(f"molecules {len(ms)} | pairwise DreaMS cosine quantiles 50/90/99/99.9: "
          + " / ".join(f"{q:.3f}" for q in np.quantile(S[np.triu_indices(len(ms), 1)], [.5, .9, .99, .999])))
    # MassSpecGym analogue support (final overlay rule)
    nbd = json.loads(gzip.decompress(nb_path.read_bytes()))
    dl.MODES = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "dreams" / "panel_modes.json").read_text())
    hits = pickle.load(open(dump_path, "rb"))["hits"]
    ok = {m: {dl.std_ik(s_) for c, s_ in hits.get(m, {}).values() if c >= 0.8} for m in lists}
    sup = dl.supports(nbd, lists, view="hidden", noself=True, selfok=ok)
    heads = {}
    for m, cand in lists.items():                       # library-first leads stay on top
        ks = {s_ for c, s_ in hits.get(m, {}).values() if c >= 0.95}
        if ks:
            heads[m] = next((i for i, c in enumerate(cand) if c not in ks), len(cand))
    keys = dl._key_cache([s_ for c in lists.values() for s_ in c], dl.metric_key)
    fps = {s_: dl.fp(s_) for c in lists.values() for s_ in c}

    def rr(m, lst):
        k = [keys[s_] for s_ in lst[:25]]
        return 1.0 / (k.index(m) + 1) if m in k else 0.0
    pw = 1.0 / np.arange(1, 6); pw /= pw.sum()
    for tau in taus:
        net, n_linked = {}, 0
        for a, m in enumerate(ms):
            links = [(ms[b], float(S[a, b])) for b in np.flatnonzero(S[a] >= tau)]
            n_linked += bool(links)
            v = np.zeros(len(lists[m]))
            for m2, c in links:
                top = [fps[s_] for s_ in lists[m2][:5] if fps[s_] is not None]
                if not top:
                    continue
                for i, s_ in enumerate(lists[m]):
                    f = fps[s_]
                    if f is not None:
                        v[i] += c ** 4 * float(np.dot(pw[:len(top)] / pw[:len(top)].sum(), DataStructs.BulkTanimotoSimilarity(f, top)))
            net[m] = v
        res = []
        for mu in mus:
            tot = []
            for m in ms:
                cand = lists[m]
                if len(cand) < 2 or m not in sup:
                    tot.append(rr(m, cand)); continue
                base = 1.0 / (3.0 + np.arange(1, len(cand) + 1))
                score = base / base.std() + lam * zd(sup[m], eps) + (mu * zd(net[m], eps) if net[m].any() else 0.0)
                new = [cand[i] for i in np.argsort(-score, kind="stable")]
                h = heads.get(m, 0)
                if h:
                    new = cand[:h] + [c for c in new if c not in set(cand[:h])]
                tot.append(rr(m, new))
            res.append(f"mu {mu:g}: {np.mean(tot):.4f}")
        print(f"  tau {tau:g} (linked molecules {n_linked}) | lambda {lam:g} | " + " | ".join(res), flush=True)


if __name__ == "__main__":
    main()
