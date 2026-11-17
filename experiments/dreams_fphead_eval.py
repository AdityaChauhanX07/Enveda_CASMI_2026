"""Panel check of the DreaMS fingerprint head (tools/build_dreams_fphead.py): candidate score = Bernoulli log-likelihood of
the candidate's Morgan bits (radius 2, 2,048) under the head's predicted logits, averaged over the molecule's
positive-mode spectra (all spectra if it has none). Reported per list file: the head alone, the list order +
kappa x zd(head score), and the full overlay (analogue support, final rule) + kappa x zd(head score).

usage: dreams_fphead_eval.py FPHEAD_DIR PANEL_EMBED_DIR NEIGHBOURS.json.gz LISTS.csv [DUMP.pkl] [--view=hidden]
       [--kappas=0,0.25,0.5,1,1.5] [--lam=1.5]
"""
from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dreams_local as dl  # noqa: E402


def bits(s_):
    m = Chem.MolFromSmiles(str(s_))
    if m is None:
        return None
    a = np.zeros(2048, np.float32); a[list(dl.GEN.GetFingerprint(m).GetOnBits())] = 1.0
    return a


def zd(v, eps=0.01):
    return (v - v.mean()) / (v.std() + eps)


def main() -> None:
    head_dir, emb_dir, nb_path, lists_path = (Path(a) for a in sys.argv[1:5])
    dump = Path(sys.argv[5]) if len(sys.argv) > 5 and not sys.argv[5].startswith("--") else None
    args = dict(a[2:].split("=", 1) for a in sys.argv[5:] if a.startswith("--") and "=" in a)
    view, lam = args.get("view", "hidden"), float(args.get("lam", 1.5))
    kappas = [float(x) for x in args.get("kappas", "0,0.25,0.5,1,1.5").split(",")]
    L = np.load(head_dir / "panel_logits.npy").astype(np.float32)
    rows = pd.read_parquet(emb_dir / "panel_rows.parquet")
    lp, ln = -np.logaddexp(0, -L), -np.logaddexp(0, L)          # log sigmoid(l), log sigmoid(-l)
    by_m = {}
    for i, (m, md) in enumerate(zip(rows.m, rows["mode"])):
        by_m.setdefault(m, []).append((i, md))
    use = {m: [i for i, md in v if md == "pos"] or [i for i, _ in v] for m, v in by_m.items()}
    sub = pd.read_csv(lists_path, dtype=str, keep_default_na=False)
    lists = {m: s.split(";") for m, s in zip(sub.molecule_id, sub.smiles)}
    keys = dl._key_cache([s_ for c in lists.values() for s_ in c], dl.metric_key)
    bit_cache = {}
    head = {}
    for m, cand in lists.items():
        if m not in use:
            continue
        for s_ in cand:
            if s_ not in bit_cache:
                bit_cache[s_] = bits(s_)
        B = np.stack([bit_cache[s_] if bit_cache[s_] is not None else np.full(2048, np.nan, np.float32) for s_ in cand])
        sc = np.mean([B @ lp[i] + (1 - B) @ ln[i] for i in use[m]], axis=0)
        head[m] = np.where(np.isfinite(sc), sc, np.nanmin(sc) if np.isfinite(sc).any() else 0.0)
    heads, ok = {}, None
    if dump is not None:
        hits = pickle.load(open(dump, "rb"))["hits"]
        ok = {m: {dl.std_ik(s_) for c, s_ in hits.get(m, {}).values() if c >= 0.8} for m in lists}
        for m, cand in lists.items():
            ks = {s_ for c, s_ in hits.get(m, {}).values() if c >= 0.95}
            if ks:
                heads[m] = next((i for i, c in enumerate(cand) if c not in ks), len(cand))
    nbd = json.loads(gzip.decompress(nb_path.read_bytes()))
    dl.MODES = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "dreams" / "panel_modes.json").read_text())
    sup = dl.supports(nbd, lists, view=view, noself=True, selfok=ok)

    def rr(m, lst):
        k = [keys[s_] for s_ in lst[:25]]
        return 1.0 / (k.index(m) + 1) if m in k else 0.0

    def order(m, score):
        cand = lists[m]
        new = [cand[i] for i in np.argsort(-score, kind="stable")]
        h = heads.get(m, 0)
        return cand[:h] + [c for c in new if c not in set(cand[:h])] if h else new
    ms = [m for m in lists if m in head and m in sup and len(lists[m]) > 1]
    base = {m: 1.0 / (3.0 + np.arange(1, len(lists[m]) + 1)) for m in ms}
    alone = np.mean([rr(m, order(m, head[m])) for m in ms])
    print(f"{lists_path.name} | n {len(ms)} | list order {np.mean([rr(m, lists[m]) for m in ms]):.4f} | head alone {alone:.4f}")
    for with_sup in (False, True):
        res = []
        for k in kappas:
            v = [rr(m, order(m, base[m] / base[m].std() + (lam * zd(sup[m]) if with_sup else 0.0) + k * zd(head[m]))) for m in ms]
            res.append(f"{k:g}: {np.mean(v):.4f}")
        print(("  + analogue overlay (lambda %g) + kappa x head | " % lam if with_sup else "  list order + kappa x head          | ") + " | ".join(res))


if __name__ == "__main__":
    main()
