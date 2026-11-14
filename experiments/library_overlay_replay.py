"""Library overlay + edit overlay vs the edit overlay alone, from panel dumps (per molecule, both halves).

Dumps: edit_overlay_panel.py --library=tau:down:min_ref --dump_state=artifacts/state_t8x (lists after the library
step) and the a7 dumps artifacts/state_t7_* (no library step). The class-2 change is measured against the original
fork-like lists, so it includes the library step's own gain; the insertion step is replayed exactly
(insertion_replay.insert reproduces the overlay's pass 3).

usage: library_overlay_replay.py t8a [t8b ...]   (compares each with t7)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.edit_overlay import merge_sources  # noqa: E402
import insertion_replay as ir  # noqa: E402
from library_overlay_panel import panel_lists  # noqa: E402
from casmi.metric import ik14  # noqa: E402

N = 250


def merged(st):
    return dict(st, scored=[(g, k, m) for g, _, k, m in merge_sources([(g, None, k, m) for g, k, m in st["scored"]], 0.0)])


def per_molecule(tag: str, orig: dict) -> dict:
    c2 = pickle.load(open(ROOT / "artifacts" / f"state_{tag}_c2.pkl", "rb"))
    c3 = pickle.load(open(ROOT / "artifacts" / f"state_{tag}_c3.pkl", "rb"))
    out = {}
    for m in orig:
        r0 = ir.rr(orig[m], m)
        r2 = ir.rr(ir.insert(merged(c2[m]), {}), m) if m in c2 else r0
        r3 = ir.rr(ir.insert(merged(c3[m]), {}), m) if m in c3 else 0.0
        lib = ir.rr(c2[m]["cand_keys"], m) if m in c2 else r0
        out[m] = (r2 - r0, r3, lib - r0)
    return out


if __name__ == "__main__":
    _, _, sub = panel_lists()
    orig = {m: [ik14(s) for s in x.split(";")] for m, x in zip(sub.molecule_id, sub.smiles)}
    mols = list(orig); halves = [mols[0::2], mols[1::2]]
    rows = {t: per_molecule(t, orig) for t in ["t7"] + sys.argv[1:]}
    print("run | class 2: library step, total (hurt/helped) | class 3 | net 10 % / 25 % class 3 | halves at 25 %")
    for t, pm in rows.items():
        d2 = np.array([pm[m][0] for m in mols]); c3 = np.array([pm[m][1] for m in mols]); lib = np.array([pm[m][2] for m in mols])
        net = lambda s, idx=slice(None), n=N: ((1 - s) * d2[idx].sum() + s * c3[idx].sum()) / n
        hv = [net(0.25, np.isin(mols, h), N / 2) for h in halves]
        print(f"{t} | {lib.sum() / N:+.4f}, {d2.sum() / N:+.4f} ({(d2 < 0).sum()}/{(d2 > 0).sum()}) | {c3.sum() / N:.4f} | "
              f"{net(0.10):+.4f} / {net(0.25):+.4f} | {hv[0]:+.4f} / {hv[1]:+.4f}")
