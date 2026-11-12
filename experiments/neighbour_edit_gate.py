"""Gate neighbour-edit products by the fold-safe FPNet: merge them into B's list by score with a bonus beta."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from casmi.data import DATA  # noqa: E402
from casmi.fpscore import fingerprint  # noqa: E402
from casmi.metric import ik14  # noqa: E402
from casmi.panel import load_candidates, tier_smiles, zscore  # noqa: E402
from neighbour_edit_oracle import EDITS  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from network_panel import shortlist  # noqa: E402

RDLogger.DisableLog("rdApp.*")
W_DOC = 0.4
REVERSE = {  # name: (mass delta, reaction removing the group from the heavier structure)
    "O-methyl": (14.01565, "[CH3][OX2:1][#6:2]>>[O:1][#6:2]"),
    "OH": (15.99491, "[OX2H1][#6:1]>>[#6:1]"),
    "hexose": (162.05282, "[O:1]C1OC(CO)C(O)C(O)C1O>>[O:1]"),
    "deoxyhexose": (146.05791, "[O:1]C1OC(C)C(O)C(O)C1O>>[O:1]"),
    "acetyl": (42.01057, "[O:1]C(C)=O>>[O:1]"),
}


def edit_products(mol, rxn, cap=400):
    out = {}
    for ps in rxn.RunReactants((mol,), maxProducts=cap):
        p = ps[0]
        try:
            Chem.SanitizeMol(p); s = Chem.MolToSmiles(p); out.setdefault(ik14(s), s)
        except Exception:
            continue
    return out


if __name__ == "__main__":
    c = load_candidates()
    c["z"] = zscore(c, "fold0_full")
    stats = c.groupby("molecule").fold0_full.agg(["mean", "std"])
    d = shortlist(c, c.z.to_numpy() + W_DOC * c.doc.to_numpy())
    d = d[d.rank0 < 26].reset_index(drop=True)
    d["smiles"] = tier_smiles(d.tier_row.to_numpy())
    d["key"] = [ik14(s) if Chem.MolFromSmiles(s) is not None else "" for s in d.smiles]
    lists = {m: g.reset_index(drop=True) for m, g in d.groupby("molecule", sort=False)}
    z = np.load(ROOT.parents[0] / "casmi_data" / "ext" / "panel" / "panel_logits.npz")
    logit = {k: z["fold0"][i].astype(np.float64) for i, k in enumerate(z["keys"])}
    t = pq.read_table(DATA / "train.parquet", columns=["ingest_lib", "normalized_smiles", "inchikey14"],
                      filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas().drop_duplicates("inchikey14")
    truth = {k: ik14(s) for k, s in zip(t.inchikey14, t.normalized_smiles)}
    mass = {k: rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(s)) for k, s in zip(t.inchikey14, t.normalized_smiles)}
    g = pd.read_parquet(ROOT / "artifacts" / "np_graph.parquet")
    e = g[(g.cosine >= 0.5) & (g.matches >= 4)]
    rev = {k: (dm_, AllChem.ReactionFromSmarts(sm)) for k, (dm_, sm) in REVERSE.items()}
    use_reverse = "--reverse" in sys.argv
    prods = {}
    for a, b in list(zip(e.a, e.b)) + list(zip(e.b, e.a)):
        if a not in lists or b not in lists or b not in logit:
            continue
        dm = mass[b] - mass[a]
        todo = [(delta, rxn) for delta, rxn in EDITS.values() if abs(dm - delta) <= 0.003]
        if use_reverse:
            todo += [(delta, rxn) for delta, rxn in rev.values() if abs(dm + delta) <= 0.003]
        for delta, rxn in todo:
            for smi in lists[a].smiles.tolist()[:3]:
                for k, s in edit_products(Chem.MolFromSmiles(smi), rxn).items():
                    fp = fingerprint(s)
                    if fp is None:
                        continue
                    zz = (fp @ logit[b] - stats.loc[b, "mean"]) / stats.loc[b, "std"]
                    prods.setdefault(b, {})[k] = max(prods.get(b, {}).get(k, -99), zz)
    print(f"molecules receiving products: {len(prods)}")

    def rr(keys, tr):
        for i, k in enumerate(keys[:25]):
            if k == tr:
                return 1.0 / (i + 1)
        return 0.0

    for beta in (2.0, 3.0, 4.0):
        g2 = g3 = 0.0
        for m, P in prods.items():
            L_ = lists[m]
            s_list = (L_.s0.to_numpy())
            base = L_.key.tolist()
            for drop_truth in (False, True):
                keep = [i for i, k in enumerate(base) if not (drop_truth and k == truth[m])]
                items = [(s_list[i], base[i]) for i in keep]
                merged = {k: s for s, k in items}
                for k, zz in P.items():
                    merged[k] = max(merged.get(k, -1e9), zz + beta) if k not in merged else merged[k] + 0.0
                order = [k for k, _ in sorted(merged.items(), key=lambda x: -x[1])]
                if "--top3" in sys.argv:
                    before_keys = [k for _, k in items]
                    ins = [k for k in order[:3] if k not in before_keys]
                    rest = [k for k in before_keys if k not in ins]
                    order = [k for k in order[:3] if k in ins or k in rest[:3]]
                    order = order + [k for k in rest if k not in order]
                before = [k for _, k in items]
                delta = rr(order, truth[m]) - rr(before, truth[m])
                if drop_truth:
                    g3 += delta
                else:
                    g2 += delta
        print(f"beta={beta}: class-2 change {g2 / 250:+.4f}, class-3 gain {g3 / 250:+.4f} | "
              f"net at 25% class 3 {0.75 * g2 / 250 + 0.25 * g3 / 250:+.4f}")
