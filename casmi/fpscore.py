"""Fold-safe FPNet scores for arbitrary structures on the panel: score = selected fingerprint bits . logits."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

V4G = Path(os.environ.get("CASMI_V4G", Path(__file__).resolve().parents[2] / "casmi_data" / "ext" / "v4g"))
_G = {}


def _init():
    g = rdFingerprintGenerator
    _G["gens"] = [g.GetMorganGenerator(radius=2, fpSize=4096), g.GetMorganGenerator(radius=3, fpSize=4096),
                  g.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=g.GetMorganFeatureAtomInvGen()),
                  g.GetRDKitFPGenerator(fpSize=2048, maxPath=6), g.GetAtomPairGenerator(fpSize=2048),
                  g.GetTopologicalTorsionGenerator(fpSize=2048)]
    _G["bits"] = np.load(V4G / "casmi26-v2-pool" / "fp_bits.npy")


def fingerprint(smiles: str):
    """The FPNet's 10,226 selected bits (float32 0/1) or None."""
    if not _G:
        _init()
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    parts = [gen.GetFingerprintAsNumPy(m).astype(np.float32) for gen in _G["gens"]]
    parts.append(np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32))
    return np.concatenate(parts)[_G["bits"]]
