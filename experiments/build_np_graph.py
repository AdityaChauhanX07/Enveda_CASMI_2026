"""Spectral graph of the enveda-np-examples molecules (modified cosine, same adduct), with structure labels.

Writes artifacts/np_graph.parquet: a, b (InChIKey14), cosine, matches, tanimoto (Morgan r2), dmass (neutral).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casmi.data import DATA  # noqa: E402
from casmi.network import build_spectra, molecule_graph, molecule_masses  # noqa: E402

RDLogger.DisableLog("rdApp.*")
COLS = ["ingest_lib", "normalized_smiles", "inchikey14", "adduct", "ionization_mode", "precursor_mz",
        "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]

if __name__ == "__main__":
    df = pq.read_table(DATA / "train.parquet", columns=COLS,
                       filters=[("ingest_lib", "=", "enveda-np-examples")]).to_pandas()
    df["molecule_id"] = df.inchikey14
    g = molecule_graph(build_spectra(df))
    smi = df.drop_duplicates("molecule_id").set_index("molecule_id").normalized_smiles
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = {k: gen.GetFingerprint(Chem.MolFromSmiles(s)) for k, s in smi.items()}
    g["tanimoto"] = [DataStructs.TanimotoSimilarity(fp[a], fp[b]) for a, b in zip(g.a, g.b)]
    m = molecule_masses(df)
    g["dmass"] = m[g.a].to_numpy() - m[g.b].to_numpy()
    out = ROOT / "artifacts" / "np_graph.parquet"
    out.parent.mkdir(exist_ok=True)
    g.to_parquet(out)
    print(f"{len(g)} molecule pairs -> {out}")
