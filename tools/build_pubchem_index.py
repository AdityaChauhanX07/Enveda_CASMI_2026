"""Build a mass-sorted PubChem candidate index with documentation counts from PubChem's primary FTP files.

Runs as a Kaggle CPU script with internet on (kernels/pubchem-index-build); its output replaces the shared
`casmi26-pubchem-tier` (no public build code) in the compliant pipeline. Same layout as that tier, so
casmi.edit_overlay.Tier and the panel tools read it unchanged:
  pc_mass.npy   float64, sorted monoisotopic mass (150 - 1250 Da)
  pc_off.npy    int64, N + 1 byte offsets into pc_smiles.npy
  pc_smiles.npy uint8, concatenated SMILES (PubChem's, one representative CID per InChIKey14)
  pc_ik14.npy   S14, InChIKey connectivity block
  pc_lsid.npy   float16, log1p(substance records), summed over the CIDs sharing the InChIKey14
  pc_lpmid.npy  float16, log1p(PubMed records), same aggregation
  pc_cid.npy    int64, representative CID (smallest)
Filters: neutral formula of C, H, N, O, P, S, F, Cl, Br, I with at least one C; one component (no '.' in SMILES).
Sources: https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/ (CID-Mass, CID-InChI-Key, CID-SMILES, CID-SID,
CID-PMID). PubChem data courtesy of the U.S. National Library of Medicine.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import os
BASE = os.environ.get("PUBCHEM_BASE", "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/")   # file:// for tests
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/kaggle/working/pubchem_index")
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0          # > 0: read only this many rows per file (dry run)
LO, HI = 150.0, 1250.0
ALLOWED = re.compile(r"^(?:(?:C|H|N|O|P|S|F|Cl|Br|I)\d*)+$")
CHUNK = 2_000_000
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f} s]", *a, flush=True)


def stream(name: str, names: list, usecols: list, dtype: dict):
    """Chunks of a gzip TSV read straight from PubChem's FTP site (nothing stored on disk)."""
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "casmi-index-build"})
    resp = urllib.request.urlopen(req, timeout=600)
    gz = gzip.GzipFile(fileobj=resp)
    rd = pd.read_csv(gz, sep="\t", header=None, names=names, usecols=usecols, dtype=dtype, chunksize=CHUNK,
                     quoting=3, na_filter=False, on_bad_lines="skip", engine="c")
    n = 0
    for ch in rd:
        if n == 0:
            log(f"{name}: first rows\n{ch.head(3).to_string()}")
        n += len(ch)
        yield ch
        if LIMIT and n >= LIMIT:
            break
    log(f"{name}: {n:,} rows read")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    # 1) masses and formulas -> kept CIDs
    cids, masses = [], []
    for ch in stream("CID-Mass.gz", ["cid", "formula", "mono", "exact"], ["cid", "formula", "mono"],
                     {"cid": np.int64, "formula": str, "mono": np.float64}):
        m = ch.mono.to_numpy()
        ok = (m >= LO) & (m <= HI)
        f = ch.formula.to_numpy()[ok]
        okf = np.fromiter((bool(ALLOWED.match(x)) and "C" in x.replace("Cl", "") for x in f), bool, len(f))
        cids.append(ch.cid.to_numpy()[ok][okf]); masses.append(m[ok][okf])
    cid = np.concatenate(cids); mass = np.concatenate(masses); del cids, masses
    o = np.argsort(cid, kind="stable"); cid, mass = cid[o], mass[o]
    report["after_mass_formula"] = int(len(cid)); log(f"kept by mass / formula: {len(cid):,}")
    maxcid = int(cid.max()) + 1
    pos = np.full(maxcid, -1, np.int64); pos[cid] = np.arange(len(cid))
    # 2) InChIKey14 for kept CIDs
    ik = np.zeros(len(cid), "S14"); have_ik = np.zeros(len(cid), bool)
    for ch in stream("CID-InChI-Key.gz", ["cid", "inchi", "key"], ["cid", "key"], {"cid": np.int64, "key": str}):
        c = ch.cid.to_numpy(); sel = c < maxcid
        p = pos[c[sel]]; m = p >= 0
        keys = ch.key.to_numpy()[sel][m]
        ik[p[m]] = np.array([k[:14] for k in keys], "S14"); have_ik[p[m]] = True
    # 3) SMILES (single component only), kept as one byte blob + per-row offset / length (no per-row objects)
    soff = np.full(len(cid), -1, np.int64); slen = np.zeros(len(cid), np.int32); have_smi = np.zeros(len(cid), bool)
    blob = bytearray()
    for ch in stream("CID-SMILES.gz", ["cid", "smiles"], ["cid", "smiles"], {"cid": np.int64, "smiles": str}):
        c = ch.cid.to_numpy(); sel = c < maxcid
        p = pos[c[sel]]; m = p >= 0
        s = ch.smiles.to_numpy()[sel][m]; p = p[m]
        single = np.fromiter(("." not in x for x in s), bool, len(s))
        enc = [x.encode() for x in s[single]]
        lens = np.fromiter((len(b) for b in enc), np.int64, len(enc))
        start = len(blob)
        soff[p[single]] = start + np.r_[0, np.cumsum(lens)[:-1]] if len(enc) else soff[p[single]]
        slen[p[single]] = lens; have_smi[p[single]] = True
        blob += b"".join(enc)
    log(f"SMILES blob {len(blob) / 1e9:.2f} GB")
    keep = have_ik & have_smi
    report["with_key_and_single_smiles"] = int(keep.sum()); log(f"with InChIKey and one-component SMILES: {keep.sum():,}")
    # 4) documentation counts per CID
    nsid = np.zeros(len(cid), np.int64); npm = np.zeros(len(cid), np.int64)
    for name, arr in (("CID-SID.gz", nsid), ("CID-PMID.gz", npm)):
        for ch in stream(name, ["cid", "b", "c"], ["cid"], {"cid": np.int64}):
            c = ch.cid.to_numpy(); c = c[c < maxcid]; p = pos[c]; p = p[p >= 0]
            arr += np.bincount(p, minlength=len(cid))
    del pos
    # 5) one row per InChIKey14: smallest CID represents, counts summed
    idx = np.flatnonzero(keep); del keep, have_ik, have_smi
    idx = idx[np.lexsort((cid[idx], ik[idx]))]
    ikx = ik[idx]
    first = np.empty(len(idx), bool); first[0] = True; first[1:] = ikx[1:] != ikx[:-1]
    del ikx
    grp = np.cumsum(first) - 1
    rep = idx[first]
    sid_sum = np.bincount(grp, weights=nsid[idx]); pm_sum = np.bincount(grp, weights=npm[idx])
    del idx, grp, first, nsid, npm
    report["unique_ik14"] = int(len(rep)); log(f"unique InChIKey14: {len(rep):,}")
    # 6) sort by mass and write
    o = np.argsort(mass[rep], kind="stable")
    rep, sid_sum, pm_sum = rep[o], sid_sum[o], pm_sum[o]
    np.save(OUT / "pc_mass.npy", mass[rep].astype(np.float64))
    np.save(OUT / "pc_ik14.npy", ik[rep]); del ik
    np.save(OUT / "pc_cid.npy", cid[rep].astype(np.int64))
    mass_range = [float(mass[rep[0]]), float(mass[rep[-1]])]; del mass, cid
    np.save(OUT / "pc_lsid.npy", np.log1p(sid_sum).astype(np.float16))
    np.save(OUT / "pc_lpmid.npy", np.log1p(pm_sum).astype(np.float16))
    lens = slen[rep].astype(np.int64)
    off = np.zeros(len(rep) + 1, np.int64); off[1:] = np.cumsum(lens)
    np.save(OUT / "pc_off.npy", off)
    mv = memoryview(blob); buf = np.empty(int(off[-1]), np.uint8)
    for a in range(0, len(rep), 1_000_000):                  # gather the representatives' SMILES in mass order
        b = min(a + 1_000_000, len(rep))
        part = b"".join(mv[soff[r]:soff[r] + slen[r]] for r in rep[a:b])
        buf[off[a]:off[b]] = np.frombuffer(part, np.uint8)
    np.save(OUT / "pc_smiles.npy", buf)
    report.update(rows=int(len(rep)), mass_range=mass_range, smiles_bytes=int(len(buf)),
                  seconds=round(time.time() - T0), limit=LIMIT, filters=dict(mass=[LO, HI], elements="CHNOPSFClBrI", single_component=True),
                  source=BASE, files=["CID-Mass.gz", "CID-InChI-Key.gz", "CID-SMILES.gz", "CID-SID.gz", "CID-PMID.gz"])
    (OUT / "BUILD_REPORT.json").write_text(json.dumps(report, indent=1))
    log("done", json.dumps(report))
