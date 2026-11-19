"""Union of the two public frontier notebooks, as one private kernel (a candidate second final).

T1 = taeyangg4's casmi26-lb-0-495-network-fr-p5-tail (public 0.495): guhongbin's 0.487 base (the 0.465 pipeline, the
     neural consensus with a second scorer, the fragment check, the v8 mass rule, the documentation prior on ranks 2-25,
     the external-library rank-1 overlay) + N-1c D-L1 network + FR 3-25 + P5 tail.
H  = heliosli's casmi26-0-496-public-demo (public 0.496): the 0.477 base + FRIGID / E1 de novo candidates on the
     second T4, the rr3 same-formula isomer reranker (FREEZE), the p1 gate, the cvjoint joint ranker (RT, FPNet,
     natural-product membership) re-judging rank 1, the M5 in-list isomer judgement (ICE-T), the FRIGID merge.
Both run the same pinned 0.465 pipeline (identical SHA-256) through the same stage machinery and helper modules, so
H's layers find the files they read (frozen_candidate_evidence.parquet, pc_lists.json, eng_lists.json,
execution_receipt.json, submission.csv) in T1's run as well; their globals (ROOT, COMP, NB_T0, IS_RERUN, SAMPLE_IDS)
are T1's too.

Union order: T1 up to its external-library overlay, H's FRIGID watcher just before T1's pipeline cell (it waits for
the pipeline's export in the background on the second T4), H's layers after T1's external-library overlay, T1's
run summary. T1's FR 3-25 (+0.001 on the board) and P5 tail (+0.002, up to 90 min) are left out by default: H's
layers stop at fixed session times (8.2-8.8 h), so the union ends when H ends, and P5 would only take their time.
Every layer of both notebooks fails closed (any error or time-out keeps its input file).

Library guard (default on): T1's external-library overlay (APEX: a test spectrum matching an external reference
spectrum, cosine >= 0.80 with >= 4 peaks within 10 ppm, puts that reference's structure at rank 1; TAU 0.85 -> 0.80
alone was +0.006 on the board) is T1's last rank-1 decision. H's layers never saw it (H's notebook has no such
overlay) and only protect lists with a strong match in the pipeline's own library: rr3 (FREEZE) reorders every other
list's non-training candidates and M5's swap can move rank 1 without reading the library at all. So the overlay also
records its matches (apex_extlib_hits.json; three added lines, nothing else changes) and a last cell puts each matched
structure back at rank 1, the rest of its list in H's order. Lists without a match keep H's layers as they are.
H's CV-JOINT cell is in H's published notebook switched off (CVJ_SWAP = CVJ_INS = 1e9: it moves nothing) and is kept
as is, like every other H cell.

usage: python tools/build_union.py T1_DIR H_DIR OUT_DIR SLUG [--ours=OWNER/KERNEL] [--commit=fast|smoke]
       [--keep-fr] [--keep-p5] [--no-guard]
--ours: the second neural scorer of T1's consensus (default: T1's popwenman v80)
--commit: T1's COMMIT_MODE ('smoke' ranks 12 visible molecules in the commit run and exercises every layer)
--no-guard: leave out the library guard (T1's overlay cell unchanged, no last cell)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HEADER = """# CASMI26 · union of the two public frontier notebooks

This private kernel runs two public notebooks' layers on one pipeline. The base and its first layers are
**taeyangg4's [casmi26-lb-0-495-network-fr-p5-tail](https://www.kaggle.com/code/taeyangg4/casmi26-lb-0-495-network-fr-p5-tail)**
(public 0.495: guhongbin's 0.487 notebook + the N-1c D-L1 neighbour network + the external-library rank-1 overlay);
after them come **heliosli's [casmi26-0-496-public-demo](https://www.kaggle.com/code/heliosli/casmi26-0-496-public-demo)**
layers (public 0.496: FRIGID / E1 de novo candidates on the second T4, the rr3 isomer reranker with FREEZE, the p1
gate, the cvjoint joint ranker, the M5 in-list isomer judgement and the FRIGID merge), unchanged. Both notebooks run the
same pinned 0.465 pipeline and stage machinery, so every layer reads the files it was written for.
{dropped}
Every cell below is the authors' code, unchanged{changes}. The authors and everyone they credit are credited in
the cells that follow.
"""


def code_text(c: dict) -> str:
    return "".join(c["source"]) if c["cell_type"] == "code" else ""


def find(cells: list, prefix: str, kind: str = "code") -> int:
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == kind and "".join(c["source"]).lstrip().startswith(prefix)]
    assert len(hits) == 1, f"{prefix!r}: {len(hits)} cells"
    return hits[0]


T_APEX = "# ===== 0.500 APEX OVERLAY"
APEX_PATCH = (                                       # (anchor, replacement): each anchor occurs once in T1's overlay cell
    ("    hits_count = 0\n", "    hits_count = 0\n    _apex_hits = {}   # union: the matches, for the library guard\n"),
    ("                    new_smis = ([match_smi] + smis)[:25]\n                    promoted_count += 1\n",
     "                    new_smis = ([match_smi] + smis)[:25]\n                    promoted_count += 1\n"
     "                _apex_hits[mid] = match_smi   # union\n"),
    ("        json.dump(audit, f, indent=2)\n",
     "        json.dump(audit, f, indent=2)\n"
     "    with open(os.path.join(work, 'apex_extlib_hits.json'), 'w') as f:   # union: the matches, for the library guard\n"
     "        json.dump(_apex_hits, f)\n"),
)

GUARD = r'''# ===== union: library guard - the external-library matches stay at rank 1 =====
# The external-library overlay above heliosli's layers (taeyangg4's notebook) puts the structure of a matching external
# reference spectrum (cosine >= 0.80, >= 4 peaks, 10 ppm, lists without a strong match in the pipeline's own library)
# at rank 1; it was that notebook's last rank-1 decision. heliosli's layers never saw those matches: their FREEZE
# reranker and M5's swap may have moved such a rank 1. This cell puts each match back at rank 1, the rest of the list in
# its current order (a match no longer in the list goes back in at rank 1, the last entry drops out). Lists without a
# match are not touched. Any failure leaves the submission unchanged.
def _union_library_guard(work='/kaggle/working'):
    import json
    import os

    import pandas as pd

    hp = os.path.join(work, 'apex_extlib_hits.json')
    if not os.path.exists(hp):
        print('[union-guard] no apex_extlib_hits.json (the overlay found no library or failed): submission unchanged')
        return
    hits = {str(k): str(v) for k, v in json.load(open(hp)).items()}
    sub_path = os.path.join(work, 'submission.csv')
    sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    out, audit = [], dict(matches=len(hits), in_submission=0, still_rank1=0, put_back=0, reinserted=0, moved_from={})
    for mid, smiles in zip(sub['molecule_id'].astype(str), sub['smiles'].astype(str)):
        lst = [s for s in smiles.split(';') if s]
        h = hits.get(mid)
        if h:
            audit['in_submission'] += 1
            if lst and lst[0] == h:
                audit['still_rank1'] += 1
            else:
                if h in lst:
                    audit['put_back'] += 1
                    k = str(lst.index(h) + 1)
                    audit['moved_from'][k] = audit['moved_from'].get(k, 0) + 1
                else:
                    audit['reinserted'] += 1
                lst = ([h] + [s for s in lst if s != h])[:25]
        out.append(';'.join(lst) if lst else smiles)
    res = sub.copy()
    res['smiles'] = out
    assert len(res) == len(sub) and list(res['molecule_id']) == list(sub['molecule_id'])
    parts = res['smiles'].str.split(';')
    assert parts.map(len).max() <= 25 and all(len(set(p)) == len(p) for p in parts)
    if audit['put_back'] or audit['reinserted']:
        res.to_csv(os.path.join(work, 'submission.guard.csv'), index=False)
        os.replace(os.path.join(work, 'submission.guard.csv'), sub_path)
    with open(os.path.join(work, 'union_guard_audit.json'), 'w') as f:
        json.dump(audit, f, indent=1)
    print('[union-guard] done:', json.dumps(audit))


try:
    _union_library_guard()
except Exception:
    import traceback

    traceback.print_exc()
    print('[union-guard] failed; submission unchanged')
'''

H_WATCHER = "# ===== heliosli: FRIGID + E1 on the second T4"
H_LAYERS = ("# ===== heliosli: time guard for the FREEZE cell", "# ===== heliosli overlay v3 on the 0.468 notebook",
            "# ===== heliosli: p1 for the CV-JOINT gate", "# ===== heliosli overlay cvjoint after FREEZE",
            "# ===== heliosli M5 part A", "# ===== heliosli: merge the FRIGID + E1 candidates", "# ===== heliosli M5 part B")


def main() -> None:
    t_dir, h_dir, out, slug = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    args = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[5:] if a.startswith("--"))
    t_nb = json.loads(next(t_dir.glob("*.ipynb")).read_text())
    h_nb = json.loads(next(h_dir.glob("*.ipynb")).read_text())
    t_meta = json.loads((t_dir / "kernel-metadata.json").read_text())
    h_meta = json.loads((h_dir / "kernel-metadata.json").read_text())
    tc, hc = t_nb["cells"], h_nb["cells"]

    watcher = hc[find(hc, H_WATCHER)]
    layers = [hc[find(hc, p)] for p in H_LAYERS]
    drop = set()
    if "keep-fr" not in args:
        drop |= {find(tc, "# ── H-A-111 s2c follow-up: FR 3-25"), find(tc, "## Ours · 2: FR 3-25", "markdown")}
    if "keep-p5" not in args:
        drop |= {find(tc, "# ── H-A-130 s5: P5 tail"), find(tc, "## Ours · 3: P5 tail", "markdown")}
    i_bundle = find(tc, "BUNDLE = materialise_sources()")
    i_extlib = find(tc, T_APEX)
    guard = "no-guard" not in args
    cells = []
    for i, c in enumerate(tc):
        if i in drop:
            continue
        if i == i_bundle:
            cells.append(watcher)
        if i == i_extlib and guard:
            s = code_text(c)
            for old, new in APEX_PATCH:
                assert s.count(old) == 1, old
                s = s.replace(old, new)
            c = dict(c, source=s.splitlines(keepends=True))
        cells.append(c)
        if i == i_extlib:
            cells.extend(layers)
            if guard:
                cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                              "source": GUARD.splitlines(keepends=True)})

    changes = []
    if guard:
        changes.append("taeyangg4's external-library overlay also records its matches, and one cell of ours after "
                       "heliosli's layers puts each match back at rank 1 (heliosli's layers never saw those matches)")
    if "commit" in args:
        k = next(i for i, c in enumerate(cells) if "COMMIT_MODE = 'fast'" in code_text(c))
        s = code_text(cells[k])
        assert s.count("COMMIT_MODE = 'fast'") == 1
        cells[k] = dict(cells[k], source=s.replace("COMMIT_MODE = 'fast'", f"COMMIT_MODE = '{args['commit']}'").splitlines(keepends=True))
        changes.append(f"COMMIT_MODE '{args['commit']}'")
    if "ours" in args:
        changes.append(f"the consensus's second scorer from `{args['ours']}`")
    dropped = []
    if "keep-fr" not in args:
        dropped.append("FR 3-25")
    if "keep-p5" not in args:
        dropped.append("the P5 tail")
    header = HEADER.format(
        dropped=(f"\ntaeyangg4's {' and '.join(dropped)} are left out: heliosli's layers stop at fixed session times, "
                 "and the union has to end inside Kaggle's 9 hours.\n") if dropped else "",
        changes=(" except: " + "; ".join(changes)) if changes else "")
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": header.splitlines(keepends=True)})

    for c in cells:                                   # the parents' cells, and the guard cell
        if c["cell_type"] == "code":
            c["execution_count"], c["outputs"] = None, []
    nb = dict(t_nb, cells=cells)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{slug}.ipynb"
    (out / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    ds = list(dict.fromkeys(t_meta.get("dataset_sources", []) + h_meta.get("dataset_sources", [])))
    ks = list(t_meta.get("kernel_sources", []))
    if "ours" in args:
        assert "popwenman/casmi26-scorer-train-v80" in ks, ks
        ks = [args["ours"] if k == "popwenman/casmi26-scorer-train-v80" else k for k in ks]
    meta = {"id": f"adityachauhanx/{slug}", "title": slug.replace("-", " "), "code_file": name, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "keywords": ["gpu"], "dataset_sources": ds, "kernel_sources": ks,
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "model_sources": [],
            "docker_image": h_meta["docker_image"], "docker_image_pinning_type": "original", "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("written", out / name, "| cells", len(cells), "| dropped", dropped or "none", "| changes", changes or "none")
    print("datasets", len(ds), "| kernels", ks, "| image", meta["docker_image"][-16:])


if __name__ == "__main__":
    main()
