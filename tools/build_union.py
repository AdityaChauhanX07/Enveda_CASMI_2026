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

usage: python tools/build_union.py T1_DIR H_DIR OUT_DIR SLUG [--ours=OWNER/KERNEL] [--commit=fast|smoke]
       [--keep-fr] [--keep-p5]
--ours: the second neural scorer of T1's consensus (default: T1's popwenman v80)
--commit: T1's COMMIT_MODE ('smoke' ranks 12 visible molecules in the commit run and exercises every layer)
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
    i_extlib = find(tc, "# ===== 0.500 APEX OVERLAY")
    cells = []
    for i, c in enumerate(tc):
        if i in drop:
            continue
        if i == i_bundle:
            cells.append(watcher)
        cells.append(c)
        if i == i_extlib:
            cells.extend(layers)

    changes = []
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

    for c in cells:                                   # every cell is a code or markdown cell of one of the two parents
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
