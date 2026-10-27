"""Batch-build and cache PyG graphs for every sweep sample produced by
flow/sweep.sh (Phase 6).

Graph construction must never run inside the training loop (per project
requirements) - this script is the one place it happens, once, up front.
Later training/eval code just loads the cached .pt files.

Uses synthetic.knn_tiled for the cell-near-cell relation: build_graph's
default knn_dense is O(n^2) memory (Phase 10: ~32 B/cell^2, ~7 GiB at 15K
cells) and does not fit real designs on an 8 GB machine.

Sample files are named sweep_<design>__<variant>_{placement,congestion}.npz
(flow/run_flow.sh). Also writes <processed_dir>/manifest.json, the
{"path", "design", "variant"} list src/train.py reads with
data.source = "manifest". A name without "__" is its own design.

Usage:
    python build_dataset.py [raw_dir] [processed_dir]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from build_graph import build_graph
from synthetic import knn_tiled


def main():
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    processed_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    placement_files = sorted(raw_dir.glob("sweep_*_placement.npz"))
    print(f"Found {len(placement_files)} sweep placement files in {raw_dir}")

    built = 0
    skipped = 0
    manifest = []
    for placement_path in placement_files:
        sample = placement_path.stem.removeprefix("sweep_").removesuffix("_placement")
        congestion_path = raw_dir / f"sweep_{sample}_congestion.npz"
        out_path = processed_dir / f"{sample}_graph.pt"

        if not congestion_path.exists():
            print(f"  SKIP {sample}: no matching congestion file")
            skipped += 1
            continue

        placement = np.load(placement_path, allow_pickle=True)
        congestion = np.load(congestion_path, allow_pickle=True)

        try:
            data = build_graph(placement, congestion, knn_fn=knn_tiled)
        except Exception as e:
            print(f"  SKIP {sample}: build_graph failed: {e}")
            skipped += 1
            continue

        torch.save(data, out_path)
        built += 1
        design = sample.split("__")[0]
        variant = sample.split("__")[-1]
        manifest.append(dict(path=str(out_path), design=design, variant=variant))
        print(f"  OK   {sample}: cell={data['cell'].x.shape[0]} "
              f"net={data['net'].x.shape[0]} gcell={data['gcell'].x.shape[0]}")

    manifest_path = processed_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"Wrote {manifest_path}: {len(manifest)} samples, "
          f"{len({m['design'] for m in manifest})} designs")

    print(f"\nBuilt {built} graphs, skipped {skipped}, saved to {processed_dir}")


if __name__ == "__main__":
    main()
