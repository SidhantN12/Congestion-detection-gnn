"""Batch-build and cache PyG graphs for every sweep sample produced by
flow/sweep.sh (Phase 6).

Graph construction must never run inside the training loop (per project
requirements) - this script is the one place it happens, once, up front.
Later training/eval code just loads the cached .pt files.

Usage:
    python build_dataset.py [raw_dir] [processed_dir]
"""

import sys
from pathlib import Path

import numpy as np
import torch

from build_graph import build_graph


def main():
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    processed_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    placement_files = sorted(raw_dir.glob("sweep_*_placement.npz"))
    print(f"Found {len(placement_files)} sweep placement files in {raw_dir}")

    built, skipped = 0, 0
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
            data = build_graph(placement, congestion)
        except Exception as e:
            print(f"  SKIP {sample}: build_graph failed: {e}")
            skipped += 1
            continue

        torch.save(data, out_path)
        built += 1
        print(f"  OK   {sample}: cell={data['cell'].x.shape[0]} "
              f"net={data['net'].x.shape[0]} gcell={data['gcell'].x.shape[0]}")

    print(f"\nBuilt {built} graphs, skipped {skipped}, saved to {processed_dir}")


if __name__ == "__main__":
    main()
