"""Phase 5 check 1, venv side: OpenROAD's getXIdx/getYIdx (dumped by
tests/dump_gcell_index.tcl) against build_graph.gcell_index, the function
build_graph.py actually uses to assign cells to gcells.

Usage:
    python tests/verify_gcell_index.py <index.tsv>
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_graph import gcell_index  # noqa: E402


def main():
    rows = []
    x_grid = None
    y_grid = None
    with open(sys.argv[1]) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "grid_x":
                x_grid = [int(v) for v in parts[1:]]
            elif parts[0] == "grid_y":
                y_grid = [int(v) for v in parts[1:]]
            else:
                rows.append(parts)

    nx = len(x_grid)
    ny = len(y_grid)
    origin_x = x_grid[0]
    origin_y = y_grid[0]
    pitch_x = x_grid[1] - x_grid[0]
    pitch_y = y_grid[1] - y_grid[0]

    kinds = np.array([r[0] for r in rows])
    x = np.array([int(r[1]) for r in rows], dtype=np.float64)
    y = np.array([int(r[2]) for r in rows], dtype=np.float64)
    or_x = np.array([int(r[3]) for r in rows])
    or_y = np.array([int(r[4]) for r in rows])
    idx = gcell_index(x, y, origin_x, origin_y, pitch_x, pitch_y, nx, ny)
    f_x = idx[0]
    f_y = idx[1]

    bad = (f_x != or_x) | (f_y != or_y)
    print(f"grid {nx}x{ny}, origin ({origin_x},{origin_y}), pitch ({pitch_x},{pitch_y})")
    print(f"checked by kind: {dict(Counter(kinds.tolist()))}")
    print(f"mismatches by kind: {dict(Counter(kinds[bad].tolist()))}")
    print(f"total mismatches: {int(bad.sum())} / {len(rows)}")
    sys.exit(1 if bad.any() else 0)


if __name__ == "__main__":
    main()
