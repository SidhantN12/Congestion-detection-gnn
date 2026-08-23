"""Phase 5 gate: verify the graph's gcell grid lines up exactly with the
congestion ground truth's grid - both the index arithmetic (numeric check)
and a visual overlay (cell positions + grid lines + congestion heatmap all
plotted in the same real coordinate space).

Runs in the project venv - only needs the two .npz files, no OpenROAD.
The index arithmetic itself was separately confirmed against OpenROAD's
own dbGCellGrid.getXIdx()/getYIdx() (0 mismatches across 614 real
instances + 6 boundary cases) - see notes/phase5-alignment-check.md.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    placement_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/gcd_blk1_placement.npz"
    congestion_path = sys.argv[2] if len(sys.argv) > 2 else "data/raw/gcd_blk1_congestion.npz"
    out_png = sys.argv[3] if len(sys.argv) > 3 else "data/raw/phase5_gcell_alignment.png"

    placement = np.load(placement_path)
    congestion = np.load(congestion_path)

    origin_x, origin_y = int(congestion["origin_x"]), int(congestion["origin_y"])
    pitch_x, pitch_y = int(congestion["pitch_x"]), int(congestion["pitch_y"])
    nx, ny = int(congestion["nx"]), int(congestion["ny"])
    die_w, die_h = float(placement["die_w"]), float(placement["die_h"])

    print("=== Metadata cross-check ===")
    print(f"congestion grid: nx={nx} ny={ny} origin=({origin_x},{origin_y}) pitch=({pitch_x},{pitch_y})")
    print(f"placement die: {die_w} x {die_h}")
    grid_span_x = pitch_x * (nx - 1)
    grid_span_y = pitch_y * (ny - 1)
    print(f"grid span (last line - origin): ({grid_span_x}, {grid_span_y}) vs die ({die_w}, {die_h}) "
          f"- last GCell is oversized to reach the die edge, by design (see Phase 2 notes)")

    cell_x = placement["cell_x"].astype(np.float64)
    cell_y = placement["cell_y"].astype(np.float64)
    n_cells = len(cell_x)

    cell_xi = np.clip(((cell_x - origin_x) / pitch_x).astype(np.int64), 0, nx - 1)
    cell_yi = np.clip(((cell_y - origin_y) / pitch_y).astype(np.int64), 0, ny - 1)

    print("\n=== Bounds check (every cell must land inside the grid) ===")
    out_of_bounds = ((cell_x < origin_x) | (cell_y < origin_y)).sum()
    print(f"cells with coordinates below grid origin: {out_of_bounds}")
    assert cell_xi.min() >= 0 and cell_xi.max() <= nx - 1
    assert cell_yi.min() >= 0 and cell_yi.max() <= ny - 1
    print(f"cell_xi range: [{cell_xi.min()}, {cell_xi.max()}] (grid: [0, {nx - 1}])")
    print(f"cell_yi range: [{cell_yi.min()}, {cell_yi.max()}] (grid: [0, {ny - 1}])")

    # Per-cell containment re-check: does (cell_x, cell_y) actually fall
    # within the [x_grid[xi], x_grid[xi+1]) rectangle its index claims?
    x_grid = origin_x + pitch_x * np.arange(nx)
    y_grid = origin_y + pitch_y * np.arange(ny)
    x_lo = x_grid[cell_xi]
    x_hi = np.where(cell_xi < nx - 1, x_grid[np.minimum(cell_xi + 1, nx - 1)], np.inf)
    y_lo = y_grid[cell_yi]
    y_hi = np.where(cell_yi < ny - 1, y_grid[np.minimum(cell_yi + 1, ny - 1)], np.inf)
    contained = (cell_x >= x_lo) & (cell_x < x_hi) & (cell_y >= y_lo) & (cell_y < y_hi)
    print(f"cells correctly contained in their assigned gcell rectangle: {contained.sum()}/{n_cells}")
    assert contained.all(), "FAIL: some cells fall outside their assigned gcell's rectangle"

    print("\nPASS: every cell's assigned gcell index matches its real coordinates")

    # ---- Visual overlay ----
    ratio = congestion["ratio"]
    combined = np.maximum(ratio[:, :, 0], ratio[:, :, 1])

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [origin_x, origin_x + nx * pitch_x, origin_y, origin_y + ny * pitch_y]
    im = ax.imshow(combined, origin="lower", cmap="inferno", extent=extent, alpha=0.85, vmin=0)

    for gx in x_grid:
        ax.axvline(gx, color="cyan", linewidth=0.4, alpha=0.6)
    for gy in y_grid:
        ax.axhline(gy, color="cyan", linewidth=0.4, alpha=0.6)

    ax.scatter(cell_x, cell_y, s=4, c="white", edgecolors="black", linewidths=0.2, zorder=5)

    ax.set_title("Phase 5: cell positions + gcell grid lines over congestion heatmap\n"
                  "(cells must fall cleanly into grid cells, no offset)")
    ax.set_xlabel("x (DBU)")
    ax.set_ylabel("y (DBU)")
    plt.colorbar(im, ax=ax, label="combined congestion ratio", fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved overlay to {out_png}")


if __name__ == "__main__":
    main()
