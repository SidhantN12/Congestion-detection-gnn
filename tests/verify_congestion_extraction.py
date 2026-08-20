"""Compare extract_congestion.py's output against OpenROAD's own GUI heatmap
dump (gui::dump_heatmap Routing <csv>) for the same .odb, as a correctness
check (Phase 2 gate).

OpenROAD's "Routing" heatmap, in its default "All directions" mode, reports
value(%) = max(horizontal_ratio, vertical_ratio) * 100 per GCell (confirmed
from OpenROAD src/grt/src/heatMap.cpp: RoutingCongestionDataSource::
setCongestionValues, the `else` branch: congestion = max(hor, ver)). This
script reproduces that same combination from our two-channel array and
checks it against the GUI's CSV dump cell-by-cell.

Runs under the project's normal venv (plain numpy/matplotlib) - no OpenROAD
Python API needed here, only reading files.
"""

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_gui_csv(path, pitch_x, pitch_y, nx, ny):
    grid = np.full((ny, nx), np.nan)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x0 = float(row["x0"])
            y0 = float(row["y0"])
            xi = round(x0 / pitch_x)
            yi = round(y0 / pitch_y)
            grid[yi, xi] = float(row["value (%)"])
    return grid


def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <extracted.npz> <gui_dump.csv> <out.png>")
        sys.exit(1)

    npz_path, csv_path, out_png = sys.argv[1], sys.argv[2], sys.argv[3]

    data = np.load(npz_path)
    ratio = data["ratio"]  # (ny, nx, 2): [..., 0]=horizontal, [..., 1]=vertical
    pitch_x_microns = float(data["pitch_x"]) / float(data["dbu_per_micron"])
    pitch_y_microns = float(data["pitch_y"]) / float(data["dbu_per_micron"])
    nx, ny = int(data["nx"]), int(data["ny"])

    mine_pct = np.maximum(ratio[:, :, 0], ratio[:, :, 1]) * 100.0
    gui_pct = load_gui_csv(csv_path, pitch_x_microns, pitch_y_microns, nx, ny)

    missing = np.isnan(gui_pct).sum()
    diff = np.abs(mine_pct - gui_pct)
    print(f"grid shape: {mine_pct.shape}")
    print(f"GUI cells missing from CSV: {missing}/{gui_pct.size}")
    print(f"max abs diff: {np.nanmax(diff):.6f}")
    print(f"mean abs diff: {np.nanmean(diff):.6f}")
    print(f"mine: min={mine_pct.min():.4f} max={mine_pct.max():.4f} mean={mine_pct.mean():.4f}")
    print(f"gui:  min={np.nanmin(gui_pct):.4f} max={np.nanmax(gui_pct):.4f} mean={np.nanmean(gui_pct):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmax = max(mine_pct.max(), np.nanmax(gui_pct))

    im0 = axes[0].imshow(mine_pct, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title("extract_congestion.py\n(max(H,V) ratio %)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(gui_pct, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("OpenROAD GUI\n(gui::dump_heatmap Routing)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff, origin="lower", cmap="viridis")
    axes[2].set_title(f"abs diff\nmax={np.nanmax(diff):.4f}")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("GCell x index")
        ax.set_ylabel("GCell y index")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved comparison plot to {out_png}")


if __name__ == "__main__":
    main()
