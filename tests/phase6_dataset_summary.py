"""Phase 6 gate: dataset summary - sample count, congestion distribution
across samples, and confirmation samples differ meaningfully.

Sweep dimensions (v3): blockage band CENTER (0.25/0.50/0.75 fraction of
die height), blockage band WIDTH (0.10/0.20/0.30 fraction), placement
seed (1-5). Layer cap fixed at "tight" (metal2-metal3), PLACE_DENSITY
fixed at 0.70. See notes/phase6-mini-dataset.md for why: v1 swept
(density, layer-cap tier, seed) and found layer-cap tier produced
identical ground truth regardless of density/seed for constrained tiers;
v2 swept (density, band center, seed) and found density itself produces
bit-identical placement regardless of target for this design. Both
replaced with parameters of the one lever proven (Phase 3) to reliably
move congestion: an explicit routing blockage.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "src")
from metrics import combined_congestion


def main():
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")
    out_png = sys.argv[2] if len(sys.argv) > 2 else "data/raw/phase6_dataset_summary.png"

    graph_files = sorted(processed_dir.glob("bc*_bw*_s*_graph.pt"))
    print(f"=== Sample count ===\n{len(graph_files)} cached graphs in {processed_dir}\n")

    rows = []
    for path in graph_files:
        sample = path.stem.removesuffix("_graph")
        parts = sample.split("_")
        band_center = float(parts[0][2:]) / 100.0
        band_width = float(parts[1][2:]) / 100.0
        seed = int(parts[2][1:])

        data = torch.load(path, weights_only=False)
        ratio = data["gcell"].y_ratio.numpy()
        combined = np.maximum(ratio[:, 0], ratio[:, 1])
        rows.append(dict(
            sample=sample, band_center=band_center, band_width=band_width, seed=seed,
            n_cell=data["cell"].x.shape[0], n_gcell=data["gcell"].x.shape[0],
            mean_cong=combined.mean(), max_cong=combined.max(),
            frac_above_09=(combined > 0.9).mean(),
            cell_x_mean=data["cell"].x[:, 3].mean().item(),  # x/die_w
        ))

    mean_congs = np.array([r["mean_cong"] for r in rows])
    max_congs = np.array([r["max_cong"] for r in rows])
    frac_above = np.array([r["frac_above_09"] for r in rows])

    print("=== Congestion distribution across samples ===")
    print(f"per-sample mean congestion:  min={mean_congs.min():.3f} max={mean_congs.max():.3f} "
          f"mean={mean_congs.mean():.3f} std={mean_congs.std():.3f}")
    print(f"per-sample max congestion:   min={max_congs.min():.3f} max={max_congs.max():.3f} "
          f"mean={max_congs.mean():.3f} std={max_congs.std():.3f}")
    print(f"per-sample frac(ratio>0.9):  min={frac_above.min():.3f} max={frac_above.max():.3f} "
          f"mean={frac_above.mean():.3f} std={frac_above.std():.3f}")
    print()

    print("=== By blockage band center ===")
    for bc in [0.25, 0.50, 0.75]:
        bc_rows = [r for r in rows if r["band_center"] == bc]
        tmc = np.array([r["mean_cong"] for r in bc_rows])
        print(f"center={bc:.2f}: n={len(bc_rows)} mean_cong={tmc.mean():.3f} "
              f"(range {tmc.min():.3f}-{tmc.max():.3f})")
    print()

    print("=== By blockage band width ===")
    for bw in [0.10, 0.20, 0.30]:
        bw_rows = [r for r in rows if r["band_width"] == bw]
        tmc = np.array([r["mean_cong"] for r in bw_rows])
        print(f"width={bw:.2f}: n={len(bw_rows)} mean_cong={tmc.mean():.3f} "
              f"(range {tmc.min():.3f}-{tmc.max():.3f})")
    print()

    print("=== Confirming samples differ meaningfully ===")
    maps = []
    for path in graph_files:
        data = torch.load(path, weights_only=False)
        ratio = data["gcell"].y_ratio.numpy()
        maps.append(np.maximum(ratio[:, 0], ratio[:, 1]))
    maps = np.stack(maps)
    flat = maps.reshape(len(maps), -1)
    corr = np.corrcoef(flat)
    off_diag = corr[~np.eye(len(corr), dtype=bool)]
    print(f"All {len(maps)} samples, pairwise congestion-map correlation: "
          f"mean={off_diag.mean():.3f} min={off_diag.min():.3f} max={off_diag.max():.3f} "
          f"(1.0 would mean identical maps)")

    exact_duplicates = 0
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            if np.array_equal(maps[i], maps[j]):
                exact_duplicates += 1
    print(f"Exact-duplicate congestion maps (bit-identical pairs): {exact_duplicates}")

    same_center_corrs, diff_center_corrs = [], []
    same_width_corrs, diff_width_corrs = [], []
    same_seed_corrs, diff_seed_corrs = [], []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            c = corr[i, j]
            (same_center_corrs if rows[i]["band_center"] == rows[j]["band_center"] else diff_center_corrs).append(c)
            (same_width_corrs if rows[i]["band_width"] == rows[j]["band_width"] else diff_width_corrs).append(c)
            (same_seed_corrs if rows[i]["seed"] == rows[j]["seed"] else diff_seed_corrs).append(c)
    print(f"Same band-center pairs:  mean correlation={np.mean(same_center_corrs):.3f} (n={len(same_center_corrs)})")
    print(f"Diff band-center pairs:  mean correlation={np.mean(diff_center_corrs):.3f} (n={len(diff_center_corrs)})")
    print(f"Same band-width pairs:   mean correlation={np.mean(same_width_corrs):.3f} (n={len(same_width_corrs)})")
    print(f"Diff band-width pairs:   mean correlation={np.mean(diff_width_corrs):.3f} (n={len(diff_width_corrs)})")
    print(f"Same seed pairs:         mean correlation={np.mean(same_seed_corrs):.3f} (n={len(same_seed_corrs)})")
    print(f"Diff seed pairs:         mean correlation={np.mean(diff_seed_corrs):.3f} (n={len(diff_seed_corrs)})")

    cell_x_means = np.array([r["cell_x_mean"] for r in rows])
    print(f"\ncell x-position fingerprint (mean normalized x) across all samples: "
          f"std={cell_x_means.std():.4f} (0 would mean every sample placed identically)")

    assert len(graph_files) >= 30, f"FAIL: only {len(graph_files)} samples, need >=30"
    assert mean_congs.std() > 0, "FAIL: zero variance in congestion across samples"
    assert exact_duplicates == 0, f"FAIL: {exact_duplicates} exact-duplicate congestion maps found"
    print("\nPASS: >=30 samples, nonzero congestion variance, no exact-duplicate maps")

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    colors = {0.25: "#4C72B0", 0.50: "#DD8452", 0.75: "#C44E52"}
    for bc in [0.25, 0.50, 0.75]:
        vals = [r["mean_cong"] for r in rows if r["band_center"] == bc]
        axes[0].scatter([bc] * len(vals), vals, alpha=0.6, color=colors[bc])
    axes[0].set_xlabel("blockage band center (fraction of die height)")
    axes[0].set_ylabel("mean combined congestion")
    axes[0].set_title("Congestion by blockage position")

    for bc in [0.25, 0.50, 0.75]:
        w = [r["band_width"] for r in rows if r["band_center"] == bc]
        m = [r["mean_cong"] for r in rows if r["band_center"] == bc]
        axes[1].scatter(w, m, label=f"center={bc}", alpha=0.7, color=colors[bc])
    axes[1].set_xlabel("blockage band width")
    axes[1].set_ylabel("mean combined congestion")
    axes[1].set_title("Congestion vs. blockage width")
    axes[1].legend()

    im = axes[2].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_title("Pairwise congestion-map correlation\n(45x45, sorted by center/width/seed)")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
