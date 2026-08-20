"""Phase 3 gate report: congestion map with induced overflow, and the
fraction of GCells above 0.9 combined congestion.
"""

import sys

sys.path.insert(0, "src")

import matplotlib.pyplot as plt
import numpy as np

from metrics import combined_congestion, fraction_above


def report(npz_path, label):
    data = np.load(npz_path)
    ratio = data["ratio"]
    combined = combined_congestion(ratio)
    frac = fraction_above(ratio, 0.9)
    print(f"{label}: shape={combined.shape} max={combined.max():.4f} "
          f"mean={combined.mean():.4f} fraction_above_0.9={frac:.4f} "
          f"({int(frac * combined.size)}/{combined.size} GCells)")
    return combined


baseline = report("data/raw/gcd_baseline_grt_congestion.npz", "baseline (base, full stack)")
cap_m3 = report("data/raw/gcd_cap_m3_congestion.npz", "layer cap only (metal2-3)")
blk1 = report("data/raw/gcd_blk1_congestion.npz", "layer cap + die-band blockage")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
vmax = max(baseline.max(), cap_m3.max(), blk1.max())

for ax, combined, title in zip(
    axes, [baseline, cap_m3, blk1],
    ["baseline\n(full stack, base density)",
     "layer cap only\n(metal2-3)",
     "layer cap + die-band blockage\n(overflow forced)"]
):
    im = ax.imshow(combined, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("GCell x index")
    ax.set_ylabel("GCell y index")
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig("data/raw/phase3_congestion_comparison.png", dpi=150)
print("Saved data/raw/phase3_congestion_comparison.png")
