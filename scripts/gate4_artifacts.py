"""Render the saved training curve and the complete measured results table."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/gate4-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = [json.loads(line) for line in (ROOT / "notes/results/gate4_training_metrics.jsonl").read_text().splitlines()]
    epochs = [row for row in records if row["type"] == "epoch"]
    best = records[-1]["best_epoch"]
    figure = plt.figure(figsize=(11, 4.2))
    loss_axis = figure.add_subplot(1, 2, 1)
    rank_axis = figure.add_subplot(1, 2, 2)
    x = [row["epoch"] for row in epochs]
    loss_axis.plot(x, [row["train"]["loss"] for row in epochs], label="Huber + 0.5 × rank")
    loss_axis.set_ylabel("Training loss")
    loss_axis.set_yscale("log")
    for channel in "hv":
        rank_axis.plot(x, [row["val"]["spearman_" + channel] for row in epochs], label="Validation " + channel.upper())
    rank_axis.set_ylabel("Midrank Spearman on ibex")
    for axis in [loss_axis, rank_axis]:
        axis.axvline(best, color="#555555", linestyle="--", label=f"Selected epoch {best}")
        axis.set_xlabel("Epoch (zero-based)")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Final real-corpus run · disjoint design split · validation-selected checkpoint")
    figure.tight_layout()
    figure.savefig(ROOT / "notes/figures/gate4_training_curve.png", dpi=180)
    figure.savefig(ROOT / "notes/figures/gate4_training_curve.pdf")
    plt.close(figure)

    result = json.loads((ROOT / "notes/results/gate4_evaluation.json").read_text())
    fields = ["sample", "method", "channel", "spearman", "kendall", "kendall_sampled", "precision", "recall", "roc_auc", "mae", "rmse"]
    rows = []
    for variant in sorted({row["variant"] for row in result["samples"]}):
        for method in ["Capacity-only", "Density-only", "RUDY", "CongestionGNN"]:
            sample = next(row for row in result["samples"] if row["variant"] == variant and row["method"] == method)
            for channel in "HV":
                metrics = sample["channels"][channel]
                row = dict(sample="aes__" + variant, method=method, channel=channel)
                row.update({key: metrics[key] for key in fields[3:]})
                rows.append(row)
    with (ROOT / "notes/results/gate4_results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Gate 4: every held-out sample and channel", "",
             "All regressions fit training designs only. Kendall is exact here. Hotspots use ratio ≥ 0.9, top-|S| predictions and fractional cutoff-tie credit; precision equals recall by construction.", "",
             "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key]) for key in fields) + " |")
    lines += ["", "RUDY leads the GNN in aggregate ranking and hotspot metrics in both channels. Density-only beats it horizontally and is close vertically. Capacity-only is a weak floor. These results support learning beyond capacity but do not establish a GNN advantage over analytical routing demand. RUDY predicts an uncalibrated demand proxy, so its MAE/RMSE assess absolute calibration as well as spatial prediction; its ranking remains useful despite scale error.", ""]
    (ROOT / "notes/results/gate4_results.md").write_text("\n".join(lines))
    print(f"Saved training curve: {len(epochs)} epochs, selected epoch {best}.")
    print(f"Saved complete results table: {len(rows)} sample/method/channel rows.")


if __name__ == "__main__":
    main()
