"""Phase 9 gate report: reads the per-scale stats written by
`python src/synthetic.py --cells N --stats-json notes/results/phase9_synth_N.json`
(each run in its own process, so peak memory is isolated per scale),
prints node/edge counts, net degree distribution and peak memory, and
plots the net degree distribution on log-log axes.

Usage:
    python tests/phase9_synthetic_gate.py [out.png]
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCALES = [35_000, 100_000, 500_000, 1_500_000]
LABELS = {35_000: "35K", 100_000: "100K", 500_000: "500K", 1_500_000: "1.5M"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
REL_ORDER = ["cell-pin-net", "net-rev_pin-cell", "cell-in-gcell",
             "gcell-contains-cell", "cell-near-cell", "gcell-adjacent-gcell"]


def load(n):
    with open(f"notes/results/phase9_synth_{n}.json") as f:
        return json.load(f)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "notes/figures/phase9_net_degree_loglog.png"
    runs = [load(n) for n in SCALES]
    mb = lambda b: b / 2**20
    cols = [LABELS[n] for n in SCALES]
    row = lambda name, vals: print(f"{name:<24}" + "".join(f"{v:>14}" for v in vals))

    print("=== Nodes ===")
    row("", cols)
    for nt in ["cell", "net", "gcell"]:
        row(nt, [f"{r['nodes'][nt]:,}" for r in runs])
    row("total", [f"{sum(r['nodes'].values()):,}" for r in runs])
    row("gcell grid (ny x nx)", [f"{r['grid'][0]}x{r['grid'][1]}" for r in runs])

    print("\n=== Edges ===")
    row("", cols)
    for rel in REL_ORDER:
        row(rel, [f"{r['edges'][rel]:,}" for r in runs])
    row("total", [f"{sum(r['edges'].values()):,}" for r in runs])

    print("\n=== Net degree (pins per net) ===")
    row("", cols)
    row("max", [f"{r['net_degree']['max']:,}" for r in runs])
    row("mean", [f"{r['net_degree']['mean']:.2f}" for r in runs])
    row("median", [f"{r['net_degree']['median']:.0f}" for r in runs])
    row("% nets with 1 pin", [f"{100 * r['net_degree']['frac_eq_1']:.1f}%" for r in runs])
    row("% nets with 2 pins", [f"{100 * r['net_degree']['frac_eq_2']:.1f}%" for r in runs])
    row("% nets with 3-4 pins", [f"{100 * r['net_degree']['frac_3_4']:.1f}%" for r in runs])
    row("# nets >= 1000 pins", [f"{r['net_degree']['n_ge_1000']}" for r in runs])
    row("pins on >=1000-pin nets", [
        f"{100 * sum(d * c for d, c in r['net_degree']['hist'] if d >= 1000) / r['edges']['cell-pin-net']:.1f}%"
        for r in runs])

    print("\n=== Spatial clustering (cells per gcell) ===")
    row("", cols)
    row("mean", [f"{r['cells_per_gcell']['mean']:.2f}" for r in runs])
    row("var/mean (Poisson=1)", [f"{r['cells_per_gcell']['dispersion_index']:.2f}" for r in runs])
    row("% empty gcells", [f"{100 * r['cells_per_gcell']['frac_empty']:.1f}%" for r in runs])
    row("max gcell cell_density", [f"{r['gcell_cell_density_max']:.2f}" for r in runs])
    row("cells with 0 pins", [f"{r['cells_with_no_pins']:,}" for r in runs])

    print(f"\n=== Construction time and memory ({runs[0]['memory']['method']}) ===")
    row("", cols)
    row("placement gen (s)", [f"{r['time_s']['placement']:.2f}" for r in runs])
    row("build_graph (s)", [f"{r['time_s']['build_graph']:.2f}" for r in runs])
    row("baseline (MiB)", [f"{mb(r['memory']['baseline_bytes']):.0f}" for r in runs])
    row("peak (MiB)", [f"{mb(r['memory']['peak_bytes']):.0f}" for r in runs])
    row("peak - baseline (MiB)", [f"{mb(r['memory']['peak_minus_baseline_bytes']):.0f}" for r in runs])
    row("graph tensors (MiB)", [f"{mb(r['memory']['graph_tensor_bytes']):.0f}" for r in runs])

    # ---- log-log degree distribution: PMF (left) and CCDF (right) ----
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax_pmf, ax_ccdf) = plt.subplots(1, 2, figsize=(13, 5.2))
    for r, color, n in zip(runs, COLORS, SCALES):
        hist = np.array(r["net_degree"]["hist"])
        deg, cnt = hist[:, 0], hist[:, 1]
        n_nets = cnt.sum()
        label = f"{LABELS[n]} cells (max {r['net_degree']['max']:,})"
        ax_pmf.scatter(deg, cnt / n_nets, s=14, color=color, label=label, alpha=0.8,
                       edgecolors="white", linewidths=0.4)
        ccdf = 1.0 - np.concatenate([[0], np.cumsum(cnt)[:-1]]) / n_nets  # P(degree >= d)
        ax_ccdf.step(deg, ccdf, where="post", color=color, lw=2, label=label)
        ax_ccdf.annotate(f"{r['net_degree']['max']:,}", (deg[-1], ccdf[-1]), xytext=(4, 0),
                         textcoords="offset points", color="#52514e", fontsize=10, va="center")

    for ax in (ax_pmf, ax_ccdf):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("pins per net (net degree)")
        ax.grid(True, which="major", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
    ax_pmf.set_ylabel("fraction of nets")
    ax_pmf.set_title("Degree distribution (PMF)", loc="left", fontsize=13)
    ax_ccdf.set_ylabel("fraction of nets with degree ≥ d")
    ax_ccdf.set_title("Tail: P(degree ≥ d), max degree labelled", loc="left", fontsize=13)
    ax_pmf.legend(frameon=False, fontsize=10)
    fig.suptitle("Synthetic net degree distribution: body at 2–4 pins, hub nets with thousands",
                 x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
