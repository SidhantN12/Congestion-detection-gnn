"""Phase 10 report: table, fits/extrapolation, and the memory-vs-size chart,
all computed from notes/results/phase10_scalability.json (written by
`python src/scalability.py run ...`). Nothing here is measured; everything
here is derived from what was.

Extrapolation method:
  - Forward/training peaks are fitted as peak = a + b*cells by least
    squares, on the points with cells >= FIT_MIN_CELLS only. Below that,
    repeat runs disagree substantially (up to 34% within one batch of 3,
    and up to ~60% between separate batches - allocator/threading noise
    on small absolute sizes, see notes) and the fixed ~0.25 GiB torch
    baseline dominates. The fit is checked by refitting without the largest
    measured point and predicting it (held-out error reported).
  - O(n^2) components (original dense kNN, train.py's pairwise ranking
    loss) are fitted as peak = a + c*n^2 on all measured points.
  - "12 GB" is taken as 12 GiB (12 * 2^30 bytes), the size of a "12 GB" GPU.

Usage:
    python src/scalability.py report
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "notes/results/phase10_scalability.json"
TABLE_OUT = "notes/results/phase10_report.txt"
FIG_OUT = "notes/figures/phase10_memory_vs_design_size.png"
GIB = 2 ** 30
LIMIT = 12 * GIB
FIT_MIN_CELLS = 200_000
SCALES = [10_000, 20_000, 35_000, 50_000, 100_000, 200_000, 350_000, 500_000,
          750_000, 1_000_000, 1_500_000]
FWD_MODES = ["fwd_nograd", "fwd_grad", "fwd_bwd"]

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_INFER, C_TRAIN, C_BUILD, C_DENSE = "#2a78d6", "#eb6834", "#1baf7a", "#b07800"


def load():
    with open(RESULTS) as f:
        rows = json.load(f)
    by = {}
    for r in rows:
        by.setdefault(r["mode"], {})[r["cells"]] = r
    return by


def ok_points(by, mode, min_cells=0):
    pts = sorted((n, r["peak_bytes"]) for n, r in by.get(mode, {}).items()
                 if r["status"] == "ok" and n >= min_cells)
    return np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def killed_at(by, mode):
    ks = [n for n, r in by.get(mode, {}).items() if r["status"] == "exceeded_budget"]
    return min(ks) if ks else None


def linfit(x, y):
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, r2, np.abs(pred / y - 1).max()


def quadfit(n, y):
    # peak = a + c*n^2, least squares in (1, n^2)
    A = np.stack([np.ones_like(n), n ** 2], 1)
    (a, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a + c * n ** 2
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, c, r2, np.abs(pred / y - 1).max()


def main():
    by = load()
    lines = []
    out = lambda s="": lines.append(s)
    gib = lambda b: f"{b / GIB:.2f}"

    def cell(mode, n, key="peak_bytes", fmt=gib):
        r = by.get(mode, {}).get(n)
        if r is None:
            return "-"
        if r["status"] == "exceeded_budget":
            return f">{r['budget_bytes'] / GIB:.1f} (killed)"
        return fmt(r[key])

    budget = next(r["budget_bytes"] for m in by.values() for r in m.values() if "budget_bytes" in r)
    any_ok = next(r for r in by["fwd_nograd"].values() if r["status"] == "ok")

    out("# Phase 10 scalability - measured results")
    out(f"Memory: {any_ok['memory_method']}; peaks are medians of 3 fresh processes.")
    out(f"torch threads: {any_ok['torch_threads']}. Runs killed when footprint > {budget / GIB:.1f} GiB "
        f"(8 GB machine).")
    out()
    hdr = ("| cells | nodes | edges | graph tensors (MiB) | .pt file (MiB) | build time (s) | build peak (GiB) "
           "| inference fwd peak (GiB) | inference fwd time (s) | train fwd peak (GiB) | train fwd+bwd peak (GiB) "
           "| fwd+bwd time (s) |")
    out(hdr)
    out("|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|")
    fmt_t = lambda v: f"{v:.2f}"
    for n in SCALES:
        g = by["build_tiled"][n]
        fwd = by["fwd_nograd"].get(n, {})
        file_mib = f"{fwd['graph_file_bytes'] / 2**20:.0f}" if "graph_file_bytes" in fwd else "-"
        out(f"| {n:,} | {g['nodes']:,} | {g['edges']:,} | {g['graph_tensor_bytes'] / 2**20:.0f} | {file_mib} "
            f"| {cell('build_tiled', n, 'time_s', fmt_t)} | {cell('build_tiled', n)} "
            f"| {cell('fwd_nograd', n)} | {cell('fwd_nograd', n, 'time_s', fmt_t)} "
            f"| {cell('fwd_grad', n)} | {cell('fwd_bwd', n)} | {cell('fwd_bwd', n, 'time_s', fmt_t)} |")
    out()

    # Load-spike check: forward peak must exceed the post-load peak.
    spikes = [(m, n) for m in FWD_MODES for n, r in by[m].items()
              if r["status"] == "ok" and r["peak_bytes"] <= r["peak_after_load_bytes"]]
    out(f"Runs whose overall peak was set during graph load, not the forward pass: {spikes or 'none'}")
    out()

    out("## Run-to-run spread (min-max of 3 repeats, GiB)")
    for m in FWD_MODES + ["build_tiled"]:
        worst = max(((max(r["peaks_bytes"]) - min(r["peaks_bytes"])) / r["peak_bytes"], n)
                    for n, r in by[m].items() if r["status"] == "ok")
        big = max((max(r["peaks_bytes"]) - min(r["peaks_bytes"])) / r["peak_bytes"]
                  for n, r in by[m].items() if r["status"] == "ok" and n >= FIT_MIN_CELLS)
        out(f"  {m:<11} worst spread {100 * worst[0]:.0f}% of median (at {worst[1]:,} cells); "
            f"worst at >= {FIT_MIN_CELLS:,} cells: {100 * big:.0f}%")
    out()

    out(f"## Linear fits, peak = a + b*cells, on measured points with cells >= {FIT_MIN_CELLS:,}")
    fits = {}
    for m in FWD_MODES + ["build_tiled"]:
        x, y = ok_points(by, m, FIT_MIN_CELLS)
        a, b, r2, maxres = linfit(x, y)
        ha, hb, _, _ = linfit(x[:-1], y[:-1])
        held = (ha + hb * x[-1]) / y[-1] - 1
        cross = (LIMIT - a) / b
        fits[m] = (a, b, x.max(), cross)
        edges_per_cell = np.mean([r["edges"] / n for n, r in by["build_tiled"].items()
                                  if r["status"] == "ok" and n >= FIT_MIN_CELLS])
        out(f"  {m:<11} points={len(x)} ({int(x.min()):,}-{int(x.max()):,})  a={a / GIB:.3f} GiB  "
            f"b={b / 1024:.2f} KiB/cell  R^2={r2:.4f}  max residual {100 * maxres:.1f}%")
        out(f"  {'':<11} held-out: fit without {int(x[-1]):,} predicts it within {100 * held:+.1f}%")
        out(f"  {'':<11} -> 12 GiB crossover at ~{cross:,.0f} cells "
            f"(~{cross * edges_per_cell / 1e6:.1f}M edges at the measured {edges_per_cell:.2f} edges/cell); "
            f"at 1.5M cells: {(a + b * 1.5e6) / GIB:.1f} GiB"
            + ("  [EXTRAPOLATED]" if cross > x.max() else ""))
    out()

    out("## O(n^2) components, peak = a + c*n^2 on all measured points")
    for m, unit, targets in [
        ("build_dense", "cells", [(35_000, "smallest CircuitNet-scale design"), (1_500_000, "largest")]),
        ("rank_loss", "gcells", [(15_376, "gcells in the 35K-cell synthetic design"),
                                 (656_100, "gcells in the 1.5M-cell synthetic design")]),
    ]:
        n, y = ok_points(by, m)
        a, c, r2, maxres = quadfit(n, y)
        out(f"  {m:<11} points={len(n)} ({int(n.min()):,}-{int(n.max()):,} {unit})  c={c:.1f} B/{unit[:-1]}^2  "
            f"R^2={r2:.4f}  max residual {100 * maxres:.1f}%  killed at {killed_at(by, m):,} {unit}")
        out(f"  {'':<11} largest that fits 12 GiB: ~{np.sqrt((LIMIT - a) / c):,.0f} {unit}")
        for t, label in targets:
            out(f"  {'':<11} at {t:,} {unit} ({label}): {(a + c * t ** 2) / GIB:,.0f} GiB  [EXTRAPOLATED]")
    out()

    text = "\n".join(lines)
    print(text)
    with open(TABLE_OUT, "w") as f:
        f.write(text + "\n")
    plot(by, fits, budget)
    print(f"\nWrote {TABLE_OUT} and {FIG_OUT}")


def plot(by, fits, budget):
    plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    M = 1e6
    xmax = 1.6e6

    ax.axvspan(35_000 / M, 1.5, color="#f1f0ec", zorder=0)
    ax.text(0.77, 17.3, "CircuitNet design range (35K – 1.5M cells)", ha="center",
            color=INK2, fontsize=13)

    ax.axhline(LIMIT / GIB, color=INK, lw=2)
    ax.text(0.02, LIMIT / GIB + 0.25, "12 GB memory limit", color=INK, fontsize=15, fontweight="bold")
    ax.axhline(budget / GIB, color=INK2, lw=1, ls=(0, (2, 3)))
    ax.text(1.58, budget / GIB - 0.25, "measurement cap (8 GB laptop)",
            color=INK2, fontsize=12, ha="right", va="top")

    series = [
        ("fwd_bwd", C_TRAIN, "Training step\n(forward + backward)"),
        ("fwd_nograd", C_INFER, "Inference\n(one forward pass)"),
        ("build_tiled", C_BUILD, "Graph construction\n(tiled kNN)"),
    ]
    for mode, color, label in series:
        x, y = ok_points(by, mode)
        ax.plot(x / M, y / GIB, "-o", color=color, lw=2.5, ms=8, mec="white", mew=1.5, zorder=3)
        a, b, xm, cross = fits[mode]
        x_end = min(max(cross, xm), xmax)
        if x_end > xm:
            xe = np.array([xm, x_end])
            ax.plot(xe / M, (a + b * xe) / GIB, "--", color=color, lw=2.5, zorder=2)
        if cross <= xmax:
            ax.plot([cross / M], [LIMIT / GIB], "o", color=color, ms=13, mec=INK, mew=1.5, zorder=4)
            ax.text(cross / M + 0.03, LIMIT / GIB - 0.35,
                    f"{label.splitlines()[0]} {label.splitlines()[1]}\nhits 12 GB at ~{cross / M:.1f}M cells",
                    color=INK, fontsize=14, va="top", fontweight="bold")
        else:
            ax.text(x_end / M + 0.015, (a + b * x_end) / GIB,
                    label + (f"\nreaches 12 GB at ~{cross / M:.1f}M" if cross < 5 * xmax else ""),
                    color=INK, fontsize=13, va="center")
        k = killed_at(by, mode)
        if k:
            ax.plot([k / M], [budget / GIB], "x", color=color, ms=12, mew=3, zorder=4)

    # Original O(n^2) dense-kNN construction: measured points + quadratic extrapolation.
    n, y = ok_points(by, "build_dense")
    a, c, _, _ = quadfit(n, y)
    ax.plot(n / M, y / GIB, "-o", color=C_DENSE, lw=2.5, ms=8, mec="white", mew=1.5, zorder=3)
    ne = np.linspace(n.max(), np.sqrt((18 * GIB - a) / c), 50)
    ax.plot(ne / M, (a + c * ne ** 2) / GIB, "--", color=C_DENSE, lw=2.5)
    k = killed_at(by, "build_dense")
    ax.plot([k / M], [budget / GIB], "x", color=C_DENSE, ms=12, mew=3, zorder=4)
    x_at15 = np.sqrt((15.5 * GIB - a) / c) / M
    ax.annotate(f"Original graph builder (dense n×n kNN):\nhits 12 GB at ~{np.sqrt((LIMIT - a) / c) / 1e3:.0f}K cells, "
                f"~{(a + c * 35_000 ** 2) / GIB:.0f} GB at 35K",
                (x_at15, 15.5), xytext=(0.06, 15.5), textcoords="data",
                color=INK, fontsize=13, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))

    ax.plot([], [], "-o", color=INK2, label="measured (median of 3 runs)")
    ax.plot([], [], "--", color=INK2, label="linear / quadratic extrapolation")
    ax.plot([], [], "x", color=INK2, mew=3, ms=10, ls="none", label="killed: exceeded measurement cap")
    ax.legend(loc="upper left", bbox_to_anchor=(0.06, 0.62), frameon=False, fontsize=13)

    ax.set_xlim(0, xmax / M)
    ax.set_ylim(0, 18)
    ax.set_xlabel("design size (million cells)")
    ax.set_ylabel("peak memory (GB)")
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_title("Full-graph training step: measured to 500K cells, reaches 12 GB at ~1.1M (extrapolated)",
                 loc="left", fontsize=17, pad=14)
    fig.text(0.01, 0.01, "Measured on CPU (Apple M1, 8 GB), synthetic graphs, peak physical footprint. "
             "GPU memory not measured. GB = GiB (2^30 bytes).", color=INK2, fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.82, top=0.90, bottom=0.12)
    fig.savefig(FIG_OUT, dpi=160)


if __name__ == "__main__":
    main()
