"""Phase 12 gate: run both baselines on synthetic graphs, check their output
matches CongestionGNN's shape/dtype, and push them through every function
in src/metrics.py.

- DensityBaseline is fitted on 35K-cell graphs with seeds 1-3 and evaluated
  on seed-0 graphs (35K, 100K, 1.5M), so no evaluation graph is in its fit.
- At 1.5M cells, model(data) doesn't fit this 8 GB machine (Phase 10), so
  the model-side shape comes from partition.reference_forward (bit-identical
  to model(data) where both fit - Phase 11).
- kendall_tau is O(n^2) in memory; it is measured in watchdogged
  subprocesses on growing gcell counts rather than called blindly at 1.5M.

Usage:
    python tests/phase12_baselines_gate.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
import memtrack

GIB = 2 ** 30
OUT = "notes/results/phase12_baselines.json"


def avg_rank_spearman(a, b):
    """Spearman with average ranks for ties (the textbook definition), to
    compare against metrics.spearman_corr's arbitrary tie-breaking."""
    def rank(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return sums[inv] / counts[inv]
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def kendall_worker(n):
    from metrics import kendall_tau
    rng = np.random.default_rng(0)
    a, b = rng.random(n), rng.random(n)
    base = memtrack.current_bytes()
    t0 = time.perf_counter()
    kendall_tau(a, b)
    print("RESULT " + json.dumps(dict(n=n, time_s=time.perf_counter() - t0,
                                      peak_bytes=memtrack.peak_bytes(), baseline_bytes=base)), flush=True)


def main():
    from baselines import DensityBaseline, RUDYBaseline
    from metrics import combined_congestion, fraction_above, kendall_tau, spearman_corr
    from model import CongestionGNN
    from partition import reference_forward
    from scalability import run_cmd
    from synthetic import generate_graph

    results = dict(eval={}, kendall=[])

    t0 = time.perf_counter()
    train = [generate_graph(35_000, seed=s) for s in (1, 2, 3)]
    dens = DensityBaseline().fit(train)
    results["density_fit"] = dict(train="35K cells, seeds 1,2,3", time_s=time.perf_counter() - t0,
                                  n_gcells=sum(g["gcell"].num_nodes for g in train),
                                  weight=dens.weight.tolist())
    del train
    rudy = RUDYBaseline()

    for n in (35_000, 100_000, 1_500_000):
        path = f"data/processed/synth_{n}.pt"
        data = torch.load(path, weights_only=False) if os.path.exists(path) else generate_graph(n, seed=0)
        ny, nx = (int(v) for v in data["gcell"].grid_shape)
        y = data["gcell"].y_ratio
        torch.manual_seed(0)
        model = CongestionGNN({nt: data[nt].x.shape[1] for nt in data.node_types}, list(data.edge_types)).eval()
        with torch.no_grad():
            ref = model(data) if n <= 100_000 else reference_forward(model, data)
        row = dict(n_gcells=int(data["gcell"].num_nodes), grid=[ny, nx],
                   model_shape=list(ref.shape), model_dtype=str(ref.dtype),
                   model_path="model(data)" if n <= 100_000 else "reference_forward")
        for name, b in (("rudy", rudy), ("density", dens)):
            t = time.perf_counter()
            pred = b(data)
            dt = time.perf_counter() - t
            grid = pred.reshape(ny, nx, 2).numpy()
            r = dict(time_s=dt, shape=list(pred.shape), dtype=str(pred.dtype),
                     shape_matches_model=list(pred.shape) == list(ref.shape) and pred.dtype == ref.dtype,
                     finite=bool(torch.isfinite(pred).all()),
                     spearman=[spearman_corr(pred[:, c], y[:, c]) for c in range(2)],
                     spearman_avg_ties=[avg_rank_spearman(pred[:, c], y[:, c]) for c in range(2)],
                     frac_tied_pred=[1 - len(np.unique(pred[:, c].numpy())) / len(pred) for c in range(2)],
                     combined_congestion_shape=list(combined_congestion(grid).shape),
                     fraction_above_0_9=fraction_above(grid, 0.9),
                     label_fraction_above_0_9=fraction_above(y.reshape(ny, nx, 2).numpy(), 0.9))
            # kendall_tau is O(n^2): run it on a fixed 2000-gcell subsample
            idx = torch.randperm(len(pred), generator=torch.Generator().manual_seed(0))[:2000]
            r["kendall_subsample_2000"] = [kendall_tau(pred[idx, c], y[idx, c]) for c in range(2)]
            row[name] = r
            print(f"{n:>9,} cells {name:<8} shape {tuple(pred.shape)} {pred.dtype} (model {tuple(ref.shape)} "
                  f"{ref.dtype}) match={r['shape_matches_model']}  {dt:.2f}s  "
                  f"spearman H/V {r['spearman'][0]:.3f}/{r['spearman'][1]:.3f} "
                  f"(avg-rank ties {r['spearman_avg_ties'][0]:.3f}/{r['spearman_avg_ties'][1]:.3f}, "
                  f"tied preds {100 * r['frac_tied_pred'][0]:.1f}%)  "
                  f"kendall(2000) {r['kendall_subsample_2000'][0]:.3f}/{r['kendall_subsample_2000'][1]:.3f}  "
                  f"frac>0.9 {r['fraction_above_0_9']:.3f} (label {r['label_fraction_above_0_9']:.3f})")
        results["eval"][str(n)] = row
        del data, ref, model

    for n in (2_000, 4_000, 8_000, 15_376):
        r = run_cmd([sys.executable, os.path.abspath(__file__), "_kendall", str(n)], int(5.5 * GIB))
        r["n"] = n
        results["kendall"].append(r)
        print(f"kendall_tau on n={n:,}: " + (f"peak {r['peak_bytes'] / GIB:.2f} GiB, {r['time_s']:.1f}s"
                                             if r["status"] == "ok" else r["status"]))

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "_kendall":
        kendall_worker(int(sys.argv[2]))
    else:
        main()
