"""Phase 10 scalability measurements: where does the current approach break?

Every measurement runs in a fresh subprocess (peak memory is a per-process
high-water mark - see memtrack.py), watched by the parent, which polls the
child's physical footprint and SIGKILLs it once it crosses --budget-gib.
That keeps an 8 GB machine usable; a killed run is recorded as "exceeded
budget", which is itself a measurement (peak > budget), not a guess.

Modes:
  build_tiled  generate_placement + build_graph(knn_fn=knn_tiled)   (Phase 9 path)
  build_dense  generate_placement + build_graph() with the original O(n^2) kNN
  fwd_nograd   one CongestionGNN forward pass, eval + torch.no_grad()  (inference)
  fwd_grad     one forward pass with autograd on (activations retained, as in training)
  fwd_bwd      forward + backward of a scalar loss (one full training step minus optimizer)
  rank_loss    train.pairwise_ranking_loss on n random gcell predictions (O(n^2))

The fwd_* modes load a graph saved by `prepare`, record the footprint after
loading as the baseline, run a warm-up forward pass on a tiny graph first
(so one-time kernel/allocator init isn't billed to the measured pass), then
measure. The peak after loading is also recorded, so a load-time spike
can't masquerade as forward-pass memory.

Usage:
    python src/scalability.py prepare --cells 35000 100000 ...
    python src/scalability.py run --mode fwd_nograd --cells 35000 100000 ... [--budget-gib 5.5]
    python src/scalability.py report
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memtrack

GRAPH_DIR = "data/processed"
RESULTS = "notes/results/phase10_scalability.json"
GIB = 2 ** 30


def graph_path(n):
    return os.path.join(GRAPH_DIR, f"synth_{n}.pt")


# ---------------------------------------------------------------------------
# worker side (runs in the measured subprocess)
# ---------------------------------------------------------------------------

def worker(mode, n):
    import numpy as np
    import torch
    torch.manual_seed(0)
    result = dict(mode=mode, cells=n, torch_threads=torch.get_num_threads())

    if mode in ("build_tiled", "build_dense"):
        from build_graph import build_graph
        from synthetic import generate_placement, graph_nbytes, knn_tiled
        base = memtrack.current_bytes()
        t0 = time.perf_counter()
        placement, congestion = generate_placement(n, 0)
        kwargs = dict(knn_fn=knn_tiled) if mode == "build_tiled" else {}
        data = build_graph(placement, congestion, **kwargs)
        result["time_s"] = time.perf_counter() - t0
        result["graph_tensor_bytes"] = graph_nbytes(data)
        result["nodes"] = sum(data[nt].num_nodes for nt in data.node_types)
        result["edges"] = sum(data[et].edge_index.shape[1] for et in data.edge_types)

    elif mode == "rank_loss":
        from train import pairwise_ranking_loss
        pred = torch.randn(n, 2, requires_grad=True)
        target = torch.randn(n, 2)
        base = memtrack.current_bytes()
        t0 = time.perf_counter()
        loss = pairwise_ranking_loss(pred, target)
        loss.backward()
        result["time_s"] = time.perf_counter() - t0

    else:
        from model import CongestionGNN
        from synthetic import generate_graph, graph_nbytes
        warm = generate_graph(2000, seed=1)
        in_dims = {nt: warm[nt].x.shape[1] for nt in warm.node_types}
        model = CongestionGNN(in_dims, list(warm.edge_types))
        with torch.no_grad():
            model(warm)
        del warm

        data = torch.load(graph_path(n), weights_only=False)
        result["graph_tensor_bytes"] = graph_nbytes(data)
        result["graph_file_bytes"] = os.path.getsize(graph_path(n))
        result["nodes"] = sum(data[nt].num_nodes for nt in data.node_types)
        result["edges"] = sum(data[et].edge_index.shape[1] for et in data.edge_types)
        result["peak_after_load_bytes"] = memtrack.peak_bytes()
        base = memtrack.current_bytes()

        t0 = time.perf_counter()
        if mode == "fwd_nograd":
            model.eval()
            with torch.no_grad():
                out = model(data)
        elif mode == "fwd_grad":
            out = model(data)
        elif mode == "fwd_bwd":
            out = model(data)
            out.square().mean().backward()
        else:
            raise ValueError(mode)
        result["time_s"] = time.perf_counter() - t0
        result["out_shape"] = list(out.shape)

    peak = memtrack.peak_bytes()
    result.update(baseline_bytes=base, peak_bytes=peak, peak_minus_baseline_bytes=peak - base,
                  memory_method=memtrack.method())
    print("RESULT " + json.dumps(result), flush=True)


# ---------------------------------------------------------------------------
# parent side
# ---------------------------------------------------------------------------

def run_one(mode, n, budget_bytes, poll_s=0.02):
    cmd = [sys.executable, os.path.abspath(__file__), "_worker", mode, str(n)]
    result = run_cmd(cmd, budget_bytes, poll_s)
    result.setdefault("mode", mode)
    result.setdefault("cells", n)
    return result


def run_cmd(cmd, budget_bytes, poll_s=0.02):
    """Run a worker command that prints one 'RESULT {json}' line, SIGKILLing
    it if its physical footprint exceeds budget_bytes."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    max_seen, killed = 0, False
    t0 = time.perf_counter()
    while proc.poll() is None:
        try:
            cur = memtrack.current_bytes(proc.pid)
        except OSError:
            break
        max_seen = max(max_seen, cur)
        if cur > budget_bytes:
            proc.send_signal(signal.SIGKILL)
            killed = True
            break
        time.sleep(poll_s)
    stdout, stderr = proc.communicate()
    wall = time.perf_counter() - t0

    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
            result["status"] = "ok"
            result["wall_s"] = wall
            return result
    return dict(wall_s=wall, max_polled_bytes=max_seen,
                budget_bytes=budget_bytes,
                status="exceeded_budget" if killed else "error",
                stderr_tail=None if killed else stderr[-2000:])


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return []


def save_result(result):
    results = [r for r in load_results()
               if not (r["mode"] == result["mode"] and r["cells"] == result["cells"])]
    results.append(result)
    results.sort(key=lambda r: (r["mode"], r["cells"]))
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=1)


def cmd_prepare(args):
    import torch
    from synthetic import generate_graph
    os.makedirs(GRAPH_DIR, exist_ok=True)
    for n in args.cells:
        if os.path.exists(graph_path(n)) and not args.force:
            print(f"{graph_path(n)} exists, skipping")
            continue
        torch.save(generate_graph(n, seed=0), graph_path(n))
        print(f"wrote {graph_path(n)} ({os.path.getsize(graph_path(n)) / 2**20:.0f} MiB)")


def run_repeated(mode, n, budget_bytes, repeats):
    """Peak memory varies run to run (measured, likely multithreaded
    allocation/free timing), so each point is `repeats` fresh processes:
    median reported, all peaks kept. Any repeat over budget -> the point is
    recorded as exceeded_budget (its median is unknown)."""
    runs = []
    for _ in range(repeats):
        r = run_one(mode, n, budget_bytes)
        runs.append(r)
        if r["status"] != "ok":
            break
    ok = [r for r in runs if r["status"] == "ok"]
    if len(ok) < len(runs):
        bad = runs[-1]
        bad["ok_peaks_bytes"] = [r["peak_bytes"] for r in ok]
        return bad

    def med(key):
        vals = sorted(r[key] for r in ok)
        return vals[len(vals) // 2]
    result = dict(ok[0])
    for key in ("peak_bytes", "peak_minus_baseline_bytes", "baseline_bytes", "time_s", "wall_s"):
        result[key] = med(key)
    result["peaks_bytes"] = [r["peak_bytes"] for r in ok]
    result["times_s"] = [r["time_s"] for r in ok]
    result["repeats"] = len(ok)
    return result


def cmd_run(args):
    budget = int(args.budget_gib * GIB)
    for n in args.cells:
        r = run_repeated(args.mode, n, budget, args.repeats)
        save_result(r)
        if r["status"] == "ok":
            print(f"{args.mode:<12} n={n:>9,}  peak median={r['peak_bytes'] / GIB:6.3f} GiB  "
                  f"[{min(r['peaks_bytes']) / GIB:.3f}-{max(r['peaks_bytes']) / GIB:.3f}]  "
                  f"(+{r['peak_minus_baseline_bytes'] / GIB:6.3f} over baseline)  time={r['time_s']:.2f}s")
        elif r["status"] == "exceeded_budget":
            print(f"{args.mode:<12} n={n:>9,}  KILLED: footprint passed {args.budget_gib} GiB budget "
                  f"after {r['wall_s']:.1f}s")
        else:
            print(f"{args.mode:<12} n={n:>9,}  ERROR\n{r['stderr_tail']}")
        if r["status"] != "ok" and args.stop_on_fail:
            break


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        worker(sys.argv[2], int(sys.argv[3]))
        return

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cells", type=int, nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--mode", required=True,
                   choices=["build_tiled", "build_dense", "fwd_nograd", "fwd_grad", "fwd_bwd", "rank_loss"])
    p.add_argument("--cells", type=int, nargs="+", required=True,
                   help="design size in cells (rank_loss: number of gcells)")
    p.add_argument("--budget-gib", type=float, default=5.5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--stop-on-fail", action="store_true",
                   help="skip larger sizes once one exceeds the budget")
    sub.add_parser("report")
    args = parser.parse_args()

    if args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        import scalability_report
        scalability_report.main()


if __name__ == "__main__":
    main()
