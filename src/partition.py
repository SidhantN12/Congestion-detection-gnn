"""Spatial partitioning of a full-design HeteroData graph into training-sized
subgraphs, with L-hop halos and degree capping for high-fanout nets.

Why spatial, not random neighbour sampling: congestion is a spatial field
over the gcell grid. A partition here is a contiguous rectangle of gcells
(its *core*), and every core gcell is predicted and supervised inside
exactly one partition. Neighbour-sampling mini-batches would instead
scatter targets across the die and thin out each target's neighbourhood.

Partitioning (spatial_parts): the gcell grid is cut into px column strips
with ~equal cell counts, then each strip into py row blocks with ~equal
cell counts - balanced by work, not by area (the placement is clustered).

Halo (expand_halo): CongestionGNN has L=3 message-passing layers, so a
gcell's output depends on every node within 3 hops *upstream* (following
edges backwards, dst -> src). Taking that L-hop in-neighbourhood of the
core gcells, plus all edges among those nodes, reproduces the full-graph
outputs of the core gcells exactly: a node at distance d < L has all of
its in-edges inside the set, which is all layer L-d needs.

Degree capping: a 3-hop halo passes through each core cell's nets and then
out to every pin of those nets. For hub nets (the pre-CTS clock touches
every flop - 150K cells at 1.5M) this would pull a large fraction of the
die into every partition. Nets with full-graph degree > degree_cap are
therefore not expanded: they keep only pins that are already in the
partition (core + halo), and at most degree_cap of those (a seeded
random sample). Their mean-aggregated embedding becomes an estimate over
local pins, so core gcells within reach of a capped net are no longer
exact - the deviation is measured, not assumed (see verify).

Memory budget (plan): peak training-step memory is estimated from a
partition's edge count, using the per-edge coefficient fitted to the
Phase 10 measurements (fwd+bwd peak vs edges, 200K-500K cells). The grid
is refined until the largest partition's estimate fits the budget.

Usage:
    python src/partition.py plan      --cells 1500000 --budget-gib 2 [--degree-cap 512]
    python src/partition.py reference --cells 1500000
    python src/partition.py verify    --cells 1500000
    python src/partition.py measure   --cells 1500000 [--budget-gib 5.5]
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memtrack

NUM_HOPS = 3  # = CongestionGNN num_layers
PIN = ("cell", "pin", "net")
# Phase 10, fwd_bwd peak vs edge count, least squares on 200K/350K/500K-cell
# graphs (pred/actual 1.007, 0.992, 1.003). Whole-process peak, incl. torch baseline.
EST_BASE_BYTES = 0.1698 * 2 ** 30
EST_BYTES_PER_EDGE = 534.7
GIB = 2 ** 30


# ---------------------------------------------------------------------------
# partitioning
# ---------------------------------------------------------------------------

def _balanced_cuts(counts, k):
    """Group consecutive bins into <=k groups of ~equal total count."""
    cum = np.cumsum(counts) - counts / 2.0  # bin midpoints in cumulative space
    total = max(counts.sum(), 1)
    return np.minimum((cum / total * k).astype(np.int64), k - 1)


def spatial_parts(data, px, py):
    """Partition id per gcell: px column strips x py row blocks, balanced by cell count."""
    ny, nx = (int(v) for v in data["gcell"].grid_shape)
    per_gcell = torch.bincount(data["cell", "in", "gcell"].edge_index[1],
                               minlength=nx * ny).numpy().reshape(ny, nx).astype(np.float64)
    col_group = _balanced_cuts(per_gcell.sum(0), px)  # (nx,)
    part = np.empty((ny, nx), dtype=np.int64)
    for c in range(px):
        cols = col_group == c
        if not cols.any():
            continue
        row_group = _balanced_cuts(per_gcell[:, cols].sum(1), py)  # (ny,)
        part[:, cols] = (c * py + row_group)[:, None]
    # relabel to 0..P-1 (a group can end up empty on tiny grids)
    _, part = np.unique(part.reshape(-1), return_inverse=True)
    return torch.from_numpy(part)


def capped_nets(data, degree_cap):
    fanout = torch.bincount(data[PIN].edge_index[1], minlength=data["net"].num_nodes)
    if degree_cap is None:
        return torch.zeros(data["net"].num_nodes, dtype=torch.bool)
    return fanout > degree_cap


def expand_halo(data, core_gcells, num_hops=NUM_HOPS, capped=None):
    """Node masks of the num_hops upstream neighbourhood of the core gcells."""
    masks = {nt: torch.zeros(data[nt].num_nodes, dtype=torch.bool) for nt in data.node_types}
    masks["gcell"][core_gcells] = True
    for _ in range(num_hops):
        new = {nt: m.clone() for nt, m in masks.items()}
        for et in data.edge_types:
            src_t, _, dst_t = et
            src, dst = data[et].edge_index
            sel = masks[dst_t][dst]
            if et == PIN and capped is not None:
                sel &= ~capped[dst]  # capped nets pull in no new pins
            new[src_t][src[sel]] = True
        masks = new
    return masks


def no_halo_masks(data, core_gcells):
    """The naive spatial partition: core gcells, the cells in them, and the
    nets on those cells - nothing beyond the partition's own rectangle."""
    masks = {nt: torch.zeros(data[nt].num_nodes, dtype=torch.bool) for nt in data.node_types}
    masks["gcell"][core_gcells] = True
    cell_idx, gcell_idx = data["cell", "in", "gcell"].edge_index
    masks["cell"][cell_idx[masks["gcell"][gcell_idx]]] = True
    c, n = data[PIN].edge_index
    masks["net"][n[masks["cell"][c]]] = True
    return masks


def _kept_edges(data, masks, capped, degree_cap, seed):
    """Per edge type, a bool mask over the full graph's edges kept in the subgraph."""
    kept = {}
    for et in data.edge_types:
        src_t, _, dst_t = et
        src, dst = data[et].edge_index
        keep = masks[src_t][src] & masks[dst_t][dst]
        if et == PIN and degree_cap is not None:
            cand = torch.nonzero(keep & capped[dst]).squeeze(1)
            if len(cand):
                g = torch.Generator().manual_seed(seed)
                cand = cand[torch.randperm(len(cand), generator=g)]
                cand = cand[torch.argsort(dst[cand], stable=True)]  # grouped by net, random within
                d = dst[cand]
                group_start = torch.searchsorted(d, d, side="left")
                rank = torch.arange(len(cand)) - group_start
                keep[cand[rank >= degree_cap]] = False
        kept[et] = keep
    return kept


def estimate_peak_bytes(n_edges):
    return EST_BASE_BYTES + EST_BYTES_PER_EDGE * n_edges


def partition_edge_count(data, masks, capped, degree_cap, seed=0):
    return int(sum(k.sum() for k in _kept_edges(data, masks, capped, degree_cap, seed).values()))


def build_subgraph(data, core_gcells, masks, capped=None, degree_cap=None, seed=0):
    """Induced subgraph on the masked nodes. Adds per node type `n_id` (index in
    the full graph) and gcell `core` (True for the gcells this partition owns)."""
    sub = HeteroData()
    remap = {}
    for nt in data.node_types:
        n_id = torch.nonzero(masks[nt]).squeeze(1)
        remap[nt] = torch.full((data[nt].num_nodes,), -1, dtype=torch.long)
        remap[nt][n_id] = torch.arange(len(n_id))
        sub[nt].x = data[nt].x[n_id]
        sub[nt].n_id = n_id
    for key in ("y_ratio", "y_demand"):
        if key in data["gcell"]:
            sub["gcell"][key] = data["gcell"][key][sub["gcell"].n_id]
    core = torch.zeros(data["gcell"].num_nodes, dtype=torch.bool)
    core[core_gcells] = True
    sub["gcell"].core = core[sub["gcell"].n_id]

    for et, keep in _kept_edges(data, masks, capped, degree_cap, seed).items():
        src_t, _, dst_t = et
        src, dst = data[et].edge_index[:, keep]
        sub[et].edge_index = torch.stack([remap[src_t][src], remap[dst_t][dst]])
    return sub


def make_partitions(data, px, py, num_hops=NUM_HOPS, degree_cap=512, halo=True):
    """Yields (part_id, subgraph) for a px x py spatial partitioning."""
    part = spatial_parts(data, px, py)
    capped = capped_nets(data, degree_cap)
    for p in range(int(part.max()) + 1):
        core = torch.nonzero(part == p).squeeze(1)
        masks = expand_halo(data, core, num_hops, capped) if halo else no_halo_masks(data, core)
        yield p, build_subgraph(data, core, masks, capped, degree_cap, seed=p)


def plan(data, budget_bytes, num_hops=NUM_HOPS, degree_cap=512, max_k=64):
    """Smallest k (k x k spatial grid) whose largest partition's estimated
    training-step peak fits in budget_bytes. Returns (k, per-partition info)."""
    total_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
    k = max(1, int(np.ceil(np.sqrt(estimate_peak_bytes(total_edges) / budget_bytes))))
    capped = capped_nets(data, degree_cap)
    while k <= max_k:
        part = spatial_parts(data, k, k)
        info = []
        for p in range(int(part.max()) + 1):
            core = torch.nonzero(part == p).squeeze(1)
            masks = expand_halo(data, core, num_hops, capped)
            e = partition_edge_count(data, masks, capped, degree_cap, seed=p)
            info.append(dict(part=p, core_gcells=len(core), edges=e,
                             nodes={nt: int(m.sum()) for nt, m in masks.items()},
                             est_peak_bytes=estimate_peak_bytes(e)))
            if info[-1]["est_peak_bytes"] > budget_bytes:
                break  # this k already fails; refine
        else:
            return k, info
        print(f"  k={k}: partition {p} estimated at {info[-1]['est_peak_bytes'] / GIB:.2f} GiB > budget, refining")
        k += 1
    raise RuntimeError(f"no k <= {max_k} fits the budget")


# ---------------------------------------------------------------------------
# full-graph reference without full-graph memory
# ---------------------------------------------------------------------------

@torch.no_grad()
def reference_forward(model, data, edge_chunk=2_000_000):
    """CongestionGNN forward, computed relation by relation with the edge
    gather done in chunks - same math as model(data) (SAGEConv mean
    aggregation: lin_l(mean_j x_j) + lin_r(x_i), HeteroConv sum over
    relations, residual + LayerNorm + ReLU), but without materialising a
    (num_edges x hidden) message tensor per relation. Only used as ground
    truth for graphs too large for model(data) on this machine; checked
    against model(data) in tests/test_partition.py."""
    x = {nt: F.relu(model.encoders[nt](data[nt].x)) for nt in model.node_types}
    for conv, norm in zip(model.convs, model.norms):
        upd = {nt: None for nt in model.node_types}
        for et, sage in conv.convs.items():
            src_t, _, dst_t = et
            src, dst = data[et].edge_index
            n_dst = data[dst_t].num_nodes
            agg = torch.zeros(n_dst, x[src_t].shape[1])
            for s in range(0, len(src), edge_chunk):
                agg.index_add_(0, dst[s:s + edge_chunk], x[src_t][src[s:s + edge_chunk]])
            agg /= torch.bincount(dst, minlength=n_dst).clamp(min=1).unsqueeze(1).to(agg.dtype)
            out = sage.lin_l(agg)
            del agg
            out += sage.lin_r(x[dst_t])
            upd[dst_t] = out if upd[dst_t] is None else upd[dst_t].add_(out)
        x = {nt: F.relu(norm[nt](upd[nt] + x[nt])) for nt in model.node_types}
    return model.head(x["gcell"])


# ---------------------------------------------------------------------------
# CLI: plan / reference / verify / measure
# ---------------------------------------------------------------------------

def graph_path(n):
    return f"data/processed/synth_{n}.pt"


def part_dir(n):
    return f"data/processed/parts_{n}"


def results_path(n):
    return f"notes/results/phase11_partition_{n}.json"


def load_results(n):
    if os.path.exists(results_path(n)):
        with open(results_path(n)) as f:
            return json.load(f)
    return {}


def save_results(n, update):
    res = load_results(n)
    res.update(update)
    with open(results_path(n), "w") as f:
        json.dump(res, f, indent=1)


def make_model(data):
    from model import CongestionGNN
    torch.manual_seed(0)
    return CongestionGNN({nt: data[nt].x.shape[1] for nt in data.node_types}, list(data.edge_types))


def cmd_plan(args):
    data = torch.load(graph_path(args.cells), weights_only=False)
    t0 = time.perf_counter()
    k, info = plan(data, args.budget_gib * GIB, degree_cap=args.degree_cap)
    t_plan = time.perf_counter() - t0
    capped = capped_nets(data, args.degree_cap)

    os.makedirs(part_dir(args.cells), exist_ok=True)
    for old in glob.glob(os.path.join(part_dir(args.cells), "part_*.pt")):
        os.remove(old)
    t0 = time.perf_counter()
    for p, sub in make_partitions(data, k, k, degree_cap=args.degree_cap):
        torch.save(sub, os.path.join(part_dir(args.cells), f"part_{p:03d}.pt"))
    t_build = time.perf_counter() - t0

    print(f"budget {args.budget_gib} GiB -> {k}x{k} grid, {len(info)} partitions "
          f"(plan {t_plan:.1f}s, build+save {t_build:.1f}s); "
          f"{int(capped.sum())} nets capped at degree {args.degree_cap}")
    save_results(args.cells, dict(plan=dict(budget_gib=args.budget_gib, degree_cap=args.degree_cap,
                                            k=k, n_parts=len(info), capped_nets=int(capped.sum()),
                                            plan_time_s=t_plan, build_time_s=t_build, parts=info)))


def cmd_reference(args):
    data = torch.load(graph_path(args.cells), weights_only=False)
    model = make_model(data)
    base = memtrack.current_bytes()
    t0 = time.perf_counter()
    ref = reference_forward(model, data)
    t = time.perf_counter() - t0
    torch.save(ref, f"data/processed/ref_{args.cells}.pt")
    peak = memtrack.peak_bytes()
    print(f"reference forward: {t:.1f}s, peak {peak / GIB:.2f} GiB (+{(peak - base) / GIB:.2f} over post-load)")
    save_results(args.cells, dict(reference=dict(time_s=t, peak_bytes=peak, baseline_bytes=base)))


def density_sums(g, core_only_ids=None):
    """Per-gcell (sum of cell site-area, pin count) recomputed from the cells
    and pins present in graph g - the two quantities behind build_graph's
    cell_density / pin_density features (both are these sums / gcell area)."""
    cell_idx, gcell_idx = g["cell", "in", "gcell"].edge_index
    area = g["cell"].x[:, 0].double() * g["cell"].x[:, 1].double()
    area_sum = torch.zeros(g["gcell"].num_nodes, dtype=torch.float64).index_add_(0, gcell_idx, area[cell_idx])
    # pins per cell counted from rev_pin edges (never degree-capped; see _kept_edges)
    pins = torch.bincount(g["net", "rev_pin", "cell"].edge_index[1], minlength=g["cell"].num_nodes).double()
    pin_sum = torch.zeros(g["gcell"].num_nodes, dtype=torch.float64).index_add_(0, gcell_idx, pins[cell_idx])
    return area_sum, pin_sum


def boundary_gcells(data, part):
    """Bool per gcell: 4-adjacent to a gcell of a different partition."""
    s, d = data["gcell", "adjacent", "gcell"].edge_index
    b = torch.zeros(data["gcell"].num_nodes, dtype=torch.bool)
    b[s[part[s] != part[d]]] = True
    return b


@torch.no_grad()
def cmd_verify(args):
    from metrics import spearman_corr
    data = torch.load(graph_path(args.cells), weights_only=False)
    ref = torch.load(f"data/processed/ref_{args.cells}.pt")
    model = make_model(data).eval()
    k = load_results(args.cells)["plan"]["k"]
    part = spatial_parts(data, k, k)
    boundary = boundary_gcells(data, part)
    full_area, full_pins = density_sums(data)
    tol = args.tol

    variants = {
        "halo_capped": dict(halo=True, degree_cap=args.degree_cap),
        "no_halo": dict(halo=False, degree_cap=args.degree_cap),
        "halo_uncapped": dict(halo=True, degree_cap=None),
    }
    report = {}
    for name, kw in variants.items():
        if name == "halo_uncapped" and args.skip_uncapped:
            continue
        stitched = torch.full_like(ref, float("nan"))
        stored_mismatch = dens_mismatch = 0
        sizes = []
        t0 = time.perf_counter()
        for p, sub in make_partitions(data, k, k, **kw):
            core = sub["gcell"].core
            gid = sub["gcell"].n_id[core]
            # 1) stored features copied exactly
            stored_mismatch += int((sub["gcell"].x[core] != data["gcell"].x[gid]).any(1).sum())
            # 2) density sums recomputed from the partition's own cells/pins
            a, pn = density_sums(sub)
            dens_mismatch += int(((a[core] != full_area[gid]) | (pn[core] != full_pins[gid])).sum())
            # 3) model outputs vs full-graph reference
            stitched[gid] = model(sub)[core]
            sizes.append({nt: sub[nt].num_nodes for nt in sub.node_types}
                         | {"edges": sum(sub[et].edge_index.shape[1] for et in sub.edge_types)})
        assert not torch.isnan(stitched).any(), "some gcell was not core in any partition"
        err = (stitched - ref).abs().max(1).values
        exact = err <= tol
        r = dict(
            time_s=time.perf_counter() - t0,
            stored_feature_mismatch_gcells=stored_mismatch,
            density_mismatch_gcells=dens_mismatch,
            max_abs_err=float(err.max()),
            frac_exact=float(exact.float().mean()),
            frac_bit_identical=float((err == 0).float().mean()),
            frac_exact_boundary=float(exact[boundary].float().mean()),
            frac_exact_interior=float(exact[~boundary].float().mean()),
            mean_abs_err_boundary=float(err[boundary].mean()),
            mean_abs_err_interior=float(err[~boundary].mean()),
            p99_abs_err=float(torch.quantile(err, 0.99)),
            ref_output_std=float(ref.std()),
            spearman_vs_ref=[float(spearman_corr(stitched[:, c], ref[:, c])) for c in range(2)],
            max_part_nodes=max(sum(v for kk, v in s.items() if kk != "edges") for s in sizes),
            max_part_edges=max(s["edges"] for s in sizes),
        )
        report[name] = r
        print(f"\n[{name}] {json.dumps(r, indent=1)}")
    report.update(tol=tol, n_boundary_gcells=int(boundary.sum()), n_gcells=int(len(boundary)))
    save_results(args.cells, dict(verify=report))


def _step_worker(path):
    """Training step (fwd + bwd, Huber on core gcells) on one saved partition."""
    sub = torch.load(path, weights_only=False)
    model = make_model(sub)
    base = memtrack.current_bytes()
    t0 = time.perf_counter()
    out = model(sub)
    core = sub["gcell"].core
    F.smooth_l1_loss(out[core], sub["gcell"].y_ratio[core]).backward()
    t = time.perf_counter() - t0
    peak = memtrack.peak_bytes()
    print("RESULT " + json.dumps(dict(time_s=t, peak_bytes=peak, baseline_bytes=base,
                                      nodes=sum(sub[nt].num_nodes for nt in sub.node_types),
                                      edges=sum(sub[et].edge_index.shape[1] for et in sub.edge_types),
                                      core_gcells=int(core.sum()))), flush=True)


def cmd_measure(args):
    from scalability import run_cmd
    budget = int(args.budget_gib * GIB)
    paths = sorted(glob.glob(os.path.join(part_dir(args.cells), "part_*.pt")))
    rows = []
    for path in paths:
        r = run_cmd([sys.executable, os.path.abspath(__file__), "_step", path], budget)
        r["part"] = os.path.basename(path)
        rows.append(r)
        if r["status"] == "ok":
            print(f"{r['part']}: nodes={r['nodes']:,} edges={r['edges']:,} core={r['core_gcells']:,} "
                  f"peak={r['peak_bytes'] / GIB:.3f} GiB time={r['time_s']:.2f}s")
        else:
            print(f"{r['part']}: {r['status']}")
    # repeat the largest partition to show the spread
    ok = [r for r in rows if r["status"] == "ok"]
    largest = max(ok, key=lambda r: r["edges"])
    reps = [largest["peak_bytes"]] + [
        run_cmd([sys.executable, os.path.abspath(__file__), "_step",
                 os.path.join(part_dir(args.cells), largest["part"])], budget).get("peak_bytes")
        for _ in range(2)]
    print(f"largest partition {largest['part']} repeated: {[round(p / GIB, 3) for p in reps]} GiB")
    save_results(args.cells, dict(measure=dict(budget_bytes=budget, parts=rows,
                                               largest_part=largest["part"], largest_repeats=reps)))


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "_step":
        _step_worker(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "reference", "verify", "measure"):
        p = sub.add_parser(name)
        p.add_argument("--cells", type=int, required=True)
        p.add_argument("--degree-cap", type=int, default=512)
        p.add_argument("--budget-gib", type=float, default=2.0 if name == "plan" else 5.5)
        p.add_argument("--tol", type=float, default=1e-6)
        p.add_argument("--skip-uncapped", action="store_true")
    args = parser.parse_args()
    dict(plan=cmd_plan, reference=cmd_reference, verify=cmd_verify, measure=cmd_measure)[args.cmd](args)


if __name__ == "__main__":
    main()
