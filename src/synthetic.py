"""Synthetic CircuitNet-scale placed netlists -> the same HeteroData graphs
the real pipeline produces, at arbitrary size, with no OpenROAD and no
benchmark data.

The generator does NOT build graphs itself. It emits `placement` and
`congestion` dicts in exactly the schema extract_placement.py /
extract_congestion.py write to .npz, then hands them to the real
build_graph.build_graph - so node features and the 6 relation types are
produced by the same code path as the gcd graphs, not by a lookalike.
The one substitution is the cell-near-cell kNN: build_graph's default
(knn_dense) materialises an n x n distance matrix, which is ~30 GB at
35K cells; knn_tiled below is an exact drop-in replacement.

What is synthesised (every knob is a keyword argument of
generate_placement; defaults are documented in
notes/phase9-synthetic-generator.md along with where each came from):
  - Cells: nangate45-like widths in sites (one row high), a fraction of
    them flip-flops (wider, and the sinks of the clock net).
  - Die: square, sized from total cell area / target utilization.
  - Placement: spatially clustered - a Gaussian mixture of "module"
    clusters (lognormal sizes) plus a uniform background fraction,
    snapped to the site/row grid. Not legalized (cells can overlap).
  - Nets: count = net_ratio * cells. Degree distribution is heavy-tailed:
    a body concentrated on 2-6 pins, a truncated power-law tail, a
    handful of hub nets with thousands of pins, and one pre-CTS clock net
    spanning every flop. Body/tail nets connect spatially local cells
    (Laplace offsets along a Morton/Z-order curve); hubs and the clock
    connect cells uniformly across the die.
  - GCell grid: fixed pitch (gcd's 4200 DBU), so gcell count grows with
    die area as it does in a real router.
  - Labels: a RUDY-style proxy (each net's bbox wirelength spread over
    the gcells its bbox covers), with capacity set so a fixed fraction
    of gcells overflow. This is NOT router ground truth - it exists so
    the graph has non-degenerate `y` tensors for pipeline/scale testing.

Usage:
    python src/synthetic.py --cells 35000 [--seed 0] [--out graph.pt] [--stats-json stats.json]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memtrack
from build_graph import build_graph

# nangate45 units, as used by the gcd pipeline (Phases 1-6).
SITE_W = 380    # DBU (0.19 um)
SITE_H = 2800   # DBU (1.4 um), one placement row
GCELL_PITCH = 4200  # DBU - gcd's GRT grid pitch (17x17 grid, Phase 5)

# Cell widths in sites. Logic cells: small combinational gates; flops are
# a fixed wider cell. Assumed nangate45-like values, not measured from a
# real library file (none available here).
LOGIC_WIDTHS = np.array([2, 3, 4, 5, 6, 8])
LOGIC_WIDTH_P = np.array([0.25, 0.30, 0.20, 0.10, 0.08, 0.07])
FLOP_WIDTH = 17

# Net-degree body: P(pins = 2..6). Mode at 3, with 3-4 pins the majority
# of body nets, per the project spec. Remaining nets come from the tail.
BODY_DEGREES = np.array([2, 3, 4, 5, 6])
BODY_P = np.array([0.25, 0.33, 0.20, 0.08, 0.04])  # sums to 0.90; 0.10 = tail


def morton_order(x, y, bits=16):
    """Permutation sorting points along a Z-order curve (spatial locality)."""
    def spread(v):
        v = v.astype(np.uint64)
        v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF)
        v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F)
        v = (v | (v << np.uint64(2))) & np.uint64(0x33333333)
        v = (v | (v << np.uint64(1))) & np.uint64(0x55555555)
        return v
    scale = (1 << bits) - 1
    qx = ((x - x.min()) / max(np.ptp(x), 1e-9) * scale).astype(np.uint64)
    qy = ((y - y.min()) / max(np.ptp(y), 1e-9) * scale).astype(np.uint64)
    code = spread(qx) | (spread(qy) << np.uint64(1))
    return np.argsort(code, kind="stable")


def sample_net_degrees(rng, n_nets, n_cells, n_flops, tail_alpha, tail_max,
                       hub_count, hub_min):
    """Returns (degrees, kind) with kind 0=body/tail (local), 1=hub, 2=clock."""
    n_body = n_nets - hub_count - 1
    tail_frac = 1.0 - BODY_P.sum()

    is_tail = rng.random(n_body) < tail_frac
    deg = rng.choice(BODY_DEGREES, size=n_body, p=BODY_P / BODY_P.sum())
    # Truncated discrete power law on [BODY_DEGREES.max()+1, tail_max],
    # P(d) ~ d^-alpha, via inverse-CDF sampling over the explicit support.
    support = np.arange(BODY_DEGREES.max() + 1, tail_max + 1)
    cdf = np.cumsum(support.astype(np.float64) ** -tail_alpha)
    cdf /= cdf[-1]
    deg[is_tail] = support[np.searchsorted(cdf, rng.random(is_tail.sum()))]

    hub_max = max(2 * hub_min, n_cells // 100)
    hub_deg = np.exp(rng.uniform(np.log(hub_min), np.log(hub_max), hub_count))
    hub_deg = np.minimum(hub_deg.round().astype(np.int64), n_cells)

    degrees = np.concatenate([deg, hub_deg, [n_flops]]).astype(np.int64)
    kind = np.concatenate([np.zeros(n_body, np.int8), np.ones(hub_count, np.int8), [2]]).astype(np.int8)
    return degrees, kind


def generate_placement(n_cells, seed=0, *, utilization=0.70, flop_frac=0.10,
                       net_ratio=631 / 606, cluster_size=4000,
                       cluster_peak_util=1.0, background_frac=0.20,
                       tail_alpha=2.5, tail_max=500, hub_every=25_000,
                       hub_min=1000, locality=2.0, overflow_frac=0.10):
    """Synthesise one placed design. Returns (placement, congestion) dicts
    in the extract_placement.py / extract_congestion.py .npz schema."""
    rng = np.random.default_rng(seed)
    n = int(n_cells)

    # ---- cells ----
    is_flop = rng.random(n) < flop_frac
    w_sites = np.where(is_flop, FLOP_WIDTH, rng.choice(LOGIC_WIDTHS, size=n, p=LOGIC_WIDTH_P))
    cell_w = (w_sites * SITE_W).astype(np.int64)
    cell_h = np.full(n, SITE_H, dtype=np.int64)

    # ---- die: square, total cell area / utilization, whole rows/sites ----
    die_area = w_sites.sum() * SITE_W * SITE_H / utilization
    side = np.sqrt(die_area)
    die_h = int(np.ceil(side / SITE_H)) * SITE_H
    die_w = int(np.ceil(side / SITE_W)) * SITE_W

    # ---- placement: Gaussian-mixture clusters + uniform background ----
    n_clusters = max(1, round(n / cluster_size))
    weights = rng.lognormal(0.0, 1.0, n_clusters)
    weights /= weights.sum()
    in_background = rng.random(n) < background_frac
    cluster_of = rng.choice(n_clusters, size=n, p=weights)
    cluster_sizes = np.bincount(cluster_of[~in_background], minlength=n_clusters)
    # sigma such that each cluster's peak density (m / 2*pi*sigma^2 cells
    # per unit area, times mean cell area) is cluster_peak_util on its own.
    mean_cell_area = die_area * utilization / n
    sigma = np.sqrt(np.maximum(cluster_sizes, 1) * mean_cell_area / (2 * np.pi * cluster_peak_util))
    cx = rng.uniform(0.1 * die_w, 0.9 * die_w, n_clusters)
    cy = rng.uniform(0.1 * die_h, 0.9 * die_h, n_clusters)

    x = cx[cluster_of] + sigma[cluster_of] * rng.standard_normal(n)
    y = cy[cluster_of] + sigma[cluster_of] * rng.standard_normal(n)
    x[in_background] = rng.uniform(0, die_w, in_background.sum())
    y[in_background] = rng.uniform(0, die_h, in_background.sum())
    # Reflect Gaussian tails that fall outside the die back inside
    # (clipping would pile them onto the boundary gcells), then snap lower-
    # left corners to the site grid / row grid.
    x = np.clip(np.abs(die_w - np.abs(die_w - np.abs(x))), 0, die_w - cell_w)
    y = np.clip(np.abs(die_h - np.abs(die_h - np.abs(y))), 0, die_h - SITE_H)
    cell_x = (np.floor(x / SITE_W) * SITE_W).astype(np.int64)
    cell_y = (np.floor(y / SITE_H) * SITE_H).astype(np.int64)

    # ---- nets: heavy-tailed degrees ----
    n_nets = max(round(net_ratio * n), 2)
    hub_count = max(4, round(n / hub_every))
    degrees, kind = sample_net_degrees(rng, n_nets, n, int(is_flop.sum()),
                                       tail_alpha, tail_max, hub_count, hub_min)

    # Local nets: driver at a random position along the Z-order curve,
    # sinks at Laplace-distributed offsets whose scale grows with degree.
    order = morton_order(cell_x + cell_w / 2, cell_y + SITE_H / 2)
    local = np.flatnonzero(kind == 0)
    local_deg = degrees[local]
    anchor = rng.integers(0, n, size=len(local))
    net_rep = np.repeat(local, local_deg)
    anchor_rep = np.repeat(anchor, local_deg)
    starts = np.cumsum(local_deg) - local_deg
    is_driver = np.zeros(len(net_rep), dtype=bool)
    is_driver[starts] = True
    offset = np.rint(rng.laplace(0.0, np.repeat(locality * local_deg, local_deg))).astype(np.int64)
    offset[is_driver] = 0
    local_cells = order[np.clip(anchor_rep + offset, 0, n - 1)]

    # Hub nets: uniformly random cells across the die. Clock: every flop.
    hub_nets, hub_cells = [], []
    for net in np.flatnonzero(kind == 1):
        hub_cells.append(rng.choice(n, size=degrees[net], replace=False))
        hub_nets.append(np.full(degrees[net], net))
    flops = np.flatnonzero(is_flop)
    clock_net = np.flatnonzero(kind == 2)[0]

    edge_net = np.concatenate([net_rep, *hub_nets, np.full(len(flops), clock_net)])
    edge_cell = np.concatenate([local_cells, *hub_cells, flops])
    del net_rep, anchor_rep, offset, local_cells, hub_nets, hub_cells

    # Shuffle net ids (the real net order carries no meaning), then drop
    # duplicate (net, cell) pins - local offsets can land on one cell twice.
    perm = rng.permutation(n_nets)
    edge_net = perm[edge_net]
    key = np.unique(edge_net * n + edge_cell)  # sorted by net, then cell
    edge_net_idx = key // n
    edge_cell_idx = key % n
    del key, edge_net, edge_cell

    # Every net keeps its driver, so every net id 0..n_nets-1 appears.
    net_fanout = np.bincount(edge_net_idx, minlength=n_nets)
    assert (net_fanout > 0).all()
    net_start = np.concatenate([[0], np.cumsum(net_fanout)[:-1]])

    pin_x = cell_x[edge_cell_idx] + cell_w[edge_cell_idx] // 2
    pin_y = cell_y[edge_cell_idx] + SITE_H // 2
    xmin = np.minimum.reduceat(pin_x, net_start)
    xmax = np.maximum.reduceat(pin_x, net_start)
    ymin = np.minimum.reduceat(pin_y, net_start)
    ymax = np.maximum.reduceat(pin_y, net_start)
    net_bbox_w = xmax - xmin
    net_bbox_h = ymax - ymin

    placement = dict(
        cell_x=cell_x, cell_y=cell_y, cell_w=cell_w, cell_h=cell_h,
        cell_pin_count=np.bincount(edge_cell_idx, minlength=n),
        net_fanout=net_fanout, net_hpwl=net_bbox_w + net_bbox_h,
        net_bbox_w=net_bbox_w, net_bbox_h=net_bbox_h,
        edge_cell_idx=edge_cell_idx, edge_net_idx=edge_net_idx,
        die_w=die_w, die_h=die_h, site_w=SITE_W, site_h=SITE_H,
    )

    # ---- gcell grid + RUDY-style proxy labels ----
    nx = int(np.ceil(die_w / GCELL_PITCH))
    ny = int(np.ceil(die_h / GCELL_PITCH))
    gx0 = np.clip(xmin // GCELL_PITCH, 0, nx - 1)
    gx1 = np.clip(xmax // GCELL_PITCH, 0, nx - 1)
    gy0 = np.clip(ymin // GCELL_PITCH, 0, ny - 1)
    gy1 = np.clip(ymax // GCELL_PITCH, 0, ny - 1)
    span_x = (gx1 - gx0 + 1).astype(np.float64)
    span_y = (gy1 - gy0 + 1).astype(np.float64)

    def spread_over_bbox(value):
        # 2D difference array over each net's gcell bbox, then prefix sums.
        diff = np.zeros((ny + 1) * (nx + 1))
        for yy, xx, sign in ((gy0, gx0, 1), (gy0, gx1 + 1, -1), (gy1 + 1, gx0, -1), (gy1 + 1, gx1 + 1, 1)):
            diff += np.bincount(yy * (nx + 1) + xx, weights=sign * value, minlength=diff.size)
        return diff.reshape(ny + 1, nx + 1).cumsum(0).cumsum(1)[:ny, :nx]

    # A net crossing span_x gcells horizontally uses ~1 track in each of
    # them, spread evenly over the span_y rows of its bbox (and vice versa).
    demand = np.stack([spread_over_bbox(1.0 / span_y), spread_over_bbox(1.0 / span_x)], axis=-1)
    cap = np.quantile(demand.reshape(-1, 2), 1.0 - overflow_frac, axis=0)
    capacity = np.broadcast_to(cap, demand.shape).copy()
    ratio = demand / capacity

    congestion = dict(
        demand=demand, capacity=capacity, ratio=ratio,
        origin_x=0, origin_y=0, pitch_x=GCELL_PITCH, pitch_y=GCELL_PITCH, nx=nx, ny=ny,
    )
    return placement, congestion


def knn_tiled(x, y, k, points_per_tile=256, query_chunk=1024):
    """Exact k-nearest-neighbour edges, same output as build_graph.knn_dense.

    Buckets points into a t x t tile grid. For each tile, candidate
    neighbours are the points in a (2r+1)^2 window of tiles around it
    (r=1 first). Any point outside the window is at least
    r * min(tile_w, tile_h) away from every query in the centre tile, so
    a query whose k-th candidate distance is within that bound has its
    exact k nearest neighbours; queries that fail the bound are retried
    with r+1. Memory is O(n*k) plus one (chunk x window) distance block.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return np.empty((2, 0), dtype=np.int64)

    t = max(1, int(np.sqrt(n / points_per_tile)))
    x0, y0 = x.min(), y.min()
    tile_w = max(np.ptp(x), 1e-9) / t
    tile_h = max(np.ptp(y), 1e-9) / t
    tx = np.minimum(((x - x0) / tile_w).astype(np.int64), t - 1)
    ty = np.minimum(((y - y0) / tile_h).astype(np.int64), t - 1)
    tile = ty * t + tx
    order = np.argsort(tile, kind="stable")
    bounds = np.searchsorted(tile[order], np.arange(t * t + 1))
    min_side = min(tile_w, tile_h)

    def window(cx, cy, r):
        rows = range(max(cy - r, 0), min(cy + r, t - 1) + 1)
        x_lo, x_hi = max(cx - r, 0), min(cx + r, t - 1)
        return np.concatenate([order[bounds[ry * t + x_lo]:bounds[ry * t + x_hi + 1]] for ry in rows])

    nbr = np.empty((n, k_eff), dtype=np.int64)
    for tid in np.flatnonzero(np.diff(bounds)):
        cy, cx = divmod(int(tid), t)
        pending = order[bounds[tid]:bounds[tid + 1]]
        r = 1
        while len(pending):
            cand = window(cx, cy, r)
            covers_all = r >= t
            if len(cand) - 1 < k_eff and not covers_all:
                r += 1
                continue
            failed = []
            for s in range(0, len(pending), query_chunk):
                q = pending[s:s + query_chunk]
                d = (x[q, None] - x[cand]) ** 2 + (y[q, None] - y[cand]) ** 2
                d[q[:, None] == cand[None, :]] = np.inf
                part = np.argpartition(d, k_eff - 1, axis=1)[:, :k_eff]
                kth = np.take_along_axis(d, part, axis=1).max(axis=1)
                ok = covers_all | (kth <= (r * min_side) ** 2)
                nbr[q[ok]] = cand[part[ok]]
                failed.append(q[~ok])
            pending = np.concatenate(failed)
            r += 1

    src = np.repeat(np.arange(n), k_eff)
    return np.stack([src, nbr.reshape(-1)])


def generate_graph(n_cells, seed=0, **kwargs):
    placement, congestion = generate_placement(n_cells, seed, **kwargs)
    return build_graph(placement, congestion, knn_fn=knn_tiled)


def graph_nbytes(data):
    return sum(v.numel() * v.element_size() for store in data.stores
               for v in store.values() if torch.is_tensor(v))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="save the HeteroData graph here (torch.save)")
    parser.add_argument("--stats-json", help="write construction stats here")
    args = parser.parse_args()

    base = memtrack.current_bytes()
    t0 = time.perf_counter()
    placement, congestion = generate_placement(args.cells, args.seed)
    t1 = time.perf_counter()
    data = build_graph(placement, congestion, knn_fn=knn_tiled)
    t2 = time.perf_counter()
    peak = memtrack.peak_bytes()

    fanout = placement["net_fanout"]
    degrees, counts = np.unique(fanout, return_counts=True)
    per_gcell = np.bincount(data["cell", "in", "gcell"].edge_index[1].numpy(),
                            minlength=data["gcell"].num_nodes)
    stats = dict(
        target_cells=args.cells,
        seed=args.seed,
        nodes={nt: int(data[nt].num_nodes) for nt in data.node_types},
        edges={"-".join(et): int(data[et].edge_index.shape[1]) for et in data.edge_types},
        grid=[int(congestion["ny"]), int(congestion["nx"])],
        net_degree=dict(
            max=int(fanout.max()), mean=float(fanout.mean()), median=float(np.median(fanout)),
            frac_eq_2=float((fanout == 2).mean()), frac_3_4=float(((fanout >= 3) & (fanout <= 4)).mean()),
            frac_ge_1000=float((fanout >= 1000).mean()), n_ge_1000=int((fanout >= 1000).sum()),
            frac_eq_1=float((fanout == 1).mean()),
            hist=[[int(d), int(c)] for d, c in zip(degrees, counts)],
        ),
        cells_per_gcell=dict(mean=float(per_gcell.mean()), var=float(per_gcell.var()),
                             dispersion_index=float(per_gcell.var() / per_gcell.mean()),
                             frac_empty=float((per_gcell == 0).mean())),
        cells_with_no_pins=int((placement["cell_pin_count"] == 0).sum()),
        gcell_cell_density_max=float(data["gcell"].x[:, 0].max()),
        label_ratio=dict(mean=float(congestion["ratio"].mean()), max=float(congestion["ratio"].max())),
        time_s=dict(placement=t1 - t0, build_graph=t2 - t1, total=t2 - t0),
        memory=dict(method=memtrack.method(), baseline_bytes=int(base), peak_bytes=int(peak),
                    peak_minus_baseline_bytes=int(peak - base), graph_tensor_bytes=int(graph_nbytes(data))),
    )

    print(data)
    print(json.dumps({k: v for k, v in stats.items() if k != "net_degree"}, indent=2))
    print("net_degree:", {k: v for k, v in stats["net_degree"].items() if k != "hist"})

    if args.stats_json:
        with open(args.stats_json, "w") as f:
            json.dump(stats, f)
    if args.out:
        torch.save(data, args.out)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
