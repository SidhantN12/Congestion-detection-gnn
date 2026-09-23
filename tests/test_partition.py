"""Phase 11 checks for src/partition.py on graphs small enough for a
full-graph model(data) forward pass (the ground truth here is the real
CongestionGNN forward, not a reimplementation).

1. reference_forward (chunked, used as ground truth at 1.5M cells where
   model(data) doesn't fit) matches model(data).
2. Every gcell is core in exactly one partition; partitions are
   rectangles of the gcell grid.
3. With a full 3-hop halo and no degree cap, stitched partition outputs
   equal model(data) on every gcell.
4. Without a halo, boundary gcells come out wrong (the halo is doing work).
5. Degree capping: capped nets keep <= cap pins, uncapped nets keep all
   their pins; with the cap, only gcells within reach of a capped net
   deviate.
6. Density sums (cell site-area, pin count) per core gcell recomputed from
   the partition equal the full graph's.

Usage:
    python tests/test_partition.py
"""

import sys

import torch

sys.path.insert(0, "src")
from partition import (PIN, boundary_gcells, capped_nets, density_sums, expand_halo,
                       make_model, make_partitions, reference_forward, spatial_parts)
from synthetic import generate_graph

TOL = 1e-5


def stitched_outputs(model, data, k, **kw):
    out = torch.full((data["gcell"].num_nodes, 2), float("nan"))
    count = torch.zeros(data["gcell"].num_nodes, dtype=torch.long)
    subs = []
    for _, sub in make_partitions(data, k, k, **kw):
        core = sub["gcell"].core
        gid = sub["gcell"].n_id[core]
        out[gid] = model(sub)[core]
        count[gid] += 1
        subs.append(sub)
    return out, count, subs


def downstream(data, seeds, hops):
    """Node masks reachable from `seeds` by following edges forward (src -> dst)."""
    masks = {nt: torch.zeros(data[nt].num_nodes, dtype=torch.bool) for nt in data.node_types}
    for nt, m in seeds.items():
        masks[nt] |= m
    for _ in range(hops):
        new = {nt: m.clone() for nt, m in masks.items()}
        for et in data.edge_types:
            src, dst = data[et].edge_index
            new[et[2]][dst[masks[et[0]][src]]] = True
        masks = new
    return masks


@torch.no_grad()
def main():
    for n in (5_000, 20_000):
        data = generate_graph(n, seed=3)
        model = make_model(data).eval()
        full = model(data)
        ref = reference_forward(model, data, edge_chunk=10_000)
        diff = (full - ref).abs().max().item()
        print(f"[1] n={n}: max |model(data) - reference_forward| = {diff:.2e}")
        assert diff < TOL

    k = 3
    part = spatial_parts(data, k, k)
    ny, nx = (int(v) for v in data["gcell"].grid_shape)
    grid = part.reshape(ny, nx)
    for p in range(int(part.max()) + 1):
        ys, xs = torch.nonzero(grid == p, as_tuple=True)
        area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        assert area == len(ys), f"partition {p} is not a rectangle"
    cells_per_part = torch.bincount(part[data["cell", "in", "gcell"].edge_index[1]])
    print(f"[2] {k}x{k}: {len(cells_per_part)} rectangular partitions, cells per partition "
          f"{cells_per_part.min().item():,}-{cells_per_part.max().item():,}")

    out, count, subs = stitched_outputs(model, data, k, degree_cap=None)
    assert (count == 1).all(), "every gcell must be core in exactly one partition"
    diff = (out - full).abs().max().item()
    print(f"[3] halo, uncapped: max |stitched - model(data)| = {diff:.2e} over all "
          f"{data['gcell'].num_nodes:,} gcells")
    assert diff < TOL

    boundary = boundary_gcells(data, part)
    out, count, _ = stitched_outputs(model, data, k, halo=False, degree_cap=None)
    err = (out - full).abs().max(1).values
    wrong_b = (err[boundary] > TOL).float().mean().item()
    wrong_i = (err[~boundary] > TOL).float().mean().item()
    print(f"[4] no halo: {100 * wrong_b:.1f}% of boundary gcells and {100 * wrong_i:.1f}% of interior "
          f"gcells deviate (max err {err.max().item():.3f})")
    assert wrong_b > 0.5

    cap = 64
    capped = capped_nets(data, cap)
    fanout = torch.bincount(data[PIN].edge_index[1], minlength=data["net"].num_nodes)
    out, count, subs = stitched_outputs(model, data, k, degree_cap=cap)
    for sub in subs:
        deg = torch.bincount(sub[PIN].edge_index[1], minlength=sub["net"].num_nodes)
        gid = sub["net"].n_id
        assert (deg[capped[gid]] <= cap).all()
        # uncapped nets reached within 2 hops keep every pin; a net that only
        # appears at hop 3 is a leaf (layer-0 features only) and may keep fewer
        in_reach = expand_halo(sub, torch.nonzero(sub["gcell"].core).squeeze(1), 2)["net"]
        full_deg = fanout[gid]
        assert (deg[~capped[gid] & in_reach] == full_deg[~capped[gid] & in_reach]).all()
    err = (out - full).abs().max(1).values
    # A capped net's aggregated (layer >= 1) embedding is what the cap perturbs;
    # it reaches gcells <= 2 hops downstream before the 3rd layer's output.
    affected = downstream(data, {"net": capped}, hops=2)["gcell"]
    assert (err[~affected] <= TOL).all(), "a gcell out of reach of every capped net deviated"
    print(f"[5] cap={cap}: {int(capped.sum())} nets capped; all {int((~affected).sum()):,} gcells out of "
          f"their 2-hop reach exact; {int(affected.sum()):,} in reach: max err {err[affected].max().item():.2e} "
          f"(output std {full.std().item():.2e})")

    full_area, full_pins = density_sums(data)
    for sub in subs:
        core = sub["gcell"].core
        a, pn = density_sums(sub)
        gid = sub["gcell"].n_id[core]
        assert torch.equal(a[core], full_area[gid]) and torch.equal(pn[core], full_pins[gid])
        assert torch.equal(sub["gcell"].x[core], data["gcell"].x[gid])
    print("[6] density sums and stored gcell features identical for every core gcell")
    print("\nPASS")


if __name__ == "__main__":
    main()
