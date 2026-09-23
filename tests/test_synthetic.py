"""Phase 9 checks for src/synthetic.py - runs in seconds, no data needed.

1. knn_tiled is exact: same neighbour distances as build_graph.knn_dense
   (the original O(n^2) implementation) on clustered, site-snapped
   placements (lots of exact distance ties) and on continuous uniform
   points (no ties -> identical neighbour sets).
2. Synthetic graphs carry exactly the 6 relations of the real pipeline,
   with the real feature widths (cell 5, net 3, gcell 4) and label shapes.
3. Structural invariants: every net has >=1 pin, every cell is in exactly
   one gcell, pin edges are symmetric, no duplicate (net, cell) pins.
4. Same seed -> identical graph; different seed -> different graph.
5. The existing CongestionGNN runs a forward pass on it.

Usage:
    python tests/test_synthetic.py
"""

import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from build_graph import knn_dense
from model import CongestionGNN
from synthetic import generate_graph, generate_placement, knn_tiled

EXPECTED_EDGE_TYPES = {
    ("cell", "pin", "net"), ("net", "rev_pin", "cell"),
    ("cell", "in", "gcell"), ("gcell", "contains", "cell"),
    ("cell", "near", "cell"), ("gcell", "adjacent", "gcell"),
}


def neighbour_dists(edges, x, y, n, k):
    src, dst = edges
    d = (x[src] - x[dst]) ** 2 + (y[src] - y[dst]) ** 2
    return np.sort(d.reshape(n, k), axis=1)


def check_knn():
    for n_cells, seed in [(3000, 0), (3000, 1), (6000, 2)]:
        p, _ = generate_placement(n_cells, seed)
        x, y = p["cell_x"].astype(np.float64), p["cell_y"].astype(np.float64)
        dense, tiled = knn_dense(x, y, 8), knn_tiled(x, y, 8)
        assert dense.shape == tiled.shape
        assert (tiled[0] == dense[0]).all()
        assert (tiled[0] != tiled[1]).all(), "self-loop in knn_tiled"
        diff = np.abs(neighbour_dists(dense, x, y, n_cells, 8) - neighbour_dists(tiled, x, y, n_cells, 8)).max()
        print(f"knn clustered n={n_cells} seed={seed}: max |dist_dense - dist_tiled| = {diff}")
        assert diff == 0

    rng = np.random.default_rng(0)
    for n, ppt in [(4000, 256), (4000, 16), (50, 256), (9, 4)]:
        x, y = rng.random(n) * 1e5, rng.random(n) * 3e4  # non-square extent
        dense, tiled = knn_dense(x, y, 8), knn_tiled(x, y, 8, points_per_tile=ppt)
        k = dense.shape[1] // n
        same_sets = (np.sort(dense[1].reshape(n, k), 1) == np.sort(tiled[1].reshape(n, k), 1)).all()
        print(f"knn uniform n={n} points_per_tile={ppt}: identical neighbour sets = {same_sets}")
        assert same_sets


def check_graph():
    data = generate_graph(20_000, seed=0)
    assert set(data.edge_types) == EXPECTED_EDGE_TYPES, data.edge_types
    assert data["cell"].x.shape[1] == 5 and data["net"].x.shape[1] == 3 and data["gcell"].x.shape[1] == 4
    n_gcell = data["gcell"].num_nodes
    assert data["gcell"].y_ratio.shape == (n_gcell, 2)
    assert int(data["gcell"].grid_shape.prod()) == n_gcell
    for store in [data["cell"], data["net"], data["gcell"]]:
        assert torch.isfinite(store.x).all()

    pin = data["cell", "pin", "net"].edge_index
    rev = data["net", "rev_pin", "cell"].edge_index
    assert torch.equal(pin.flip(0), rev)
    assert torch.bincount(pin[1], minlength=data["net"].num_nodes).min() >= 1
    key = pin[1] * data["cell"].num_nodes + pin[0]
    assert key.unique().numel() == key.numel(), "duplicate (net, cell) pins"
    cig = data["cell", "in", "gcell"].edge_index
    assert torch.equal(cig[0], torch.arange(data["cell"].num_nodes))
    assert torch.bincount(data["gcell", "adjacent", "gcell"].edge_index[0], minlength=n_gcell).min() >= 2
    print(f"graph invariants OK: {data['cell'].num_nodes} cells, {data['net'].num_nodes} nets, {n_gcell} gcells")
    return data


def check_determinism(data):
    again = generate_graph(20_000, seed=0)
    other = generate_graph(20_000, seed=1)
    for et in EXPECTED_EDGE_TYPES:
        assert torch.equal(data[et].edge_index, again[et].edge_index)
    for nt in data.node_types:
        assert torch.equal(data[nt].x, again[nt].x)
    assert not torch.equal(data["cell"].x, other["cell"].x)
    print("determinism OK: seed 0 twice -> identical, seed 1 -> different")


def check_model(data):
    model = CongestionGNN({nt: data[nt].x.shape[1] for nt in data.node_types}, list(data.edge_types))
    with torch.no_grad():
        out = model(data)
    assert out.shape == (data["gcell"].num_nodes, 2) and torch.isfinite(out).all()
    print(f"CongestionGNN forward OK: output {tuple(out.shape)}, "
          f"{sum(p.numel() for p in model.parameters())} params")


def main():
    check_knn()
    data = check_graph()
    check_determinism(data)
    check_model(data)
    print("\nPASS")


if __name__ == "__main__":
    main()
