"""Phase 4 gate checks: node/edge counts, degree distributions per edge
type, no isolated gcell nodes, every net has >=1 pin edge.
"""

import sys

import torch


def degree_stats(edge_index, num_src, num_dst, label):
    src, dst = edge_index[0], edge_index[1]
    out_deg = torch.bincount(src, minlength=num_src)
    in_deg = torch.bincount(dst, minlength=num_dst)
    print(f"{label}: {edge_index.shape[1]} edges")
    print(f"  out-degree (src side): min={out_deg.min().item()} max={out_deg.max().item()} "
          f"mean={out_deg.float().mean().item():.3f}")
    print(f"  in-degree  (dst side): min={in_deg.min().item()} max={in_deg.max().item()} "
          f"mean={in_deg.float().mean().item():.3f}")
    return out_deg, in_deg


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/gcd_baseline_graph.pt"
    data = torch.load(path, weights_only=False)

    n_cells = data["cell"].x.shape[0]
    n_nets = data["net"].x.shape[0]
    n_gcells = data["gcell"].x.shape[0]

    print("=== Node counts ===")
    print(f"cell:  {n_cells}")
    print(f"net:   {n_nets}")
    print(f"gcell: {n_gcells}")
    print()

    print("=== Edge counts and degree distributions ===")
    _, net_in_deg = degree_stats(data["cell", "pin", "net"].edge_index, n_cells, n_nets, "cell -pin-> net")
    cell_in_deg_from_net, _ = degree_stats(data["net", "rev_pin", "cell"].edge_index, n_nets, n_cells, "net -rev_pin-> cell")
    degree_stats(data["cell", "in", "gcell"].edge_index, n_cells, n_gcells, "cell -in-> gcell")
    _, cell_in_deg_from_gcell = degree_stats(data["gcell", "contains", "cell"].edge_index, n_gcells, n_cells, "gcell -contains-> cell")
    degree_stats(data["cell", "near", "cell"].edge_index, n_cells, n_cells, "cell -near-> cell")
    _, gcell_deg = degree_stats(data["gcell", "adjacent", "gcell"].edge_index, n_gcells, n_gcells, "gcell -adjacent-> gcell")
    print()

    print("=== Gate checks ===")
    isolated_gcells = (gcell_deg == 0).sum().item()
    print(f"Isolated gcell nodes (0 gcell-adjacent-gcell edges): {isolated_gcells}")
    assert isolated_gcells == 0, "FAIL: isolated gcell nodes found"

    gcell_contains_deg = torch.bincount(data["gcell", "contains", "cell"].edge_index[0], minlength=n_gcells)
    empty_gcells = (gcell_contains_deg == 0).sum().item()
    print(f"GCells with zero cells assigned: {empty_gcells}/{n_gcells} "
          f"(not a failure by itself - small/sparse regions are legitimate)")

    nets_with_no_pin_edge = (net_in_deg == 0).sum().item()
    print(f"Nets with zero pin edges: {nets_with_no_pin_edge}")
    assert nets_with_no_pin_edge == 0, "FAIL: nets with no pin edges found"

    print()
    print("PASS: no isolated gcell nodes, every net has >=1 pin edge")


if __name__ == "__main__":
    main()
