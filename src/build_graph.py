"""Assemble a PyG HeteroData graph from extract_placement.py's and
extract_congestion.py's .npz outputs.

Runs in the normal project venv (plain numpy/torch/torch_geometric) - the
two .npz inputs are already plain arrays, no OpenROAD Python API needed
here. See notes/phase4-graph-construction.md for why this is split from
the odb-side extraction into two scripts/environments.

Node features are all technology-relative (multiples of site width/height,
routing tracks, dimensionless ratios) rather than raw DBU/nanometres, so
the graph transfers across technologies later without re-deriving units:
  - cell:  [width/site_w, height/site_h, pin_count, x/die_w, y/die_h]
  - net:   [fanout, hpwl/site_w, bbox_aspect_ratio]
  - gcell: [cell_density, pin_density, capacity_horizontal, capacity_vertical]
    (density = site-area occupied per gcell / gcell area in site-units;
    capacity is already in routing-track units, per extract_congestion.py)

Edge types:
  - ('cell', 'pin', 'net') and ('net', 'rev_pin', 'cell'): pin connectivity
  - ('cell', 'in', 'gcell') and ('gcell', 'contains', 'cell'): spatial containment
  - ('cell', 'near', 'cell'): k=8 nearest neighbor by center distance
  - ('gcell', 'adjacent', 'gcell'): 4-connected grid adjacency

The post-route gcell grid's demand/ratio (ground truth congestion) is
attached to gcell nodes as `y` for later supervised training (Phase 7),
not used as an input feature.

Usage:
    python build_graph.py <placement.npz> <congestion.npz> <output.pt>
"""

import sys

import numpy as np
import torch
from torch_geometric.data import HeteroData


def gcell_index(x, y, origin_x, origin_y, pitch_x, pitch_y, nx, ny):
    xi = np.clip(((x - origin_x) / pitch_x).astype(np.int64), 0, nx - 1)
    yi = np.clip(((y - origin_y) / pitch_y).astype(np.int64), 0, ny - 1)
    return xi, yi


def build_graph(placement, congestion):
    site_w, site_h = float(placement["site_w"]), float(placement["site_h"])
    die_w, die_h = float(placement["die_w"]), float(placement["die_h"])

    origin_x, origin_y = int(congestion["origin_x"]), int(congestion["origin_y"])
    pitch_x, pitch_y = int(congestion["pitch_x"]), int(congestion["pitch_y"])
    nx, ny = int(congestion["nx"]), int(congestion["ny"])

    # ---- cell nodes ----
    cell_x = placement["cell_x"].astype(np.float64)
    cell_y = placement["cell_y"].astype(np.float64)
    cell_w = placement["cell_w"].astype(np.float64)
    cell_h = placement["cell_h"].astype(np.float64)
    cell_pin_count = placement["cell_pin_count"].astype(np.float64)
    n_cells = len(cell_x)

    cell_feat = np.stack([
        cell_w / site_w,
        cell_h / site_h,
        cell_pin_count,
        cell_x / die_w,
        cell_y / die_h,
    ], axis=1)

    # ---- net nodes ----
    net_fanout = placement["net_fanout"].astype(np.float64)
    net_hpwl = placement["net_hpwl"].astype(np.float64)
    net_bbox_w = placement["net_bbox_w"].astype(np.float64)
    net_bbox_h = placement["net_bbox_h"].astype(np.float64)
    n_nets = len(net_fanout)

    eps = 1e-6
    aspect_ratio = (net_bbox_w + eps) / (net_bbox_h + eps)
    net_feat = np.stack([
        net_fanout,
        net_hpwl / site_w,
        aspect_ratio,
    ], axis=1)

    # ---- gcell nodes ----
    gcell_area_sites = (pitch_x / site_w) * (pitch_y / site_h)

    cell_xi, cell_yi = gcell_index(cell_x, cell_y, origin_x, origin_y, pitch_x, pitch_y, nx, ny)
    cell_gcell_flat = cell_yi * nx + cell_xi

    cell_area_sites = (cell_w / site_w) * (cell_h / site_h)
    cell_density = np.bincount(cell_gcell_flat, weights=cell_area_sites, minlength=nx * ny) / gcell_area_sites

    edge_cell_idx = placement["edge_cell_idx"].astype(np.int64)
    pin_xi = cell_xi[edge_cell_idx]
    pin_yi = cell_yi[edge_cell_idx]
    pin_gcell_flat = pin_yi * nx + pin_xi
    pin_density = np.bincount(pin_gcell_flat, minlength=nx * ny).astype(np.float64) / gcell_area_sites

    capacity = congestion["capacity"]  # (ny, nx, 2): [...,0]=horizontal, [...,1]=vertical
    cap_h = capacity[:, :, 0].reshape(-1)
    cap_v = capacity[:, :, 1].reshape(-1)

    gcell_feat = np.stack([cell_density, pin_density, cap_h, cap_v], axis=1)

    demand = congestion["demand"].reshape(nx * ny, 2)
    ratio = congestion["ratio"].reshape(nx * ny, 2)

    # ---- edges: cell <-> net (pin connectivity) ----
    edge_net_idx = placement["edge_net_idx"].astype(np.int64)
    cell_pin_net = np.stack([edge_cell_idx, edge_net_idx])
    net_rev_pin_cell = np.stack([edge_net_idx, edge_cell_idx])

    # ---- edges: cell <-> gcell (spatial containment) ----
    cell_in_gcell = np.stack([np.arange(n_cells), cell_gcell_flat])
    gcell_contains_cell = np.stack([cell_gcell_flat, np.arange(n_cells)])

    # ---- edges: cell <-> cell (k=8 nearest neighbor, geometric) ----
    k = 8
    diff_x = cell_x[:, None] - cell_x[None, :]
    diff_y = cell_y[:, None] - cell_y[None, :]
    dist_sq = diff_x ** 2 + diff_y ** 2
    np.fill_diagonal(dist_sq, np.inf)
    k_eff = min(k, n_cells - 1)
    nn_idx = np.argpartition(dist_sq, k_eff, axis=1)[:, :k_eff]
    src = np.repeat(np.arange(n_cells), k_eff)
    dst = nn_idx.reshape(-1)
    cell_near_cell = np.stack([src, dst])

    # ---- edges: gcell <-> gcell (4-connected) ----
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    gx, gy = gx.reshape(-1), gy.reshape(-1)
    flat = gy * nx + gx
    adj_src, adj_dst = [], []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx_, ny_ = gx + dx, gy + dy
        valid = (nx_ >= 0) & (nx_ < nx) & (ny_ >= 0) & (ny_ < ny)
        adj_src.append(flat[valid])
        adj_dst.append((ny_[valid] * nx + nx_[valid]))
    gcell_adjacent_gcell = np.stack([np.concatenate(adj_src), np.concatenate(adj_dst)])

    data = HeteroData()
    data["cell"].x = torch.tensor(cell_feat, dtype=torch.float32)
    data["net"].x = torch.tensor(net_feat, dtype=torch.float32)
    data["gcell"].x = torch.tensor(gcell_feat, dtype=torch.float32)
    data["gcell"].y_demand = torch.tensor(demand, dtype=torch.float32)
    data["gcell"].y_ratio = torch.tensor(ratio, dtype=torch.float32)
    data["gcell"].grid_shape = torch.tensor([ny, nx], dtype=torch.int64)

    data["cell", "pin", "net"].edge_index = torch.tensor(cell_pin_net, dtype=torch.long)
    data["net", "rev_pin", "cell"].edge_index = torch.tensor(net_rev_pin_cell, dtype=torch.long)
    data["cell", "in", "gcell"].edge_index = torch.tensor(cell_in_gcell, dtype=torch.long)
    data["gcell", "contains", "cell"].edge_index = torch.tensor(gcell_contains_cell, dtype=torch.long)
    data["cell", "near", "cell"].edge_index = torch.tensor(cell_near_cell, dtype=torch.long)
    data["gcell", "adjacent", "gcell"].edge_index = torch.tensor(gcell_adjacent_gcell, dtype=torch.long)

    return data


def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <placement.npz> <congestion.npz> <output.pt>")
        sys.exit(1)

    placement_path, congestion_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    placement = np.load(placement_path, allow_pickle=True)
    congestion = np.load(congestion_path, allow_pickle=True)

    data = build_graph(placement, congestion)
    torch.save(data, out_path)

    print(data)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
