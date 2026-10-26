"""Turn the text dumps of extract_congestion.tcl / extract_placement.tcl into
the exact .npz files extract_congestion.py / extract_placement.py write.

The arithmetic here is copied from those two scripts (direction
aggregation in tech-layer order, ratio, uniform-pitch check; HPWL, bbox
and fanout from iterm bbox centres), so build_graph.py sees the same
schema either way. Runs in the project venv.

Usage:
    python src/odb_text_to_npz.py congestion <in.txt> <out.npz>
    python src/odb_text_to_npz.py placement  <in.txt> <out.npz>
"""

import sys

import numpy as np


DIR_INDEX = {"HORIZONTAL": 0, "VERTICAL": 1}


def congestion(in_path):
    meta = {}
    usage_rows = []
    x_grid = None
    y_grid = None
    with open(in_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "meta":
                meta[parts[1]] = parts[2]
            elif parts[0] == "grid_x":
                x_grid = [int(v) for v in parts[1:]]
            elif parts[0] == "grid_y":
                y_grid = [int(v) for v in parts[1:]]
            elif parts[0] == "usage":
                usage_rows.append(parts[2:])

    nx = len(x_grid)
    ny = len(y_grid)
    pitch_x = x_grid[1] - x_grid[0] if nx > 1 else 0
    pitch_y = y_grid[1] - y_grid[0] if ny > 1 else 0
    if nx > 2 and any(x_grid[i + 1] - x_grid[i] != pitch_x for i in range(nx - 1)):
        raise RuntimeError(f"Non-uniform X grid pitch in {in_path}: {x_grid}")
    if ny > 2 and any(y_grid[i + 1] - y_grid[i] != pitch_y for i in range(ny - 1)):
        raise RuntimeError(f"Non-uniform Y grid pitch in {in_path}: {y_grid}")

    demand = np.zeros((ny, nx, 2), dtype=np.float64)
    capacity = np.zeros((ny, nx, 2), dtype=np.float64)
    # Rows are in (layer, xi, yi) order, the same accumulation order as
    # extract_congestion.py's loops.
    for direction, xi, yi, use, cap in usage_rows:
        d = DIR_INDEX[direction]
        demand[int(yi), int(xi), d] += float(use)
        capacity[int(yi), int(xi), d] += float(cap)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(capacity > 0, demand / capacity, 0.0)

    metadata = dict(
        origin_x=int(x_grid[0]),
        origin_y=int(y_grid[0]),
        pitch_x=int(pitch_x),
        pitch_y=int(pitch_y),
        nx=nx,
        ny=ny,
        dbu_per_micron=int(meta["dbu_per_micron"]),
        source_odb=meta["source_odb"],
    )
    return dict(demand=demand, capacity=capacity, ratio=ratio, **metadata)


def placement(in_path):
    meta = {}
    cell_names = []
    cell_x = []
    cell_y = []
    cell_w = []
    cell_h = []
    cell_pin_count = []
    net_names = []
    net_pins = []
    edge_cell_idx = []
    edge_net_idx = []
    with open(in_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            kind = parts[0]
            if kind == "meta":
                meta[parts[1]] = parts[2]
            elif kind == "cell":
                cell_names.append(parts[1])
                cell_x.append(int(parts[2]))
                cell_y.append(int(parts[3]))
                cell_w.append(int(parts[4]))
                cell_h.append(int(parts[5]))
                cell_pin_count.append(int(parts[6]))
            elif kind == "net":
                net_names.append(parts[1])
                net_pins.append([])
            elif kind == "pin":
                edge_cell_idx.append(int(parts[1]))
                edge_net_idx.append(int(parts[2]))
                net_pins[int(parts[2])].append((int(parts[3]), int(parts[4])))

    net_fanout = []
    net_hpwl = []
    net_bbox_w = []
    net_bbox_h = []
    for pins in net_pins:
        pin_xs = [p[0] for p in pins]
        pin_ys = [p[1] for p in pins]
        bbox_w = max(pin_xs) - min(pin_xs)
        bbox_h = max(pin_ys) - min(pin_ys)
        net_fanout.append(len(pins))
        net_hpwl.append(bbox_w + bbox_h)
        net_bbox_w.append(bbox_w)
        net_bbox_h.append(bbox_h)

    data = dict(
        cell_names=cell_names,
        cell_x=cell_x, cell_y=cell_y, cell_w=cell_w, cell_h=cell_h,
        cell_pin_count=cell_pin_count,
        net_names=net_names,
        net_fanout=net_fanout, net_hpwl=net_hpwl,
        net_bbox_w=net_bbox_w, net_bbox_h=net_bbox_h,
        edge_cell_idx=edge_cell_idx, edge_net_idx=edge_net_idx,
        die_w=int(meta["die_w"]),
        die_h=int(meta["die_h"]),
        site_w=int(meta["site_w"]),
        site_h=int(meta["site_h"]),
        dbu_per_micron=int(meta["dbu_per_micron"]),
        source_odb=meta["source_odb"],
    )
    return {k: np.asarray(v) for k, v in data.items()}


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("congestion", "placement"):
        print(f"Usage: python {sys.argv[0]} congestion|placement <in.txt> <out.npz>")
        sys.exit(1)
    mode = sys.argv[1]
    in_path = sys.argv[2]
    out_path = sys.argv[3]
    if mode == "congestion":
        out = congestion(in_path)
        np.savez(out_path, **out)
        nonzero = int(np.count_nonzero(out["demand"]))
        print(f"Wrote {out_path}: shape={out['demand'].shape} nonzero_demand={nonzero}/{out['demand'].size} "
              f"max_ratio={out['ratio'].max():.4f}")
    else:
        out = placement(in_path)
        np.savez(out_path, **out)
        print(f"Wrote {out_path}: {len(out['cell_names'])} cells, "
              f"{len(out['net_names'])} nets, {len(out['edge_cell_idx'])} pin edges")


if __name__ == "__main__":
    main()
