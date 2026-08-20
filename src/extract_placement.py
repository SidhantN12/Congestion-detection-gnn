"""Extract raw placement/netlist/gcell-grid data from an OpenROAD .odb into
a plain NumPy .npz - the odb-side half of graph construction.

Must run through OpenROAD's own bundled Python interpreter (`openroad
-python`), not the project venv - same constraint as extract_congestion.py,
and for the same reason: `ord`/`odb` only exist there. Conversely, this
script does NOT import torch/torch_geometric - confirmed empirically that
`openroad -python` cannot see the venv's packages (separate Python
installation entirely, system Python 3.10 inside the container, no access
to our pip environment). build_graph.py (plain venv Python) is the
counterpart that turns this file's output into a PyG HeteroData object.

Run against a PLACED, pre-route .odb (e.g. .../3_place.odb) - this is the
"placed-but-unrouted design" the project predicts congestion from. Routing
capacity (needed for gcell features) is a property of the tech/layer-range/
blockage configuration, not of how nets end up routed, but this build's
dbGCellGrid is only actually populated by global_route (see
notes/phase2-congestion-extraction.md) - so capacity is instead pulled
from extract_congestion.py's output (run against the matching post-route
.odb) in build_graph.py, rather than re-derived here.

Usage:
    openroad -python extract_placement.py <placed.odb> <output.npz>
"""

import sys

import ord


def extract_placement(odb_path):
    tech = ord.Tech()
    design = ord.Design(tech)
    design.readDb(odb_path)
    block = design.getBlock()

    die = block.getDieArea()
    die_w = die.xMax() - die.xMin()
    die_h = die.yMax() - die.yMin()

    rows = block.getRows()
    if len(rows) == 0:
        raise RuntimeError(f"{odb_path} has no placement rows - is this a placed odb?")
    site = rows[0].getSite()
    site_w, site_h = site.getWidth(), site.getHeight()

    # GCell grid geometry (origin/pitch/nx/ny) is NOT available yet at this
    # stage - dbGCellGrid is only created/populated by global_route (see
    # notes/phase2-congestion-extraction.md), and this script is meant to
    # run against a pre-route placed odb. build_graph.py instead takes grid
    # geometry from extract_congestion.py's output for the matching
    # post-route odb of the same design/run.

    # Cells: one row per instance. Index = position in this array, used as
    # the cell-node id downstream.
    insts = block.getInsts()
    cell_names = []
    cell_x = []
    cell_y = []
    cell_w = []
    cell_h = []
    cell_pin_count = []
    inst_to_cell_idx = {}
    for idx, inst in enumerate(insts):
        master = inst.getMaster()
        loc = inst.getLocation()
        iterms = [it for it in inst.getITerms() if str(it.getSigType()) in ("SIGNAL", "CLOCK")]
        cell_names.append(inst.getName())
        cell_x.append(loc[0])
        cell_y.append(loc[1])
        cell_w.append(master.getWidth())
        cell_h.append(master.getHeight())
        cell_pin_count.append(len(iterms))
        inst_to_cell_idx[inst.getId()] = idx

    # Nets (SIGNAL + CLOCK only - excludes POWER/GROUND, handled by the PDN
    # separately, not part of the routed signal netlist graph) and the
    # cell<->net pin-connectivity edges, built in the same pass.
    nets = block.getNets()
    net_names = []
    net_fanout = []
    net_hpwl = []
    net_bbox_w = []
    net_bbox_h = []
    edge_cell_idx = []
    edge_net_idx = []
    for net_idx, net in enumerate(nets):
        sig_type = str(net.getSigType())
        if sig_type not in ("SIGNAL", "CLOCK"):
            continue

        iterms = net.getITerms()
        pin_xs = []
        pin_ys = []
        for it in iterms:
            inst = it.getInst()
            cell_idx = inst_to_cell_idx[inst.getId()]
            edge_cell_idx.append(cell_idx)
            edge_net_idx.append(len(net_names))  # index into the filtered net list below
            bbox = it.getBBox()
            pin_xs.append(bbox.xCenter())
            pin_ys.append(bbox.yCenter())

        if len(pin_xs) == 0:
            # A net with zero instance pins (e.g. only a top-level port) -
            # exclude, it can't have a cell-net edge regardless.
            continue

        bbox_w = max(pin_xs) - min(pin_xs)
        bbox_h = max(pin_ys) - min(pin_ys)
        net_names.append(net.getName())
        net_fanout.append(len(iterms))
        net_hpwl.append(bbox_w + bbox_h)
        net_bbox_w.append(bbox_w)
        net_bbox_h.append(bbox_h)

    metadata = dict(
        die_w=die_w,
        die_h=die_h,
        site_w=site_w,
        site_h=site_h,
        dbu_per_micron=int(block.getDbUnitsPerMicron()),
        source_odb=odb_path,
    )
    return dict(
        cell_names=cell_names,
        cell_x=cell_x, cell_y=cell_y, cell_w=cell_w, cell_h=cell_h,
        cell_pin_count=cell_pin_count,
        net_names=net_names,
        net_fanout=net_fanout, net_hpwl=net_hpwl,
        net_bbox_w=net_bbox_w, net_bbox_h=net_bbox_h,
        edge_cell_idx=edge_cell_idx, edge_net_idx=edge_net_idx,
        **metadata,
    )


def main():
    if len(sys.argv) != 3:
        print(f"Usage: openroad -python {sys.argv[0]} <placed.odb> <output.npz>")
        sys.exit(1)

    odb_path, out_path = sys.argv[1], sys.argv[2]
    data = extract_placement(odb_path)

    # np.savez needs numpy: available in the container's system python
    # (confirmed present) even without torch.
    import numpy as np
    np.savez(out_path, **{k: np.asarray(v) for k, v in data.items()})

    print(f"Wrote {out_path}: {len(data['cell_names'])} cells, "
          f"{len(data['net_names'])} nets, {len(data['edge_cell_idx'])} pin edges")


if __name__ == "__main__":
    main()
