"""Extract per-GCell routing congestion from an OpenROAD .odb into a NumPy array.

Must be run through OpenROAD's own bundled Python interpreter, not a normal
venv `python` - the `ord`/`odb` modules only exist inside `openroad -python`.
`ord`/`odb` come from that interpreter and are not installable via pip.

Usage (from inside the ORFS docker container, or any environment where
`openroad` is on PATH):

    openroad -python extract_congestion.py <input.odb> <output.npz>

Data source: OpenDB's dbGCellGrid, populated by `global_route` (grt module).
Congestion is a global-route-stage concept - run this against a post-global-
route .odb (e.g. flow/results/<platform>/<design>/base/5_1_grt.odb), not a
pre-route or detailed-routed one.

dbGCellGrid.getDirectionCongestionMap() would be the natural one-call API for
this (see OpenROAD src/odb/include/odb/db.h), but its dbMatrix<GCellData>
return type is not usably wrapped in this build's Python bindings (SWIG
reports "no destructor found" and returns an opaque SwigPyObject with no
accessors). Falling back to the scalar dbGCellGrid.getCapacity(layer, x, y) /
getUsage(layer, x, y) accessors instead, aggregating across routing layers by
direction in Python - this reproduces exactly what getDirectionCongestionMap
does in C++ (see OpenROAD src/odb/src/db/dbGCellGrid.cpp), just one layer at
a time.
"""

import sys

import numpy as np
import ord


DIR_INDEX = {"HORIZONTAL": 0, "VERTICAL": 1}


def extract_congestion(odb_path):
    tech = ord.Tech()
    design = ord.Design(tech)
    design.readDb(odb_path)

    block = design.getBlock()
    gcell = block.getGCellGrid()
    if gcell is None:
        raise RuntimeError(
            f"{odb_path} has no GCell grid - was global_route ever run on "
            "this design? (dbGCellGrid is created/populated by the grt "
            "module during global_route, not at load time.)"
        )

    db_tech = tech.getDB().getTech()
    x_grid = gcell.getGridX()
    y_grid = gcell.getGridY()
    nx, ny = len(x_grid), len(y_grid)

    pitch_x = x_grid[1] - x_grid[0] if nx > 1 else 0
    pitch_y = y_grid[1] - y_grid[0] if ny > 1 else 0
    # Sanity check: everything downstream (Phase 4/5 alignment) assumes a
    # uniform grid pitch, which is what OpenROAD's GCell grid always is in
    # practice, but verify rather than assume.
    if nx > 2 and any(x_grid[i + 1] - x_grid[i] != pitch_x for i in range(nx - 1)):
        raise RuntimeError(f"Non-uniform X grid pitch in {odb_path}: {x_grid}")
    if ny > 2 and any(y_grid[i + 1] - y_grid[i] != pitch_y for i in range(ny - 1)):
        raise RuntimeError(f"Non-uniform Y grid pitch in {odb_path}: {y_grid}")

    routing_layers = [
        layer for layer in db_tech.getLayers() if str(layer.getType()) == "ROUTING"
    ]

    # array layout: [row=y_idx, col=x_idx, dir] to match image/imshow
    # convention (use origin='lower' when plotting to match GUI's
    # bottom-left-origin coordinate system).
    demand = np.zeros((ny, nx, 2), dtype=np.float64)
    capacity = np.zeros((ny, nx, 2), dtype=np.float64)

    for layer in routing_layers:
        d = DIR_INDEX[layer.getDirection()]
        for xi in range(nx):
            for yi in range(ny):
                demand[yi, xi, d] += gcell.getUsage(layer, xi, yi)
                capacity[yi, xi, d] += gcell.getCapacity(layer, xi, yi)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(capacity > 0, demand / capacity, 0.0)

    metadata = dict(
        origin_x=int(x_grid[0]),
        origin_y=int(y_grid[0]),
        pitch_x=int(pitch_x),
        pitch_y=int(pitch_y),
        nx=nx,
        ny=ny,
        dbu_per_micron=int(block.getDbUnitsPerMicron()),
        source_odb=odb_path,
    )
    return demand, capacity, ratio, metadata


def main():
    if len(sys.argv) != 3:
        print(f"Usage: openroad -python {sys.argv[0]} <input.odb> <output.npz>")
        sys.exit(1)

    odb_path, out_path = sys.argv[1], sys.argv[2]
    demand, capacity, ratio, metadata = extract_congestion(odb_path)

    np.savez(
        out_path,
        demand=demand,
        capacity=capacity,
        ratio=ratio,
        **metadata,
    )

    nonzero = int(np.count_nonzero(demand))
    total = demand.size
    print(f"Wrote {out_path}: shape={demand.shape} nonzero_demand={nonzero}/{total} "
          f"max_ratio={ratio.max():.4f} metadata={metadata}")


if __name__ == "__main__":
    main()
