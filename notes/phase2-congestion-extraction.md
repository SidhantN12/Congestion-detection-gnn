# Phase 2 — Ground truth extraction

## What ended up being the real API

Investigated three candidate routes before settling on one, in this order:

1. **`global_route -congestion_report_file <file>`** (documented in
   OpenROAD's `src/grt/README.md`). Ruled out: the docs describe this as
   "can be read by the DRC viewer in the GUI" - it's a *sparse marker
   report of overflow violations*, not a dense per-GCell grid. Confirmed
   empirically: ORFS's own flow already passes this flag
   (`flow/scripts/global_route.tcl:47`, writing to
   `reports/.../congestion.rpt`), and the file simply doesn't exist after
   our Phase 1 run - there's no overflow to report on a design this small,
   so nothing gets written. Not usable for a dense array regardless.

2. **GUI heatmap CSV export**, `gui::dump_heatmap <name> <file>`
   (documented in `src/gui/README.md`). This *does* produce a dense
   per-GCell grid, and works headlessly in this container via the same
   `gui::show { script } false` pattern ORFS itself uses for
   `final_congestion.webp` in `flow/scripts/save_images.tcl`. Used this
   as the independent ground truth for verification (see below), but it's
   a single combined percentage per GCell (see formula below), not split
   into demand/capacity/horizontal/vertical - not the right shape for
   `extract_congestion.py`'s actual output, only for checking it.

3. **OpenDB's `dbGCellGrid` object directly** (`odb/include/odb/db.h`) -
   what `extract_congestion.py` actually uses. Confirmed (via
   `grt/src/GlobalRouter.cpp:updateDbCongestionFromGuides`) that
   `global_route` populates this object on the block as a side effect,
   with per-(layer, x, y) `capacity`/`usage` floats.

## Two API gaps hit along the way

- `dbGCellGrid.getDirectionCongestionMap(direction)` is the natural
  one-call API (returns a whole direction's aggregated matrix at once)
  but needs a `dbTechLayerDir` C++ enum value as input. Python's SWIG
  binding only exposes `layer.getDirection()` as an output-converted
  plain string ("HORIZONTAL"/"VERTICAL") - there's no
  `odb.dbTechLayerDir.HORIZONTAL` symbol to construct one to pass back
  in. Confirmed via runtime `TypeError: argument 2 of type 'dbTechLayerDir
  const &'`, not by reading source alone.
- `dbGCellGrid.getLayerCongestionMap(layer)` (only needs a `dbTechLayer*`,
  which we do have) *does* accept its argument fine, but its
  `dbMatrix<GCellData>` return type comes back as a raw, unusable
  `SwigPyObject` ("no destructor found", no attribute access) - a gap in
  this build's bindings for that templated return type, not something
  fixable from the Python side.
- Workaround, used in `extract_congestion.py`: the scalar accessors
  `getCapacity(layer, x_idx, y_idx)` / `getUsage(layer, x_idx, y_idx)`
  return plain floats with no issues. Loop over every (routing layer, x,
  y) triple and aggregate by direction in Python. Confirmed this
  reproduces the C++ implementation exactly (see verification below) -
  `src/grt/src/heatMap.cpp`'s `RoutingCongestionDataSource` does the same
  per-layer-direction summation in C++, just once, via the API gap above.

## `extract_congestion.py`

Must run through OpenROAD's *own* bundled Python interpreter
(`openroad -python extract_congestion.py <in.odb> <out.npz>`), not the
project venv - `ord`/`odb` only exist inside `openroad -python`, they are
not pip-installable.

Output: `.npz` with
- `demand`, `capacity`, `ratio`: each shape `(ny, nx, 2)`, last axis =
  [horizontal, vertical]. Row-major on Y so it matches image/`imshow`
  convention (use `origin='lower'` when plotting, to match the GUI's
  bottom-left-origin coordinate system).
- `origin_x`, `origin_y`, `pitch_x`, `pitch_y` (DBU), `dbu_per_micron`,
  `nx`, `ny`: grid metadata for Phase 5's alignment check.

Ran against `flow/results/nangate45/gcd/base/5_1_grt.odb` (the
global-route-stage output from Phase 1): shape `(12, 12, 2)`, 266/288
nonzero demand cells, max per-direction ratio 0.7778 - i.e. **not
all-zero**, consistent with a genuinely (if mildly) congested tiny design,
not an empty one.

## Verification (`tests/verify_congestion_extraction.py`)

OpenROAD's "Routing" heatmap, in its default all-directions mode, reports
`value(%) = max(horizontal_ratio, vertical_ratio) * 100` per GCell -
confirmed by reading `RoutingCongestionDataSource::setCongestionValues` in
`src/grt/src/heatMap.cpp` (the `else` branch, when direction isn't
restricted to just H or just V). Reproduced that exact combination from
`extract_congestion.py`'s two-channel `ratio` array and compared
cell-by-cell against a `gui::dump_heatmap Routing <csv>` dump of the
*same* `.odb`.

Result: all 144/144 GCells matched, **max absolute difference 0.000005**
(floating-point noise only). Side-by-side heatmap + diff plot:
`data/raw/gcd_baseline_congestion_comparison.png` (gitignored like the
rest of `data/raw/*` - regenerate via the commands below rather than
expecting it to be in git).

## Commands to reproduce

```bash
# 1. Extract congestion (inside the ORFS container, both flow/ and the
#    project dir mounted):
docker run --rm -u 1000:1000 \
  -v ~/congestion-gnn/OpenROAD-flow-scripts/flow:/work \
  -v "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project:/project" \
  -e FLOW_HOME=/OpenROAD-flow-scripts/flow/ -e WORK_HOME=/work \
  openroad/orfs:latest bash -c '
    cd /OpenROAD-flow-scripts/flow && source ../env.sh &&
    openroad -python "/project/src/extract_congestion.py" \
      /work/results/nangate45/gcd/base/5_1_grt.odb \
      "/project/data/raw/gcd_baseline_grt_congestion.npz"'

# 2. (Optional, for verification only) dump the GUI's own heatmap for the
#    same odb - write a small Tcl script containing:
#      read_db /work/results/nangate45/gcd/base/5_1_grt.odb
#      gui::show "gui::set_heatmap Routing rebuild; \
#        gui::dump_heatmap Routing /work/gui_dump.csv" false
#    then: util/docker_shell openroad -no_init /work/that_script.tcl

# 3. Compare (plain project venv, no OpenROAD needed):
source ~/congestion-gnn/.venv/bin/activate
cd "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project"
python tests/verify_congestion_extraction.py \
  data/raw/gcd_baseline_grt_congestion.npz \
  data/raw/gcd_baseline_gui_dump.csv \
  data/raw/gcd_baseline_congestion_comparison.png
```
