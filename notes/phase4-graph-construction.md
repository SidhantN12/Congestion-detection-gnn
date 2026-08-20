# Phase 4 — Graph construction

## DEF parsing: OpenROAD Python API vs. standalone parser

**Recommendation: OpenROAD's own Python API (`odb` via `openroad -python`),
not a standalone DEF/LEF parser.** Reasons, in order of weight:

1. **A standalone parser can't get GCell grid data at all, regardless of
   effort.** Checked directly: at the placement stage, our DEF files have
   no `GCELLGRID` section at all (`grep -c GCELLGRID 3_place.odb.def` ->
   0 matches), and `dbGCellGrid` itself doesn't exist in-memory until
   `global_route` creates it (confirmed in Phase 2 -
   `GlobalRouter.cpp:updateDbCongestionFromGuides`). This isn't a parser
   limitation - the data structure a standalone parser would need to
   read simply doesn't exist as static DEF content at any stage a
   text-format parser could intercept. Getting gcell/congestion data
   requires actually running OpenROAD regardless of what parses the
   cell/net geometry, so there's no "avoid OpenROAD entirely" option on
   the table to begin with.
2. **Exact coordinate-system consistency with Phase 2's congestion
   extraction.** Using the same `odb` API for both means cell positions,
   die area, and GCell grid indexing all come from the identical
   in-memory representation OpenROAD itself uses - no risk of a
   standalone parser's DEF-unit handling or GCell-derivation logic
   silently drifting from OpenROAD's own (this matters a lot for Phase
   5's alignment check).
3. Already proven reliable in Phase 2 - reusing known-working API
   patterns (`ord.Tech()`, `ord.Design(tech)`, `design.readDb(path)`,
   `block.getInsts()`/`getNets()`/etc.) rather than introducing a second,
   independent parsing stack.

## Two scripts, two environments (a real constraint, not a design choice)

Confirmed empirically that `openroad -python` cannot import `torch`:

```
$ openroad -python -c "import torch"
torch NOT available: No module named 'torch'
sys.executable: /OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad
```

It's a separate embedded Python 3.10 inside the container, entirely
disconnected from our project's venv. So graph construction is
necessarily two scripts:

- **`extract_placement.py`** (run via `openroad -python`): reads a
  *placed* `.odb` (e.g. `3_place.odb`), dumps cell/net/pin arrays to a
  plain `.npz` (numpy only, confirmed available in that interpreter from
  Phase 2's `extract_congestion.py`). No GCell grid access here - that
  doesn't exist yet at the placement stage (see above); grid geometry is
  taken from `extract_congestion.py`'s output instead in the next step.
- **`build_graph.py`** (run in the project venv): loads that `.npz`
  together with `extract_congestion.py`'s `.npz` (from the matching
  post-route `.odb` of the *same* run) and assembles the actual PyG
  `HeteroData` object. This is also where gcell "capacity per direction"
  comes from - see the note on that below.

## A design tension worth naming: capacity needs global_route to extract, but not conceptually

"Routing capacity per direction" (a gcell input feature) depends only on
the tech's routing layers, any layer-adjustment/blockage configuration,
and the GCell grid geometry - none of which depend on how nets actually
get routed. In principle it's knowable from a placed-but-unrouted design,
matching the project's "without running a router" framing. In practice,
the only tool-supported way found so far to read it out (`dbGCellGrid`)
requires `global_route` to have run at least once to create and populate
that object. Worked around it pragmatically here by taking capacity from
`extract_congestion.py`'s output (from the post-route odb of the same
run) rather than re-deriving it independently - fine for this pipeline-
validation phase, but worth remembering: a "true" from-placement-only
capacity extraction would need a lighter-weight capacity-only computation
that doesn't currently exist in what we've explored, if that ever matters
for the eventual real (non-toy) pipeline.

## Feature design (technology-relative units throughout)

| Node | Features | Units |
|---|---|---|
| cell | width/site_w, height/site_h, pin_count, x/die_w, y/die_h | site-width/height multiples; dimensionless count; [0,1] normalized position |
| net | fanout, HPWL/site_w, bbox aspect ratio | dimensionless count; site-width multiples; dimensionless ratio |
| gcell | cell_density, pin_density, capacity_horizontal, capacity_vertical | site-area occupied per gcell-area-in-sites (both densities); routing tracks (capacity, already track-based per `extract_congestion.py`/GRT's own units) |

Cell/pin "density" per gcell: `sum(cell_width_in_sites * cell_height_in_sites)
/ (gcell_pitch_x/site_w * gcell_pitch_y/site_h)` - a site-area occupancy
fraction, technology-relative by construction (never raw DBU/microns).

Nets filtered to `SIGNAL`+`CLOCK` sigType (excludes `POWER`/`GROUND`,
handled by the PDN separately, not part of the routed signal netlist) -
confirmed via `Counter` over `net.getSigType()`: 630 SIGNAL + 1 CLOCK + 1
POWER + 1 GROUND on the baseline design, matching Phase 1's DEF header
exactly (`NETS 631` = 630+1, `SPECIALNETS 2` = the power/ground pair).

## Gate results (baseline design, `3_place.odb` + `5_1_grt.odb`)

```
cell:  606 nodes
net:   631 nodes
gcell: 144 nodes (12x12, matching Phase 2/3's grid)

cell -pin-> net:        1707 edges (out-deg 0-5, in-deg 1-35)
net -rev_pin-> cell:    1707 edges
cell -in-> gcell:        606 edges (every cell in exactly 1 gcell)
gcell -contains-> cell:  606 edges (0-8 cells per gcell)
cell -near-> cell:      4848 edges (exactly k=8 out-degree everywhere)
gcell -adjacent-> gcell: 528 edges (2-4, matching grid boundary/interior)
```

Gate checks: **0 isolated gcell nodes** (every gcell has >=1
gcell-adjacent-gcell edge), **0 nets with zero pin edges**. Both pass -
see `tests/verify_graph_construction.py`.

One honest note, not a failure: cell out-degree on `cell-pin-net` has a
minimum of 0 - some cells (tapcells/endcaps, inserted during floorplan,
already present in `3_place.odb`) have no SIGNAL/CLOCK pins at all, only
power/ground connections, which are filtered out by design. Also 5/144
gcells have zero cells assigned - plausible on a sparse ~600-cell design
across a 12x12 grid (~4.2 cells/gcell average) and not itself a problem,
just noted for completeness (checked, not blindly assumed benign).

## Commands to reproduce

```bash
# Placement extraction (odb-side, dual-mount pattern from Phase 2/3):
docker run --rm -u 1000:1000 \
  -v ~/congestion-gnn/OpenROAD-flow-scripts/flow:/work \
  -v "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project:/project" \
  -e FLOW_HOME=/OpenROAD-flow-scripts/flow/ -e WORK_HOME=/work \
  openroad/orfs:latest bash -c '
    cd /OpenROAD-flow-scripts/flow && source ../env.sh &&
    openroad -python "/project/src/extract_placement.py" \
      /work/results/nangate45/gcd/base/3_place.odb \
      "/project/data/raw/gcd_baseline_placement.npz"'

# Graph assembly + gate checks (project venv):
source ~/congestion-gnn/.venv/bin/activate
cd "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project"
python src/build_graph.py \
  data/raw/gcd_baseline_placement.npz \
  data/raw/gcd_baseline_grt_congestion.npz \
  data/processed/gcd_baseline_graph.pt
python tests/verify_graph_construction.py
```
