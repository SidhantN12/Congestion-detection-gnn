# Phase 3 — Inducing congestion

## Summary of what actually worked

Of the four levers, only **explicit routing blockages** produced real
overflow (combined congestion > 1.0) on this design. Layer capping alone,
and the density/core-utilization levers, did not - see findings below.
The working combination: layer cap (metal2-metal3) + an explicit routing
blockage band across 40-60% of the die height, on the same capped layers.

| Experiment | max combined congestion | fraction > 0.9 |
|---|---|---|
| baseline (Phase 1/2, full stack, default density) | 0.7778 | 0/144 (0%) |
| layer cap only (metal2-metal3) | 0.6364 | 0/289 (0%) |
| layer cap + die-band blockage | **1.6364** | **186/289 (64.4%)** |

Heatmap: `data/raw/phase3_congestion_comparison.png` (gitignored,
regenerate via `tests/phase3_congestion_report.py`).

## An important environment gotcha discovered along the way

`util/docker_shell` only bind-mounts the current directory as
`WORK_HOME=/work`. `FLOW_HOME=/OpenROAD-flow-scripts/flow` (where
`platforms/`, `scripts/`, etc. live) is the **image's own internal copy**
of the ORFS repo, baked in at image build time - completely separate from
our host `~/congestion-gnn/OpenROAD-flow-scripts` clone. Editing the host
clone's `.tcl`/`.mk` files has **zero effect** on what actually runs;
confirmed empirically (added a `puts` debug line to the host copy of
`platforms/nangate45/fastroute.tcl`, reran, saw nothing - the container
never read that file). Any override file (custom hook script, replacement
`FASTROUTE_TCL`, etc.) must live inside whatever directory is bind-mounted
as `/work`, and be referenced via an absolute `/work/...` path in the
relevant env var. Our sweep `.tcl` files are tracked in this repo's
`flow/` directory but must be **copied into the WSL ORFS checkout's
`flow/`** (or that whole checkout's directory bind-mounted alongside this
project's, as done for `extract_congestion.py` runs) before use - see
"Commands to reproduce" below.

## Lever 1: cap routing layers to metal2-metal3 (biggest expected lever)

`MIN_ROUTING_LAYER`/`MAX_ROUTING_LAYER` (`platforms/nangate45/config.mk:74,76`,
default `metal2`/`metal10`) control the signal routing layer range. Metal1
is excluded from GRT even at platform defaults (Nangate45-specific rule,
not our choice) so "metal1-metal3" from the brief becomes metal2-metal3
in practice.

**Gotcha**: `platforms/nangate45/fastroute.tcl:4` also sets
`-clock $MIN_CLK_ROUTING_LAYER-$MAX_ROUTING_LAYER`, and
`MIN_CLK_ROUTING_LAYER` defaults to `metal4` (`config.mk:75`). Capping
`MAX_ROUTING_LAYER` to `metal3` alone makes the clock range invalid
(`metal4-metal3`, min > max) and hard-errors
(`ODB-0137: min routing layer is greater than max routing layer`) before
routing even starts. Also needed `MIN_CLK_ROUTING_LAYER=metal2` (or
metal3) - and per OpenROAD's `set_routing_layers`, min == max is *also*
rejected (same ODB-0137 error), it must be a strict `<`.

**Result: did not increase congestion.** Ran `make grt` with
`MAX_ROUTING_LAYER=metal3 MIN_CLK_ROUTING_LAYER=metal2` alone: max
combined congestion actually *dropped* slightly (0.7778 -> 0.6364)
relative to baseline. The GCell grid also became finer (12x12 @ 5700 DBU
pitch -> 17x17 @ 4200 DBU pitch) since GCell size derives from the top
routing layer's track pitch (GRT's own convention, ~15 M3 pitches here
vs whatever M10-relative sizing applied before) - more, smaller GCells
diluted the per-cell demand rather than concentrating it. Also relevant:
detailed route (`5_2_route`) failed on this capped stack with
`initMazeIdx_connFig via no idx` errors, almost certainly because the PDN
power grid (`grid_strategy-M1-M4-M7.tcl`, from `designs/nangate45/gcd/
config.mk`) still straps M4/M7 regardless of the signal-layer cap, and
detailed route couldn't resolve vias into a signal stack that stops at
M3. Not an obstacle for this project since congestion ground truth
(`dbGCellGrid`) comes from global route, not detailed route - used the
dedicated `make grt` target (builds only through `5_1_grt.odb`) instead
of the full `make`/`all` flow.

## Lever 2: raise PLACE_DENSITY toward 0.9+

`PLACE_DENSITY` (`platforms/nangate45/config.mk:68`, default `0.30`) is
RePlAce's target packing density, distinct from `CORE_UTILIZATION` (which
sizes the floorplan). Overridable directly via `make ... PLACE_DENSITY=0.90`.

**Result: no effect when combined with the layer cap alone** (identical
max congestion 0.6364, identical nonzero-cell count, to the layer-cap-only
run). GRT's own congestion-removal iterations (up to
`-congestion_iterations 30`, see `scripts/global_route.tcl:42`) actively
redistribute routing to avoid overflow wherever any slack exists;
raising the *placement* density target didn't change what routing
resources GRT had to work with, so it made no measurable difference here.

## Lever 3: shrink core area (raise CORE_UTILIZATION)

`CORE_UTILIZATION` (`designs/nangate45/gcd/config.mk:11`, currently `55`)
sizes the floorplan (`scripts/floorplan.tcl:83`, via
`initialize_floorplan -utilization`).

**Result: infeasible above ~80% for this specific design, independent of
PLACE_DENSITY/PLACE_DENSITY_LB_ADDON.** Tried `CORE_UTILIZATION=85` and
`90`, with `PLACE_DENSITY` at both its 0.30 default and raised to 0.90,
and `PLACE_DENSITY_LB_ADDON` (gcd's own tuning, `config.mk:12`, default
`0.20`) at both its default and `0`. Every combination hard-errored at
global placement: `[ERROR FLW-0024] Place density exceeds 1.0`. The
resulting floorplan utilization (`89-96%`, reported by the tool) already
left too little slack for RePlAce's legalizer once tapcells/endcaps
(fixed per-row overhead, proportionally larger on a tiny ~500-cell
design) are accounted for - this isn't a target-density knob problem, the
physical core is too tight to legalize *regardless* of what density
target is requested. Abandoned this lever for gcd; would likely need a
much smaller CORE_UTILIZATION increase (maybe 65-70) to stay feasible, or
matters differently on a bigger design.

## Lever 4: routing blockages (what actually worked)

No existing ORFS variable adds an arbitrary rectangular blockage -
`scripts/add_routing_blk.tcl` exists but is hardcoded to GF12 macro
instance names (`*gf12*`/`IN12LP*`), not usable for a standard-cell-only
design like gcd. Used the same underlying primitive
(`odb::dbObstruction_create`) in a custom script
(`flow/add_die_blockage.tcl`), wired in via the `PRE_GLOBAL_ROUTE_TCL`
hook (`scripts/global_route.tcl:9`, `source_step_tcl PRE GLOBAL_ROUTE` -
fires right after `load_design`, so cells are already placed/CTS'd,
meaning this only removes *routing* capacity, doesn't disturb placement).
Blocks a horizontal band spanning the full die width, at 40-60% of die
height, on every layer in the capped range (metal2-metal3).

**Result: real overflow.** GRT's own final congestion report showed
"Total Congestion 389" (edge-instances over capacity) after all 30
removal iterations - it could not route around the blockage cleanly.
Global route exited with `[ERROR GRT-0116] Global routing finished with
congestion` (a strict-mode gate, not a crash) but still wrote a fully
valid `5_1_grt-failed.odb` (`GENERATE_ARTIFACTS_ON_FAILURE` must be on by
default in this image) with real `dbGCellGrid` data - that's the file
`extract_congestion.py` read for the numbers above. Tried
`GLOBAL_ROUTE_ARGS=-allow_congestion` (a documented passthrough env var,
`scripts/global_route.tcl:48`) to get a clean non-error pass instead of
relying on the `-failed.odb` fallback, but it only suppresses the check
on the *first* `global_route` call - the later incremental repair-stage
calls (`repair_design`/`repair_timing`/`recover_power` helpers,
`scripts/global_route.tcl:91-92,112-113,142-143`) don't reference
`GLOBAL_ROUTE_ARGS` at all, so they hit a *different* strict check
(`GRT-0232`) instead. Not worth chasing further - the `-failed.odb` file
already contains everything `extract_congestion.py` needs (the file is
not corrupt, just flagged), and `5_1_grt` is the right pipeline stage for
this project regardless of what happens in later incremental stages.

## Commands to reproduce

```bash
# From ~/congestion-gnn/OpenROAD-flow-scripts/flow (WSL) - hook scripts
# must be copied into this bind-mounted tree first (see gotcha above):
cp "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project/flow/fastroute_capped_metal3.tcl" .
cp "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project/flow/add_die_blockage.tcl" .

rm -rf results/nangate45/gcd/blk1 logs/nangate45/gcd/blk1 \
       objects/nangate45/gcd/blk1 reports/nangate45/gcd/blk1

util/docker_shell make grt FLOW_VARIANT=blk1 \
  FASTROUTE_TCL=/work/fastroute_capped_metal3.tcl \
  MAX_ROUTING_LAYER=metal3 MIN_CLK_ROUTING_LAYER=metal2 \
  PRE_GLOBAL_ROUTE_TCL=/work/add_die_blockage.tcl

# Extract congestion (dual-mount pattern, from Phase 2):
docker run --rm -u 1000:1000 \
  -v ~/congestion-gnn/OpenROAD-flow-scripts/flow:/work \
  -v "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project:/project" \
  -e FLOW_HOME=/OpenROAD-flow-scripts/flow/ -e WORK_HOME=/work \
  openroad/orfs:latest bash -c '
    cd /OpenROAD-flow-scripts/flow && source ../env.sh &&
    openroad -python "/project/src/extract_congestion.py" \
      /work/results/nangate45/gcd/blk1/5_1_grt-failed.odb \
      "/project/data/raw/gcd_blk1_congestion.npz"'

# Report + heatmap (project venv):
source ~/congestion-gnn/.venv/bin/activate
cd "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project"
python tests/phase3_congestion_report.py
```
