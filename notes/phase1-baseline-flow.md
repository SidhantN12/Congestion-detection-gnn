# Phase 1 — Baseline flow (gcd / nangate45)

## Command

Ran from `OpenROAD-flow-scripts/flow/`:

```
util/docker_shell make
```

`gcd`/`nangate45` is the Makefile's own default (`DESIGN_CONFIG ?=
./designs/nangate45/gcd/config.mk`, `flow/Makefile:87`), so this ran the
unmodified default flow: `.DEFAULT_GOAL := all` ->
`all: check-yosys check-openroad synth floorplan place cts route finish`
(`flow/Makefile:136,802`).

Wall clock 70s (incl. Docker overhead), tool-reported time 41s (from the
flow's own per-stage timing table).

## Output locations

`RESULTS_DIR = $(WORK_HOME)/results/$(PLATFORM)/$(DESIGN_NICKNAME)/$(FLOW_VARIANT)`
(`flow/scripts/variables.mk:49`), with `FLOW_VARIANT` defaulting to `base`
(`flow/Makefile:97`) -> `flow/results/nangate45/gcd/base/`. Matching
`logs/`, `reports/`, `objects/` directories exist alongside.

By default only `6_final.def` is written (intermediate stages only get
`.odb`). Used the flow's own `make all_defs` target
(`flow/Makefile:840,860`) to materialize a DEF for every stage's `.odb`.

- Placed DEF: `results/nangate45/gcd/base/3_place.odb.def`
- Global route output DEF: `results/nangate45/gcd/base/5_1_grt.odb.def`
- Global route report: `reports/nangate45/gcd/base/5_global_route.rpt`

## Design stats (from DEF headers and `logs/*.log`, not estimated)

| Stage | Components | Signal nets | Special nets | Design area | Utilization |
|---|---|---|---|---|---|
| 1_synth | - | - | - | 627 um^2 | 100% (pre-placement) |
| 3_place (detailed placement) | 606 | 631 | 2 | 672 um^2 | 62.4% |
| 5_1_grt (global route) | 614 | 636 | 2 | 683 um^2 | 63.5% |
| 6_final | 1068 (incl. 454 filler + 46 tap) | 636 | 2 | 683 um^2 | 63% |

Die area, from `3_place.odb.def`: `DIEAREA ( 0 0 ) ( 71510 71510 )` with
`UNITS DISTANCE MICRONS 2000` -> 71510/2000 = 35.755 um/side -> 1278.4 um^2
(square die). Core area is smaller (~1077 um^2, back-calculated from
672 um^2 used at 62.4%) since the die includes margin outside the
placement rows.

Component/net counts extracted via:
```
grep -E "^UNITS DISTANCE|^DIEAREA|^COMPONENTS [0-9]|^PINS [0-9]|^NETS [0-9]|^SPECIALNETS [0-9]" <file>.def
```
Design area/utilization extracted via:
```
grep -riH "design area\|utilization" logs/nangate45/gcd/base/*.log
```

## Why these files matter for the project

- `3_place.odb`/`.def` is the placed-but-unrouted state - this is the
  Phase 4 graph-construction input.
- `5_1_grt.odb` is the global-route result - congestion (Phase 2's ground
  truth) is a global-route-stage concept, computed before detailed
  routing (`5_2_route`) even runs.

## Stage-by-stage summary (plain terms)

- **synth** (`1_synth`): Yosys turns `gcd.v` into a gate-level netlist
  against the Nangate45 library. No physical layout yet.
- **floorplan** (`2_*`): defines the die/core rectangle (sized for the
  55% target utilization in `config.mk`'s `CORE_UTILIZATION`), places
  I/O pins, inserts tap cells (latch-up prevention, not functional
  logic), builds the power/ground grid (PDN).
- **place** (`3_*`): global placement (skip-IO, then with IO), resizing/
  buffering (timing fixes), detailed placement (legalizes every cell
  into non-overlapping row positions).
- **cts** (`4_1_cts`): builds the clock tree (buffers distributing the
  clock to every flip-flop with low skew).
- **route** (`5_*`): global route (`5_1_grt` - coarse per-GCell channel
  assignment; this is where congestion is first computed), detailed
  route (`5_2_route` - literal metal geometry), filler-cell insertion.
- **finish** (`6_*`): density fill (manufacturing-uniformity dummy
  shapes, not functional), final timing/power/IR-drop reports, GDS
  generation, DRC/LVS-adjacent checks.
