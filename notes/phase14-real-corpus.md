# Phase 14 — Real multi-design corpus (nangate45, ORFS)

Goal: a corpus of real post-global-route ground truth on four nangate45
designs (aes, ibex, tinyRocket, dynamic_node) x 3 placement variants,
built on a 4 vCPU / 8 GB DigitalOcean droplet with `openroad/orfs:latest`.
This note is written gate by gate. **Gate 1** (one design end to end) is
below; Gates 2-3 are appended when run.

## Design selection: measured, not assumed

The image ships no reference cell counts, so each design was synthesised
alone (`make synth`, sequential) and the count read from
`logs/nangate45/<d>/base/1_synth.json`:

| design | synth std-cells | std-cell area (um^2) | macros | synth time |
|---|---|---|---|---|
| dynamic_node | 10,141 | 21,669 | 0 | 25 s |
| aes | 11,406 | 15,351 | 0 | 117 s |
| ibex | 14,423 | 28,151 | 0 | 154 s |
| tinyRocket | 27,622 | 50,945 | 2 (fakeram45) | 146 s |

None lands cleanly in the scoped 15K-25K band. All four were kept: the
remaining nangate45 designs are either tiny (gcd) or excluded / far
larger (jpeg, swerv, bp_*, ariane, cva6, tpu, mempool). picorv32 was not
measured; it was rejected on diversity grounds (it would make three of
four designs RISC-V cores, so a held-out design would be architecturally
near-identical to a training one). dynamic_node (a NoC router) is kept
for exactly that diversity. **The corpus table reports each design's
actual placed cell count; the real range is ~13K-28K+, not 15K-25K.**

## Environment finding: this image's OpenROAD has no Python

`openroad -help` lists no `-python` flag, and no odb Python module exists
anywhere in the image (`find / -name "_odb*.so" -o -name odb.py` empty,
`python3 -c "import odb"` fails). `src/extract_congestion.py` and
`src/extract_placement.py` cannot run here. The Tcl odb API is complete
(`dbGCellGrid getUsage/getCapacity/getGridX/getXIdx/getYIdx`, 3,903
`odb::` commands), so both extractors were ported:

- `src/extract_congestion.tcl`, `src/extract_placement.tcl`: the same odb
  calls in the same order as the .py versions, dumping raw values as text.
- `src/odb_text_to_npz.py`: the .py versions' arithmetic (direction
  aggregation in tech-layer order, ratio, uniform-pitch check; HPWL, bbox,
  fanout from iterm bbox centres) producing the **identical .npz schema**,
  so `build_graph.py` is unchanged.

The .py extractors are left in place for builds with Python. The port is
validated end to end by the same checks that validated the originals
(OpenROAD heatmap agreement, Phase 5 alignment), below.

Phase 5 check 1 (`getXIdx`/`getYIdx` vs `build_graph.gcell_index`) was an
uncommitted ad-hoc script before; it is now `tests/dump_gcell_index.tcl`
(OpenROAD side) + `tests/verify_gcell_index.py` (imports
`build_graph.gcell_index` itself, so it tests the real function).

## Flow recipe and deviations from stock ORFS

`flow/run_flow.sh <design> <variant> [VAR=value ...]` runs synth, then
floorplan -> global route, then extraction. **Applied uniformly to every
run in this phase (all designs, all variants):**

| setting | effect | why |
|---|---|---|
| `SKIP_INCREMENTAL_REPAIR=1` | `scripts/global_route.tcl` skips the post-GRT `repair_design_helper` + DPL + `global_route -start/-end_incremental`, `repair_timing_helper` + DPL + incremental reroute, and the `OPT_POST_GRT_WNS` repair_timing block | Phase 6 traced its bit-identical congestion maps to these passes; they are also the slow part of GRT on large designs. aes's own `config.mk` already sets it. |
| `FLOORPLAN_DEF=` (empty) | ignores any shipped fixed floorplan DEF (aes has `aes_ng45_fp.def`) | so `CORE_UTILIZATION` actually sizes the core; otherwise a utilisation variant is a no-op |
| `GLOBAL_ROUTE_ARGS=-allow_congestion` | GRT finishes normally if overflow remains, instead of erroring to `5_1_grt-failed.odb` | one code path for every run; no effect when there is no overflow |

**Deviation from stock ORFS, stated plainly:** labels are the first-pass
global route of the placed+CTS'd design, **without** the timing-driven
incremental repair stock ORFS performs after GRT. Still run, as stock:
the `recover_power` incremental reroute pair (a no-op here:
`recover_power_helper` returns immediately with the default
`RECOVER_POWER=0`) and GRT antenna repair (`MAX_REPAIR_ANTENNAS_ITER_GRT`
default 5; it reported 0 net / 0 pin violations on dynamic_node).

Graphs are built from `3_place.odb`; labels come from `5_1_grt.odb`,
which also contains the CTS buffers (dynamic_node: 12,977 placed cells vs
13,613 post-route instances). Same as Phases 1-6.

## Congestion lever (dynamic_node)

Natural congestion (no intervention, run `g1`) is low: max combined
ratio 0.9737, 24/7,225 GCells (0.33%) above 0.9, GRT final report
0 overflow on every layer. So the Phase 3 blockage was applied, with one
change: `flow/add_die_blockage.tcl` gained optional
`BLOCKAGE_MIN_LAYER`/`BLOCKAGE_MAX_LAYER` (defaults unchanged = whole
routing stack, Phase 3/6 behaviour). Blocking every layer in the band
would leave band GCells with capacity 0, which `extract_congestion`
labels ratio 0, i.e. a fully blocked region labelled uncongested. Only
the upper layers are blocked, so band GCells keep metal2/metal3 capacity.

Band fixed at centre 0.5, width 0.2 of die height. Blocked-layer sweep,
**GRT re-run only on an identical placement** (clones of run `g1`;
`flow/blockage_layer_sweep.sh`, report `tests/blockage_layer_sweep.py`).
The clone method was validated: GRT-only clone `x1` and the full fresh
run `g1b` (same settings) produced bit-identical demand and capacity
arrays, and identical `3_5_place_dp` odb hashes (`adf508d7c4`).

| blocked layers | GCells == 1.0 | > 0.9 | > 1.0 | IQR comb | IQR H | IQR V | > 0.9 H / V | ==1.0 H / V |
|---|---|---|---|---|---|---|---|---|
| none | 0 (0.00%) | 24 (0.33%) | 0 | 0.1701 | 0.1849 | 0.1495 | 24 / 0 | 0 / 0 |
| metal6-10 | 0 (0.00%) | 44 (0.61%) | 0 | 0.2250 | 0.2105 | 0.2195 | 34 / 11 | 0 / 0 |
| metal5-10 | 27 (0.37%) | 367 (5.08%) | 0 | 0.2778 | 0.2857 | 0.2276 | 350 / 35 | 26 / 2 |
| **metal4-10** | 282 (3.90%) | 1,215 (16.82%) | 0 | **0.2819** | **0.2913** | **0.2360** | 480 / 1,159 | 49 / 251 |

**Chosen: metal4-10.** Chosen on spread, not on count above 0.9: it has
the widest IQR in the combined map and in each channel, and it is the
only setting giving both channels a non-trivial hotspot set (metal5-10
leaves V with 35 hotspots, making a per-channel hotspot score nearly
meaningless). Its cost is a 3.90% tie mass at exactly 1.0; ~930 GCells
lie between 0.9 and 0.99, so there is a gradient below the ceiling.

Observations:
- **No setting produced overflow.** Demand never exceeds capacity in any
  GCell; the router saturates at 1.0 and detours instead. The plateau at
  1.0 is router behaviour, not an extraction cap.
- **The H/V asymmetry follows which layers are blocked**, not the band's
  orientation alone: metal5-10 is H-dominant (350 vs 35 above 0.9),
  metal4-10 is V-dominant (480 vs 1,159) once metal4 (vertical) is
  removed in the band. Channels are reported separately throughout.
- **Ties are structural, not only at the ceiling.** Demand is integer and
  capacity takes 6 distinct values per direction, so the 7,225-GCell map
  has only 163 (no blockage) to 199 (metal4-10) distinct combined values.
  The largest tie group with *no* blockage is 0.4167 x 426 GCells (5.9%).
  `metrics.spearman_corr` now uses midranks (see "Metric fixes" below).

## Hotspot metric (replaces top-20 hit rate)

`metrics.hotspot_detection` / `hotspot_detection_per_channel`, per
channel, never averaged:
- S = GCells with true ratio >= 0.9; the model's set = its top-|S|
  predictions; precision and recall of that set vs S; ROC-AUC of the
  prediction as a score for membership in S over the whole map.
- Target ties only decide membership in S. Prediction ties straddling the
  top-|S| cutoff get fractional (expected) credit, so the score never
  depends on tie order. AUC is Mann-Whitney with average ranks.
- With k = |S|, **precision == recall by construction**; both reported.
- Wired into `train.evaluate` (per-epoch val + test records) and
  `tests/phase12_baselines_gate.py` (RUDY, density), so model and
  baselines use one definition. `tests/test_metrics.py` checks it against
  brute force (pairwise AUC exact to 1e-12; tie-aware precision 0.7166
  vs 0.7167 over 2,000 random tie orders) and edge cases. PASS.
- Real-data numbers for all three baselines: see "Baselines on the real
  Gate 1 map" below.

## Metric fixes after Gate 1

**`metrics.spearman_corr` now uses midranks** (`metrics._avg_rank`):
Pearson correlation of average ranks, the standard tie-corrected
definition. Previously ties were broken by argsort order, which at 163-199
distinct values per 7,225 GCells makes the value depend on the order of
tied elements. `tests/test_metrics.py` [6] checks it against an
independent O(n^2) textbook midrank implementation on a 7,225-value
vector with 71 distinct targets / 128 distinct predictions (0.884627 ==
0.884627), the exact hand-worked case x=[1,2,2,3], y=[1,2,3,4] ->
3/sqrt(10) = 0.948683 (the old code gave 1.0 here), and invariance to the
order of tied elements. `kendall_tau` was already tau-b (tie-corrected).

**`baselines.CapacityBaseline`**: least squares on
[1, cap_h, cap_v, 1/cap_h, 1/cap_v] only; no netlist, no placement.
Reported next to RUDY and density in `tests/phase12_baselines_gate.py`
and `tests/test_baselines.py` [6]/[7] ([7]: output unchanged when every
non-capacity feature is scrambled and every edge removed). On the Phase
12 *synthetic* graphs capacity is a single constant per channel, so this
baseline is a constant predictor there; its synthetic numbers mean
nothing.

## Baselines on the real Gate 1 map (dynamic_node `g1b`)

Only one real design exists at this point, so density and capacity are
fitted **in-sample** on `g1b` (flatters them; conservative for any claim
that a model beats them). RUDY is unfitted. Spearman shown with midranks
and, for comparison, the old argsort tie-breaking.

| baseline | ch | Spearman | (old) | hotspot precision = recall | ROC-AUC | \|S\| | distinct preds |
|---|---|---|---|---|---|---|---|
| RUDY | H | 0.6789 | 0.6787 | 0.0437 | 0.6942 | 480 | 6,363 |
| RUDY | V | 0.5843 | 0.5846 | 0.0380 | 0.5270 | 1,159 | 6,030 |
| density | H | 0.5950 | 0.5932 | 0.0979 | 0.6637 | 480 | 5,978 |
| density | V | 0.5654 | 0.5647 | 0.1139 | 0.5753 | 1,159 | 5,975 |
| capacity | H | 0.0917 | 0.0874 | 0.1636 | 0.5749 | 480 | 36 |
| capacity | V | 0.0335 | 0.0268 | 0.1516 | 0.4962 | 1,159 | 36 |

Base rates of |S| / 7,225: H 6.6%, V 16.0%. Every baseline is near or
below chance at finding the hotspots, while RUDY and density still
rank-correlate 0.57-0.68 with the map as a whole.

## Finding: the blockage is invisible to the model's inputs

The capacity-only result is weak because **GRT records an obstruction as
usage, not as reduced capacity**. One band GCell (x=40, y=42), usage /
capacity per layer:

| layer | g1 (no blockage) | g1b (metal4-10 blocked) |
|---|---|---|
| metal2 V | 10 / 15 | 12 / 15 |
| metal3 H | 11 / 20 | 14 / 20 |
| metal4 V | 3 / 10 | **10 / 10** |
| metal5 H | 3 / 10 | **10 / 10** |
| metal6 V | 3 / 10 | **10 / 10** |
| metal7 H | 1 / 4 | **4 / 4** |
| metal8 V | 1 / 3 | **3 / 3** |
| metal9 H | 1 / 2 | **2 / 2** |
| metal10 V | 0 / 1 | **1 / 1** |

Capacity is identical between the two runs (band mean H 35.82 / V 40.76
in both); band mean ratio rises H 0.599 -> 0.864, V 0.493 -> 0.938. So:
1. the gcell capacity feature does **not** encode the blockage (this
   corrects the Gate 1 open issue that said it did);
2. placement is bit-identical between g1 and g1b, so **no input feature
   marks the blockage at all**; a model can only find the band through
   cell position, and with the band fixed at 0.5/0.2 in every sample it
   would learn a positional prior that transfers to any held-out design;
3. inside the band the label is mostly blocked tracks counted as demand,
   not netlist routing demand. The 1.0 plateau is largely fully-blocked
   layers.

The blockage recipe is therefore **on hold for the corpus** pending a
decision (see Gate 1 reply).

## Gate 1 result: dynamic_node, run `g1b`

Settings: stock dynamic_node config + the uniform settings above +
`PRE_GLOBAL_ROUTE_TCL=flow/add_die_blockage.tcl`, band 0.5/0.2,
metal4-metal10 blocked.

| quantity | measured |
|---|---|
| placed cells (`3_place.odb`) | 12,977 |
| nets (SIGNAL+CLOCK, >=1 pin) | 15,171 |
| pin edges | 44,254 |
| GCell grid | 85 x 85 (7,225), pitch 5700 DBU (2.85 um), origin (0,0) |
| utilisation after detailed placement | 45.1% (DPL-0009); design area 24,216 um^2 |
| max combined congestion ratio | 1.0000 (H 1.0000, V 1.0000) |
| fraction > 0.9 | 1,215 / 7,225 = 16.82% (H 480, V 1,159) |
| ties at exactly 1.0 | 282 (3.90%; H 49, V 251) |
| extraction vs OpenROAD `gui::dump_heatmap Routing` | 7,225/7,225 GCells present, max abs diff 0.000005, mean 0.000002 (%) |
| grid alignment, `getXIdx/getYIdx` vs `build_graph.gcell_index` | **0 mismatches / 27,102** (12,977 placed cells, 13,613 post-route instances, 512 boundary probes) |
| per-cell containment (`verify_gcell_alignment.py`) | PASS, every cell inside its assigned GCell |
| graph nodes | cell 12,977, net 15,171, gcell 7,225 |
| graph edges | pin 44,254, rev_pin 44,254, in 12,977, contains 12,977, near 103,816, adjacent 28,560 |
| isolated GCell nodes / nets without pins | 0 / 0 (PASS) |
| GCells with no cells | 2,221 / 7,225 (30.7%), not a failure (45% utilisation) |
| graph build (`build_dataset.py`, tiled kNN) | 7.3 s wall, 374 MiB max RSS, 4.6 MB `.pt` |
| **`reference_router_seconds` (5_1_grt step)** | **122 s** |
| floorplan -> global route | 469.2 s |
| synth | 24.9 s |
| total wall clock incl. synth | 494.1 s |

**Which time to compare against inference: 122 s**, recorded as
`reference_router_seconds`. The model takes a placed design as input, so
what it replaces is global routing, not synthesis, floorplan or
placement. The figure is the `5_1_grt` step's elapsed time from ORFS's
own per-stage table, which includes `pin_access` (76 s here),
`global_route` and GRT antenna repair. `flow/run_flow.sh` now writes it
to `logs/<design>_<variant>.timing` for every run. Per-stage breakdown
(`logs/dynamic_node_g1b.log`): floorplan 89 s, global placement 48 s,
resize 22 s, detailed placement 24 s, CTS 11 s, global route 122 s. The
unblocked run `g1` had a 101 s GRT step.

Tiled kNN on real data: `synthetic.knn_tiled` matched an exact chunked
brute-force kNN on all 12,977 dynamic_node cells (identical sorted
neighbour-distance multisets for every cell, 0 self loops), 0.85 s.
`build_dataset.py` now uses it; the dense default would need ~5 GiB here
and ~24 GiB at tinyRocket's size.

## Commands to reproduce (Gate 1)

```bash
cd ~/Congestion-detection-gnn     # repo root on the Linux host
# ORFS work tree: /root/orfs_work (override with ORFS_WORK=...)

# timed run, natural congestion
bash flow/run_flow.sh dynamic_node g1
# timed run, Gate 1 recipe
bash flow/run_flow.sh dynamic_node g1b \
  GLOBAL_ROUTE_ARGS=-allow_congestion PRE_GLOBAL_ROUTE_TCL=/project/flow/add_die_blockage.tcl \
  BLOCKAGE_BAND_CENTER=0.5 BLOCKAGE_BAND_WIDTH=0.2 BLOCKAGE_MIN_LAYER=metal4 BLOCKAGE_MAX_LAYER=metal10
cat logs/dynamic_node_g1b.timing

# blocked-layer sweep (GRT only, on g1's placement), and the report
bash flow/blockage_layer_sweep.sh dynamic_node g1 metal5:x2 metal6:x3
bash flow/blockage_layer_sweep.sh dynamic_node g1 metal4:x1
.venv/bin/python tests/blockage_layer_sweep.py none=data/raw/sweep_dynamic_node__g1_congestion.npz \
  metal6-10=data/raw/sweep_dynamic_node__x3_congestion.npz \
  metal5-10=data/raw/sweep_dynamic_node__x2_congestion.npz \
  metal4-10=data/raw/sweep_dynamic_node__g1b_congestion.npz

# OpenROAD-side checks (heatmap dump + gcell index dump)
docker run --rm -v /root/orfs_work:/work -v "$PWD:/project" -e QT_QPA_PLATFORM=offscreen \
  openroad/orfs:latest bash -c 'source /OpenROAD-flow-scripts/env.sh >/dev/null
    R=/work/results/nangate45/dynamic_node/g1b
    HEATMAP_ODB=$R/5_1_grt.odb HEATMAP_CSV=/project/data/raw/dynamic_node__g1b_gui_heatmap.csv \
      openroad -no_init -no_splash -exit /project/tests/dump_routing_heatmap.tcl
    GRID_ODB=$R/5_1_grt.odb PLACEMENT_TXT=/project/data/raw/sweep_dynamic_node__g1b_placement.txt \
      INDEX_OUT=/project/data/raw/dynamic_node__g1b_gcell_index.tsv \
      openroad -no_init -no_splash -exit /project/tests/dump_gcell_index.tcl'

.venv/bin/python tests/verify_congestion_extraction.py data/raw/sweep_dynamic_node__g1b_congestion.npz \
  data/raw/dynamic_node__g1b_gui_heatmap.csv data/raw/dynamic_node__g1b_comparison.png
.venv/bin/python tests/verify_gcell_index.py data/raw/dynamic_node__g1b_gcell_index.tsv
.venv/bin/python tests/verify_gcell_alignment.py data/raw/sweep_dynamic_node__g1b_placement.npz \
  data/raw/sweep_dynamic_node__g1b_congestion.npz data/raw/dynamic_node__g1b_alignment.png
.venv/bin/python src/build_dataset.py <dir with the two g1b .npz> data/processed
.venv/bin/python tests/verify_graph_construction.py data/processed/dynamic_node__g1b_graph.pt
.venv/bin/python tests/test_metrics.py
```

## Open issues (Gate 1)

1. **Labels are saturating, not overflowing.** No blockage setting tried
   produced demand > capacity. If an overflow regime is wanted later, it
   needs a narrower routing stack (Phase 3's metal2-3 cap), which reopens
   the zero-capacity labelling question.
2. ~~`spearman_corr` tie order~~: fixed, midranks (above).
3. ~~Node features contain the blockage~~: **wrong**, they do not; GRT
   stores obstructions as usage. See "the blockage is invisible".
4. Timing is one run per configuration; no repeat-to-repeat spread was
   measured. The two full dynamic_node runs (`g1`, `g1b`) took 469.6 s
   and 469.2 s floorplan -> GRT.

## Capacity cut via `set_global_routing_region_adjustment`: also stored as usage

To make the band visible, `flow/add_capacity_band.tcl` (PRE_GLOBAL_ROUTE
hook) cuts capacity with `set_global_routing_region_adjustment` instead of
an obstruction (same band 0.5/0.2, metal4-10, adjustment 1.0; GRT-only
clone `c1` of `g1`'s placement, `HOOK=add_capacity_band.tcl bash
flow/blockage_layer_sweep.sh dynamic_node g1 metal4:c1`). **It does not
change capacity either.** From the extracted .npz (band = GCell rows
34-50):

| run | ch | capacity in band | capacity out | demand in band | demand out | ratio in band |
|---|---|---|---|---|---|---|
| g1 (none) | H | 35.82 | 35.97 | 21.45 | 20.25 | 0.599 |
| g1 (none) | V | 40.76 | 40.76 | 20.10 | 20.45 | 0.493 |
| g1b (obstruction) | H | 35.82 | 35.97 | 30.94 | 20.87 | 0.864 |
| g1b (obstruction) | V | 40.76 | 40.76 | 38.20 | 20.55 | 0.938 |
| c1 (region adjustment) | H | 35.82 | 35.97 | 30.59 | 20.74 | 0.854 |
| c1 (region adjustment) | V | 40.76 | 40.76 | 36.79 | 20.52 | 0.903 |

c1's capacity array differs from g1's in 0 GCells; per layer, GCell
(40,42) reads metal4-10 usage == capacity exactly as with the obstruction.
g1 vs g1b capacity arrays are bit-identical too. **So the Gate 1 labels
were positionally memorisable**: the band changes the labels but no input.

Hypothesis, not verified: `dbGCellGrid` stores *nominal* capacity and
counts every capacity reduction as usage. If so, the platform's own layer
adjustments (`fastroute.tcl`: 0.5 on metal2-3, 0.25 on metal4+) are in
every label since Phase 2 too: metal2 capacity 15 is the raw track count
(2.85 um GCell / 0.19 um pitch), and the no-blockage run shows small
constant usage on upper layers (metal8 1/3, metal9 1/2) consistent with
reserved tracks. Phase 2's GUI agreement cannot rule this out (the GUI
reads the same values). Spatially near-uniform, so it inflates absolute
labels more than it misleads ranking.

Agreed fix, **not implemented** (machine ran out of time): snap the band
to the GCell grid (pitch 5700 DBU at every utilisation measured: 85x85 at
CORE_UTILIZATION 40, 65x65 at 70, origin 0), assert that cut layers read
usage == capacity in every band GCell, and remove those layers' usage and
capacity for band GCells in `odb_text_to_npz.py`, keeping the raw arrays
alongside. Then verify capacity drops in band in the .npz before using it.
Band position and width per sample to come from a seeded list.

## Status when the droplet expired

Gate 2 (corpus) and Gate 3 (smoke test) were **not run**. The capacity-cut
recipe was not settled in time: the obstruction (Gate 1) and the region
adjustment both proved invisible in the inputs, and the extraction-side
correction above is the open next step. Measured natural-congestion
reference for that decision: dynamic_node at CORE_UTILIZATION 70 (78.1%
after detailed placement, 65x65) with no blockage: max 0.9722, 72 GCells
(1.70%) > 0.9 (H 62, V 10), IQR 0.165, reference_router_seconds 116.

## Routing-layer cap instead of a band (after the deadline was lifted)

The "Status when the droplet expired" section above was wrong about the
deadline: the six hours were a working budget, not an expiry. Work
continued. The extraction-side band correction was **not** done (it would
make labels differ from what the router reports and give up the Phase 2
heatmap agreement). All artifacts up to this point were archived first:
`/root/phase14_artifacts_20261026.tar.gz`, 152,331,355 bytes, 625 files,
sha256 `ad89fa5edc1b290d5d6a46aa4e8c7ab88eba3eaa3822afc00d90eef7b75f4f9f`.

**The cap must be applied at GRT, not via MAX_ROUTING_LAYER on a GRT-only
re-run.** ORFS calls `set_routing_layers` once, at floorplan
(`scripts/floorplan.tcl`, via FASTROUTE_TCL), and later stages read it
from the odb. `flow/cap_routing_layers.tcl` (PRE_GLOBAL_ROUTE hook,
`CAP_MAX_LAYER`) sets signal metal2-CAP and clock metal4-CAP (metal3-metal4
for a metal4 cap: set_routing_layers rejects min == max). Verified in the
extracted per-layer capacity: every layer above the cap reads 0/0.

### The leftover incremental reroute overwrote the GCell grid

With `SKIP_INCREMENTAL_REPAIR=1` one incremental pair still ran
(`global_route -start_incremental` / `-end_incremental`, the
recover_power block, which runs when OPT_POST_GRT_WNS is 0; its
recover_power call itself is a no-op). With it, the metal2-4 and metal2-5
caps (`k4`, `k5`) extracted **bit-identical demand on metal2, metal3 and
metal4 in all 7,225 GCells**, and less metal2/3 usage than the uncapped
run (57,947 vs 79,273 on metal2): physically backwards. Re-run with
`OPT_POST_GRT_WNS=1` (with SKIP_INCREMENTAL_REPAIR=1 this skips both
incremental blocks: 0 `start_incremental` calls in the log) and
`GLOBAL_ROUTE_ARGS='-allow_congestion -verbose'` (restores the router's
own per-layer table, which `-allow_congestion` alone had replaced):

| run | incremental pass | extracted metal2 / metal3 / metal4 / metal5 usage | router first-pass demand metal2 / metal3 / metal4 / metal5 |
|---|---|---|---|
| k5 | yes | 57,947 / 74,594 / 22,320 / 21,726 | - |
| k4 | yes | 57,947 / 74,594 / 22,320 / - | - |
| k5b | no | 79,768 / 103,900 / 33,388 / 29,039 | 21,821 / 29,306 / 11,068 / 7,313 |
| k4b | no | 81,330 / 111,195 / 35,876 / - | 23,383 / 36,601 / 13,556 / - |

Without the incremental pass the extracted grid follows the router's
first-pass ordering on every layer (k4b > k5b), and only 84-126 GCells
coincide between caps. The router's "demand" and the grid's "usage" are
in different units (edges vs per-GCell tracks, the latter likely
including reserved tracks), so totals are compared by ordering, not
value. Every run before this section (g1, g1b, the band sweep, c1,
nat_u70) had the incremental pass. Whether that changed *their* grids
was tested afterwards and the answer is no for the two runs checked;
see "Which earlier runs the incremental pass affected" below.

Heatmap agreement still holds on a capped first-pass run (k5b):
7,225/7,225 GCells, max abs diff 0.000005.

### Cap sweep (GRT only on g1's placement, no incremental pass, no band)

`EXTRA_MAKE="OPT_POST_GRT_WNS=1 GLOBAL_ROUTE_ARGS='-allow_congestion -verbose'" bash flow/layer_cap_sweep.sh dynamic_node g1 metal6:k6b metal5:k5b metal4:k4b`

| cap | == 1.0 | > 0.9 | > 1.0 | IQR comb | IQR H | IQR V | > 0.9 H / V | == 1.0 H / V | max H / V | distinct |
|---|---|---|---|---|---|---|---|---|---|---|
| metal2-6 | 8 (0.11%) | 105 (1.45%) | 0 | 0.2000 | 0.2312 | 0.1714 | 103 / 2 | 8 / 0 | 1.0000 / 0.9143 | 84 |
| metal2-5 | 78 (1.08%) | 421 (5.83%) | 0 | 0.2267 | 0.2312 | 0.2400 | 109 / 364 | 11 / 69 | 1.0000 / 1.0000 | 84 |
| metal2-4 | 625 (8.65%) | 1,745 (24.15%) | 0 | 0.2810 | 0.2810 | 0.2400 | 1,506 / 627 | 543 / 135 | 1.0000 / 1.0000 | 47 |

Router first pass: no overflow at any cap (metal2-4 total usage 43.03%,
metal2-5 31.31%, metal2-6 25.34%). GRT step on these: 102-114 s.

## Which earlier runs the incremental pass affected

The cap-sweep finding raised the question of whether every pre-fix
measurement (Gate 1, the band sweep, the 78% natural-congestion run)
describes an overwritten grid. Tested directly: clone the run, re-run
GRT only on the identical placement with `OPT_POST_GRT_WNS=1` (0
`start_incremental` calls in the new logs, 1 in each original), extract,
and compare arrays.

| original (incremental pass) | re-run (no incremental pass) | demand / capacity / ratio GCells that differ |
|---|---|---|
| nat_u70 (CORE_UTILIZATION 70, no blockage, full stack) | nat_u70b | 0 / 0 / 0 of 4,225 (bit-identical) |
| g1b (Gate 1: metal4-10 obstruction band, full stack) | g1c | 0 / 0 / 0 of 7,225 (bit-identical) |

**So the overwrite is not universal.** On the full routing stack (with or
without the obstruction band) the leftover incremental reroute leaves
the grid exactly as the first pass wrote it. It changed the grid only
when the stack was capped by `set_routing_layers` (k4 / k5). The
mechanism is not traced at source level.

Status of the earlier measurements:

- **Gate 1 (g1, g1b): not superseded by the incremental-pass finding.**
  g1b is verified bit-identical without the pass; g1 differs from g1b
  only by being an earlier identical run. (Gate 1's labels are still
  superseded as a corpus recipe, for the separate reason above: the
  obstruction is stored as usage, so no input marks it.)
- **Band sweep (blockage layer sweep, c1 region adjustment): not
  individually re-run.** They use the same full stack and hook mechanism
  as g1b, so they are very likely unaffected, but that is inferred, not
  measured. Superseded as a recipe by the metal2-5 cap (k5b and the
  Gate 2 corpus), not as measurements.
- **k4, k5 (capped, with the pass): superseded** by k4b, k5b. Their
  metal2-4 demand is the overwritten grid.
- **nat_u70: stands, verified** (nat_u70b bit-identical).

### Natural congestion remains insufficient after the fix

nat_u70b (dynamic_node, CORE_UTILIZATION 70 = 78.1% after detailed
placement, full metal2-10 stack, no blockage, no incremental pass):
max H / V 0.9722 / 0.9487, 72 GCells (1.70%) combined > 0.9 (H 62,
V 10), 0 at 1.0, 0 above 1.0, IQR combined 0.1646 (H 0.1780, V 0.1500),
188 distinct combined values (H 101, V 117). Router first pass: total
usage 34.00%, 0 overflow. GRT step 110 s. Identical to nat_u70 in every
GCell. With 10 vertical hotspots on 4,225 GCells at 78% utilisation,
natural congestion remains insufficient after the fix, and the metal2-5
cap stays in the recipe.

### Phase 6: probable cause of the identical maps

Phase 6 v1 (`notes/phase6-mini-dataset.md`) found that the two
*constrained* layer-cap tiers (caps set by `set_routing_layers` in a
FASTROUTE_TCL copy, `flow/fastroute_capped_metal3.tcl`) produced
bit-identical final `dbGCellGrid`s across seeds, while the
*unconstrained* "full" tier stayed distinct, and it pointed at the
incremental repair stages. That is exactly the pattern measured here:
the incremental reroute overwrites a capped stack's grid and leaves a
full stack's grid alone. The incremental reroute is therefore the
**probable** cause of Phase 6 v1's identical maps. **Not proven**: Phase 6
ran on gcd, a different ORFS version and a different cap mechanism
(FASTROUTE_TCL rather than a PRE_GLOBAL_ROUTE hook), and was not re-run.
Phase 6 v2's identical *placements* across PLACE_DENSITY are a separate
effect (RePlAce converging on a ~500-cell design) and are not explained
by this.

## Gate 2 recipe

Added to `flow/run_flow.sh`, applied to every corpus run:

| setting | why |
|---|---|
| `OPT_POST_GRT_WNS=1` | with `SKIP_INCREMENTAL_REPAIR=1`, removes the last incremental reroute (recover_power block), which overwrote capped grids |
| `GLOBAL_ROUTE_ARGS='-congestion_report_iter_step 5 -verbose -allow_congestion'` | stock default plus `-allow_congestion`; `-verbose` keeps the router's own per-layer table in every log |
| `PRE_GLOBAL_ROUTE_TCL=/project/flow/cap_routing_layers.tcl CAP_MAX_LAYER=metal5` | signal stack metal2-5, clock metal4-5, whole die, applied at GRT only (placement and CTS see the stock stack) |

Sweep (`bash flow/sweep_corpus.sh`, one command, sequential):
CORE_UTILIZATION default / +15 / +25 with GPL_RANDOM_SEED 1 / 2 / 3;
defaults dynamic_node 40, aes 45, ibex 50, tinyRocket 60. aes ships a
fixed floorplan, not a utilisation; that floorplan implies ~30 (core
229.9 x 229.6 um from its ROW lines vs 15,351 um^2 synth area = 29.1%),
which would make aes much sparser than the other three, so 45 is used.
A failed run steps utilisation back 5 points, at most twice, never to
or below the previous variant's value; a design that still fails is
logged and the sweep continues. The first two dynamic_node maps are
hashed before the rest run. Graphs: `src/build_dataset.py
data/raw/corpus data/processed/corpus`; table: `src/corpus_report.py`.

Unattended-run hardening: every `rm` in `flow/*.sh` uses `${VAR:?}`
form (an unset variable aborts the command instead of widening its
target).

## Gate 2: the corpus (12 runs)

`bash flow/sweep_corpus.sh` (recipe above), 2026-10-26 20:50-22:34,
sequential. **12 of 12 runs OK, no step-back needed, no failures.**
Hash check after the first two dynamic_node variants passed (u40_s1
`3198914e...`, 85x85; u55_s2 `cf0e9a7a...`, 73x73).

Reproducibility: dynamic_node u40_s1 (full flow, GPL seed 1 = the ORFS
default) is bit-identical in demand, capacity and ratio to k5b (GRT-only
metal2-5 re-run on g1's placement), so the full flow and the GRT-only
clone path agree exactly.

Table (`src/corpus_report.py data/raw/corpus logs`): cells and nets are
the placed design (after buffering/resizing), "> 0.9 H/V" per channel,
"== 1.0" and IQR on the combined max(H, V) map, distinct = number of
distinct ratio values (combined / H / V). Times in seconds: synth,
floorplan -> GRT, the 5_1_grt step alone (reference_router_seconds),
total.

| sample | cells | nets | grid | max H / V | > 0.9 H | > 0.9 V | == 1.0 | IQR comb | distinct comb / H / V | synth s | place->GRT s | GRT step s | total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aes__u45_s1 | 14,524 | 15,021 | 65x65 | 1.0000 / 1.0000 | 428 (10.13%) | 1527 (36.14%) | 586 (13.87%) | 0.2292 | 95 / 61 / 41 | 101 | 227 | 52 | 328 |
| aes__u60_s2 | 14,120 | 14,662 | 56x56 | 1.0000 / 1.0000 | 806 (25.70%) | 1449 (46.21%) | 676 (21.56%) | 0.1267 | 94 / 69 / 45 | 98 | 219 | 54 | 317 |
| aes__u70_s3 | 13,826 | 14,388 | 52x52 | 1.0000 / 1.0000 | 1301 (48.11%) | 1832 (67.75%) | 1396 (51.63%) | 0.0800 | 98 / 69 / 49 | 103 | 257 | 84 | 361 |
| dynamic_node__u40_s1 | 12,977 | 15,171 | 85x85 | 1.0000 / 1.0000 | 109 (1.51%) | 364 (5.04%) | 78 (1.08%) | 0.2267 | 84 / 55 / 36 | 22 | 382 | 83 | 404 |
| dynamic_node__u55_s2 | 12,787 | 15,041 | 73x73 | 1.0000 / 1.0000 | 230 (4.32%) | 305 (5.72%) | 81 (1.52%) | 0.1731 | 88 / 58 / 37 | 19 | 402 | 93 | 421 |
| dynamic_node__u65_s3 | 12,703 | 14,987 | 67x67 | 1.0000 / 1.0000 | 149 (3.32%) | 434 (9.67%) | 105 (2.34%) | 0.2000 | 84 / 58 / 34 | 20 | 368 | 97 | 388 |
| ibex__u50_s1 | 14,620 | 16,850 | 83x83 | 1.0000 / 1.0000 | 247 (3.59%) | 675 (9.80%) | 131 (1.90%) | 0.2246 | 86 / 56 / 41 | 138 | 271 | 107 | 408 |
| ibex__u65_s2 | 14,560 | 16,842 | 73x73 | 1.0000 / 1.0000 | 515 (9.66%) | 1042 (19.55%) | 244 (4.58%) | 0.2077 | 91 / 59 / 41 | 136 | 277 | 118 | 413 |
| ibex__u75_s3 | 14,536 | 16,843 | 68x68 | 1.0000 / 1.0000 | 713 (15.42%) | 1246 (26.95%) | 380 (8.22%) | 0.1867 | 97 / 64 / 43 | 126 | 271 | 118 | 397 |
| tinyRocket__u60_s1 | 28,587 | 33,712 | 106x106 | 1.0000 / 1.0000 | 1725 (15.35%) | 2617 (23.29%) | 995 (8.86%) | 0.2000 | 93 / 70 / 33 | 120 | 583 | 185 | 704 |
| tinyRocket__u75_s2 | 28,437 | 33,628 | 95x95 | 1.1333 / 1.0800 | 3302 (36.59%) | 5117 (56.70%) | 1966 (21.78%) | 0.0867 | 82 / 56 / 33 | 139 | 847 | 295 | 985 |
| tinyRocket__u85_s3 | 28,219 | 33,446 | 89x89 | 1.3871 / 1.4800 | 5248 (66.25%) | 6259 (79.02%) | 2244 (28.33%) | 0.0800 | 143 / 94 / 71 | 120 | 893 | 385 | 1012 |

Router first pass (its own per-layer table, `-verbose`): 0 overflow on
10 of 12 runs. **tinyRocket u75_s2 overflows (total 47) and u85_s3
overflows heavily (total 5,872, 87.28% usage)**; their ratios exceed 1.0
(max 1.1333 / 1.0800 and 1.3871 / 1.4800). These are kept: the labels
are what the router reports with `-allow_congestion`.

Checks the brief asks for:

- **Duplicates: 0.** 12 samples, 12 distinct congestion maps (sha256 of
  the ratio array).
- **Failures: none.**
- **Designs that did not congest: none.** Every sample has >= 109 GCells
  > 0.9 in each channel.
- **Do the levers move congestion? Utilisation does, strongly, for three
  of four designs.** Fraction > 0.9 rises monotonically with utilisation
  on aes (H 10.1 -> 25.7 -> 48.1%, V 36.1 -> 46.2 -> 67.8%), ibex (H 3.6
  -> 9.7 -> 15.4%, V 9.8 -> 19.6 -> 27.0%) and tinyRocket (H 15.4 ->
  36.6 -> 66.3%, V 23.3 -> 56.7 -> 79.0%). dynamic_node barely moves and
  not monotonically (H 1.5 -> 4.3 -> 3.3%, V 5.0 -> 5.7 -> 9.7%). Unlike
  Phase 6 on gcd, the placements and maps all differ. **Seed and
  utilisation change together in every variant**, so their effects are
  not separable from this corpus.
- **Distinct values: 82-143 combined per sample (H 55-94, V 33-71)**,
  vs 163-199 on the Gate 1 maps and 188 on the uncapped 78% run, so the
  capped labels are more heavily tied (why the cap reduces the count was
  not investigated). Spearman on this corpus must use midranks
  (`metrics.spearman_corr` since 8c2f0e9); the size of the difference
  from the old argsort-order ranking was not measured on these maps.

**Concern for the user (not acted on): the high-utilisation variants of
aes and tinyRocket saturate.** aes u70_s3 has 51.6% of GCells at exactly
1.0 and IQR 0.080; tinyRocket u85_s3 28.3% at 1.0, 66-79% of GCells
> 0.9 per channel. On those samples the hotspot set S is most of the
map, so threshold hotspot precision/recall is near-trivial there, and
the tie mass at 1.0 limits what Spearman can resolve. A per-design
metal cap (e.g. no cap for aes/tinyRocket) or lower utilisation offsets
for them would spread the corpus more evenly; the recipe was left
uniform as agreed.

Graph cache: `data/processed/corpus/` (12 graphs + manifest.json, 69 MB,
not tracked). Rebuild: `bash flow/sweep_corpus.sh` (runs + graphs +
table), or graphs alone from existing raw files:
`python src/build_dataset.py data/raw/corpus data/processed/corpus`.

## Gate 3: smoke training on the corpus

`python src/train.py --config configs/corpus_smoke.toml` (CPU, 4 epochs,
explicit design split: train dynamic_node + tinyRocket, val ibex, test
aes; 6 / 3 / 3 samples):

start corpus_smoke [82425ad4c8db] device=cpu amp=auto on cpu -> fp32; designs train/val/test = 2/1/1, samples 6/3/3, 6 train units
epoch   0  train loss 0.2504 (huber 0.2371, rank 0.0265)  val spearman 0.0379 (H -0.0581 V 0.1338)  *best*  7.9s
epoch   1  train loss 0.0622 (huber 0.0586, rank 0.0072)  val spearman 0.1726 (H 0.2524 V 0.0928)  *best*  6.8s
epoch   2  train loss 0.0200 (huber 0.0175, rank 0.0050)  val spearman 0.3037 (H 0.3684 V 0.2390)  *best*  6.8s
epoch   3  train loss 0.0223 (huber 0.0202, rank 0.0041)  val spearman 0.4011 (H 0.3931 V 0.4090)  *best*  6.9s
final: best epoch 3 val spearman 0.4011; test spearman 0.3000 (H 0.4377 V 0.1622) on 3 samples

First epoch train loss 0.2504, last (epoch 3) 0.0223; lowest 0.0200 at
epoch 2. Val Spearman rises every epoch (0.0379 -> 0.4011). Test
Spearman 0.3000 (H 0.4377, V 0.1622) on aes, a design never seen in
training. Four epochs is a smoke test, not a result.

Split rejection (`python tests/phase14_split_check.py`):

[1] split from configs/corpus_smoke.toml
    train designs=['dynamic_node', 'tinyRocket'] samples=6 ['u40_s1', 'u55_s2', 'u60_s1', 'u65_s3', 'u75_s2', 'u85_s3']
    val   designs=['ibex'] samples=3 ['u50_s1', 'u65_s2', 'u75_s3']
    test  designs=['aes'] samples=3 ['u45_s1', 'u60_s2', 'u70_s3']
    12 of 12 samples assigned, 0 in more than one split
[2] load_config with ibex in val and test -> ValueError: a design cannot be in both val and test
[3] split_designs with ibex in val and test -> AssertionError: design in both val and test
PASS

## First genuine overflow in the project: tinyRocket at 75% and 85%

**tinyRocket u75_s2 and u85_s3 (first Gate 2 sweep) are the first runs
anywhere in this project, Phases 2-14, where the router reports real
overflow** (router first pass, its own `-verbose` table, metal2-5 cap):

| sample | total usage | overflow H / V / total | max ratio H / V | GCells > 1.0 (combined) |
|---|---|---|---|---|
| tinyRocket u75_s2 | 73.36% | 4 / 3 / 47 | 1.1333 / 1.0800 | 39 of 9,025 (0.43%; H 28, V 13) |
| tinyRocket u85_s3 | 87.28% | 14 / 19 / 5,872 | 1.3871 / 1.4800 | 2,670 of 7,921 (33.71%; H 1,301, V 2,005) |

(Overflow columns are the router's "Max H / Max V / Total" overflow.)
Every earlier run, including the 78% uncapped dynamic_node run and the
metal2-4 cap, routed with 0 overflow. This establishes that these designs
can exceed routing capacity under enough pressure (placement utilisation
plus the metal2-5 cap). Both samples are replaced in the corpus (next
section) because their maps are saturated, not because the overflow is
wrong; the runs, logs and .npz files are kept (`data/raw/
sweep_tinyRocket__u75_s2_*`, `__u85_s3_*`, archive
`phase14_corpus_20261027.tar.gz`).

## Known limitation: dynamic_node and confounded variants

Utilisation barely moves dynamic_node's congestion and not monotonically
(> 0.9 H 1.5 -> 4.3 -> 3.3%, V 5.0 -> 5.7 -> 9.7% at 40 / 55 / 65).
Seed and utilisation vary together in every variant (seed 1 / 2 / 3 with
utilisation step 0 / 1 / 2), so their effects on congestion cannot be
separated from this corpus.

## What the midrank Spearman fix changes on real samples

`python tests/phase14_tie_effect.py <graphs>`: Spearman between a
predictor and the true ratio per channel, midrank (current
`metrics.spearman_corr`) vs the pre-8c2f0e9 argsort ranking, plus the
min..max over 50 random tie-breaks. RUDY is continuous (ties are almost
all on the target side); the capacity-only baseline is fitted on the
Gate 3 training designs (dynamic_node, tinyRocket) and has 12 distinct
predicted values (ties on both sides).

| sample | predictor | ch | distinct target / pred | midrank | old argsort | old - midrank | random tie-break min..max | max abs diff vs midrank |
|---|---|---|---|---|---|---|---|---|
| dynamic_node__u40_s1 | RUDY | H | 55 / 6238 | 0.8517 | 0.8497 | -0.0019 | 0.8501..0.8512 | 0.0019 |
| dynamic_node__u40_s1 | RUDY | V | 36 / 5993 | 0.8426 | 0.8393 | -0.0032 | 0.8394..0.8414 | 0.0032 |
| dynamic_node__u40_s1 | capacity | H | 55 / 12 | 0.0861 | -0.0156 | -0.1017 | 0.0886..0.1115 | 0.1017 |
| dynamic_node__u40_s1 | capacity | V | 36 / 12 | 0.1010 | 0.0443 | -0.0568 | 0.1066..0.1233 | 0.0568 |
| ibex__u65_s2 | RUDY | H | 59 / 5224 | 0.8316 | 0.8307 | -0.0008 | 0.8305..0.8316 | 0.0010 |
| ibex__u65_s2 | RUDY | V | 41 / 5221 | 0.7813 | 0.7801 | -0.0012 | 0.7791..0.7813 | 0.0023 |
| ibex__u65_s2 | capacity | H | 59 / 12 | 0.1150 | 0.0598 | -0.0552 | 0.1078..0.1307 | 0.0552 |
| ibex__u65_s2 | capacity | V | 41 / 12 | 0.1181 | 0.0386 | -0.0795 | 0.1132..0.1399 | 0.0795 |
| aes__u45_s1 | RUDY | H | 61 / 3959 | 0.8548 | 0.8542 | -0.0007 | 0.8539..0.8552 | 0.0009 |
| aes__u45_s1 | RUDY | V | 41 / 3948 | 0.8204 | 0.8176 | -0.0028 | 0.8171..0.8193 | 0.0033 |
| aes__u45_s1 | capacity | H | 61 / 12 | 0.1621 | 0.0771 | -0.0850 | 0.1500..0.1687 | 0.0850 |
| aes__u45_s1 | capacity | V | 41 / 12 | 0.1282 | 0.0634 | -0.0648 | 0.1311..0.1544 | 0.0648 |

- **Continuous predictor (RUDY): the fix changes Spearman by at most
  0.0033** (old minus midrank -0.0007 to -0.0032; random tie-breaks
  span at most 0.0022). Ties in the labels alone, at 36-61 distinct
  values per channel, barely move Spearman when the prediction is
  continuous.
- **Tied predictor (capacity baseline): the old code understated it by
  0.055-0.102**, e.g. dynamic_node H 0.0861 -> -0.0156 (sign flip), and
  random tie-breaks alone move it by up to 0.025. The argsort order is
  not neutral: it was below every random draw in 8 of 12 rows (all six
  capacity rows and dynamic_node's two RUDY rows).
- So the fix was necessary for any tied predictor: the capacity and
  density baselines, and a model that outputs near-constant values early
  in training or on flat regions. For a trained model with continuous
  output the old number would have been within ~0.003.
- Side observation: RUDY reaches 0.78-0.85 Spearman per channel on these
  samples, a strong baseline for Gate 4.
