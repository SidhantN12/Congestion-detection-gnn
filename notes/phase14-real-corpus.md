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
