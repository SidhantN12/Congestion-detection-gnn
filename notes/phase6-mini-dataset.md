# Phase 6 — Mini-dataset

## Final design (v3) and result

Swept **blockage band center** (0.25/0.50/0.75 fraction of die height) x
**blockage band width** (0.10/0.20/0.30 fraction) x **placement seed**
(1-5) = 45 samples. Layer cap fixed at "tight" (metal2-metal3) and
`PLACE_DENSITY` fixed at 0.70 throughout - both proven, in v1/v2 below,
to not be usable as swept dimensions for this design.

```
45/45 samples succeeded, 0 build failures
per-sample mean congestion:  0.917 - 1.046 (std 0.044)
per-sample max congestion:   1.455 - 1.750 (std 0.084) - genuine overflow in every sample
per-sample frac(ratio>0.9):  0.599 - 0.782 (std 0.049)
Exact-duplicate congestion maps: 0
```

Band width shows a clean, monotonic, physically-sensible effect on
severity (mean congestion 0.930 -> 0.985 -> 1.036 as width goes
0.10 -> 0.20 -> 0.30 - wider blockage, more congestion, exactly as
expected). Band center shows a much weaker effect on *overall* severity
but a very strong effect on *spatial pattern*: pairwise congestion-map
correlation is 0.799 for same-center pairs vs 0.130 for different-center
pairs, and the full 45x45 correlation matrix
(`data/raw/phase6_dataset_summary.png`) shows clean block-diagonal
structure - center=0.25 vs center=0.75 samples are actually *negatively*
correlated, since a blockage at the bottom vs top of the die pushes the
congestion hotspot to opposite ends of the grid. Seed's effect is real
but comparatively subtle (same-seed vs different-seed pairwise
correlation: 0.325 vs 0.346 - not a strong grouping signal on its own,
consistent with seed's role being placement-level noise within an
otherwise-similar configuration, not a dominant driver).

Cached to `data/processed/*_graph.pt` via `src/build_dataset.py` -
graph construction happens once here, not inside any training loop.

## Two false starts, both caught by checking actual numbers, not scripts running successfully

### v1: swept (density, layer-cap tier, seed) - as originally specified

All 45 `make grt` runs succeeded (no errors), but the *gate check itself*
caught a serious problem: 30 of 45 samples (the "mid" metal2-5 and
"tight" metal2-3 layer-cap tiers) produced **bit-for-bit identical**
congestion ground truth across every density/seed combination, despite
placement genuinely differing underneath (confirmed: cell positions
different, ~89% of cells moved between seeds - matching the "full" tier's
own real placement variation).

Traced as far as: each run's own `extract_congestion.py` output correctly
recorded a distinct, correct `source_odb` path per sample (ruling out a
path-resolution bug in the sweep script), and OpenROAD's *own* routing
log showed genuinely different demand numbers at the *first* global_route
pass (e.g. metal2 demand 548 vs 566 between two samples) - but the
*final* `dbGCellGrid` written to the `.odb` (after three incremental
repair stages: repair_design, repair_timing, recover_power) came out
identical for the constrained tiers, while it stayed distinct for the
unconstrained ("full") tier. Manually re-summed `dbGCellGrid`'s own
per-layer capacity from the final odb and found it *didn't match* the
first-pass log numbers either (capacity 2256 vs 935 for metal2 on one
sample) - ruling out a simple "frozen at first pass" explanation too.
Did not fully pin down the exact OpenROAD-internal mechanism (would
need a source-level trace through the incremental-repair/`write_guides`
path); reported as a confirmed, reproducible fact about this pipeline
rather than staying silent about the incomplete root cause.

Fix: dropped layer-cap tier as a swept dimension (fixed at "tight"),
swept **blockage position** instead - reusing Phase 3's independently-
proven lever (an explicit routing obstruction) rather than one that
turned out not to reliably move congestion for this design.

### v2: swept (density, band center, seed)

Same gate check, same kind of finding: within every (band_center, seed)
pair, all 5 `PLACE_DENSITY` values (0.50-0.95) produced **bit-identical**
placement (`cell_x`/`cell_y` arrays exactly equal) and therefore
identical congestion - 90 exact-duplicate pairs out of 990. This time the
cause was findable and confirmed directly (not just inferred): RePlAce's
global placement converges to the same legalized solution regardless of
target density, because this design's utilization is only ~55-65% at
default core sizing - the density target never actually constrains
anything since there's abundant slack either way. Retested at
`CORE_UTILIZATION=70` (still avoiding Phase 3's confirmed failure point at
85-90%) - `make grt` succeeded, but the resulting `.odb` files were
byte-identical again (matching sha1sums in ORFS's own build log for
`PLACE_DENSITY=0.50` vs `0.95` at the same seed) - so this isn't a
core-sizing artifact, it holds even with a tighter core.

Fix: dropped `PLACE_DENSITY` as a swept dimension (fixed at 0.70), swept
**blockage width** instead - the other free parameter of the same proven
lever.

## Commands to reproduce

```bash
# From ~/congestion-gnn/OpenROAD-flow-scripts/flow (WSL):
cp "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project/flow/sweep.sh" .
bash sweep.sh   # ~30 samples/hour; 45 samples takes ~25-30 min

# Build + cache all graphs, then run the gate check (project venv):
source ~/congestion-gnn/.venv/bin/activate
cd "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project"
python src/build_dataset.py
python tests/phase6_dataset_summary.py
```
