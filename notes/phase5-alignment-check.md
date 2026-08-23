# Phase 5 — Alignment check

## What was actually at risk

`build_graph.py` takes grid origin/pitch/nx/ny directly from
`extract_congestion.py`'s output, so the *metadata* trivially agrees by
construction - that alone proves nothing. The real risk: `build_graph.py`
assigns each cell to a gcell via its own hand-rolled arithmetic
(`(x - origin) / pitch`, clipped to `[0, n-1]`), independently of
OpenROAD. If that formula didn't exactly match OpenROAD's own internal
`dbGCellGrid.getXIdx()`/`getYIdx()` - e.g. at grid-line boundaries, or for
the oversized last row/column (GRT extends the last GCell to the die
edge, per Phase 2's notes on `getDirectionCongestionMap`) - cells would
silently land in the wrong gcell, corrupting `cell-in-gcell` edges and
the density features without any obvious symptom.

## Check 1: index arithmetic vs. OpenROAD's own function

Loaded `5_1_grt.odb` (baseline) via `openroad -python` and compared, for
every one of its 614 instances, `gcell.getXIdx(x)`/`getYIdx(y)` (the real
OpenROAD function) against `build_graph.py`'s formula reproduced
verbatim. Also checked 6 explicit boundary cases: the grid origin, the
exact last grid line, one DBU past the last grid line, the die's max
corner, one DBU before the second grid line, and exactly on the second
grid line.

**Result: 0 mismatches, all 614 instances + all 6 boundary cases.**

## Check 2: per-cell containment + visual overlay

Independently (pure Python, only reading the two `.npz` files - no
OpenROAD) re-derived each cell's gcell index from its real coordinates,
then checked the cell's coordinates actually fall inside
`[x_grid[xi], x_grid[xi+1]) x [y_grid[yi], y_grid[yi+1])` - the rectangle
its own index claims. **606/606 cells correctly contained** (using the
`blk1` sample - layer cap + die-band blockage from Phase 3, chosen
because its congestion pattern is spatially distinctive enough that a
misalignment would be visually obvious, unlike the fairly uniform
baseline).

Plotted the congestion heatmap, gcell grid lines, and real cell positions
together in the same DBU coordinate space:
`data/raw/phase5_gcell_alignment.png`. Grid lines exactly bound the
heatmap's colored cells (no half-cell or full-cell offset anywhere), cell
positions form the expected standard-cell row pattern within those
boundaries, and the high-congestion band lands at gcell rows 6-10 (y ~
25200-46200 DBU) - matching `add_die_blockage.tcl`'s intended
`y in [0.40, 0.60] x die_height` = `[28604, 42906]` DBU once snapped to
the 4200 DBU grid pitch.

## Not separately re-tested (true by construction)

`build_graph.py`'s gcell node features/labels come from a direct
`.reshape(nx*ny, ...)` of `extract_congestion.py`'s arrays, using the same
`row-major (y, x)` flattening convention as every other grid computation
in the codebase (cell-to-gcell assignment, gcell-gcell adjacency). This
introduces no new arithmetic to distrust - it's a reshape, not a
recomputation - so it wasn't independently re-verified beyond confirming
(Phase 4's gate) that no gcell node ended up isolated or with nonsensical
degree.

## Commands to reproduce

```bash
source ~/congestion-gnn/.venv/bin/activate
cd "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project"
python tests/verify_gcell_alignment.py \
  data/raw/gcd_blk1_placement.npz \
  data/raw/gcd_blk1_congestion.npz \
  data/raw/phase5_gcell_alignment.png
```
