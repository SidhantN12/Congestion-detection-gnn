# Phase 12 (Task 4) — Analytical baselines: RUDY and density-only

## What was built (`src/baselines.py`)

Both baselines are `torch.nn.Module`s whose `forward(data)` takes the same
`HeteroData` as `CongestionGNN` and returns a `(num_gcells, 2)` float32
tensor of [horizontal, vertical] congestion-ratio estimates. So
`baseline(data)` drops in wherever `model(data)` is used, with gcells in
the same row-major `(y, x)` order.

### RUDYBaseline: from the paper, not from memory

Source: P. Spindler, F. M. Johannes, *Fast and Accurate Routing Demand
Estimation for Efficient Routability-driven Placement*, DATE 2007, §2.
The PDF was fetched from the DATE proceedings archive and read directly;
its text was extracted locally with macOS PDFKit, and no tool was added
to the project. The definitions used, as written in the paper:

- **Eq. 1:** `d_n = WA_n / NA_n`, with `NA_n = w_n · h_n` (net area) and
  `WA_n = L_n · p` (wire area). The wire width is `p = p̄ / l` (average
  wire-to-wire pitch / number of routing layers). `L_n` is the estimated
  routed wirelength; the paper uses HPWL, `L_n = w_n + h_n`.
- **Eq. 2:** `R(x,y; x_ll,y_ll,w,h) = 1` if `0 ≤ x−x_ll ≤ w ∧ 0 ≤ y−y_ll ≤ h`,
  else `0`.
- **Eq. 3:** `D(x,y) = Σ_n d_n · R(x,y; x_n,y_n,w_n,h_n)`.
- The paper stresses that RUDY "depends neither on a bin structure nor on
  a certain routing model". So on the gcell grid, the value reported for a
  gcell is the **exact area-average of D over that gcell** (the integral of
  eq. 3 / gcell area), not a rasterised approximation.
- Computing that integral takes O(nets + grid): each net's 1D coverage
  splits into an interval indicator plus end-column/row corrections, and
  every resulting term goes into one 2D difference array
  (`rect_average`). It matches a brute-force per-net, per-gcell overlap
  loop to 1e-14.

**Choices the paper doesn't make, and what I chose:**
1. **Two channels.** The paper's RUDY is a single scalar field. I split
   `L_n` into its horizontal and vertical parts: `d_n^H = w_n·p/NA_n`,
   `d_n^V = h_n·p/NA_n`. Then **D^H + D^V equals the paper's D exactly**
   (tested: max difference 6.4e-14).
2. **Degenerate rectangles.** Eq. 1 divides by `w_n·h_n`, which is
   undefined for nets whose pins share a row or column. The rectangle is
   widened symmetrically to at least one gcell in each dimension, while
   `L_n` keeps the true `w_n + h_n`. A same-row net therefore deposits
   exactly its length `w_n` of horizontal wire and 0 vertical (tested).
   1-pin nets have `L_n = 0` and contribute nothing. Widening can push a
   rectangle past the die edge; that part is dropped.
3. **Units.** Coordinates are in gcell pitches: the graph's normalised
   `x/die_w`, `y/die_h` times the grid shape, which assumes
   `pitch_x == pitch_y` (true in this pipeline). With `p = 1`, D is tracks
   used per gcell, the same unit as the capacity features, so the output
   is `D / capacity` per direction. **The absolute scale depends on the
   true p, which the graph doesn't carry.** Rank metrics are unaffected;
   thresholded metrics like `fraction_above(0.9)` are not calibrated.
4. **Pin positions.** The graph stores each cell's lower-left corner, not
   pin coordinates, so every pin sits at its cell's corner.

### DensityBaseline: no connectivity

- **Features per gcell:** the `cell_density` and `pin_density` input
  features, each also box-averaged over 3×3 and 5×5 gcell windows
  (radii 0, 1, 2), plus a bias. That's 7 features.
- **Model:** least-squares linear map to `[H, V]`, fitted with
  `.fit(graphs)`. It refuses to predict before being fitted.
- **No connectivity:** it reads only gcell features and the grid layout.
  Tested: deleting every `pin`, `rev_pin` and `near` edge leaves its
  output bit-identical.

## Tests (`tests/test_baselines.py`, ~3 s)

```
[1] rect_average vs brute force, 400 rectangles on 7x9: max diff 1.07e-14
[2] hand example: H row 0 = [0.2499..., 0.4999..., 0.2499...], V row 0 = [0.1249..., 0.2500..., 0.1249...], total demand 3.000 = HPWL 3
[3] synthetic design, 3124 nets: max |D^H + D^V - paper D| = 6.39e-14, max |D^H - direct| = 4.17e-14 (D range 0-35.95)
[4] 1-pin net: total demand 0.0; same-row net: finite, H total 2.40 (= w), V total 0.0
[5] density: recovers exact linear weights (max err 2.16e-08); output unchanged with all pin/rev_pin/near edges removed: True
[6] RUDY: shape (1369, 2) torch.float32 == model (1369, 2) torch.float32; ...
[6] density: shape (1369, 2) torch.float32 == model (1369, 2) torch.float32; ...
```

- **[2]** is a hand-worked example: a 2-pin net from (0.5, 0.5) to
  (2.5, 1.5) in gcell units. Here `w = 2`, `h = 1` and `NA = 2`, so
  `d^H = 1` and `d^V = 0.5`. The column overlaps are [0.5, 1, 0.5] and
  the row overlaps [0.5, 0.5, 0]. The ~1e-8 deviations come from the
  graph storing positions as float32.
- **[3]** evaluates eqs. 1-3 directly, one net at a time with Python loops,
  and compares that with the vectorised implementation.

## Gate results (`tests/phase12_baselines_gate.py`)

The density baseline was fitted on three 35K-cell graphs (seeds 1-3,
46,128 gcells) and evaluated on seed-0 graphs it never saw. At 1.5M
cells, `model(data)` doesn't fit this machine, so the model-side shape
comes from `partition.reference_forward` (bit-identical to `model(data)`,
Phase 11). Raw data: `notes/results/phase12_baselines.json`.

| design | baseline | output (= model's) | time | Spearman H / V | Kendall H / V (2000-gcell sample) | frac > 0.9 (label) |
|---|---|---|---|---|---|---|
| 35K | RUDY | (15376, 2) float32 ✓ | 0.04 s | 0.891 / 0.861 | 0.729 / 0.694 | 0.048 (0.186) |
| 35K | density | (15376, 2) float32 ✓ | 0.01 s | 0.890 / 0.899 | 0.699 / 0.715 | 0.119 (0.186) |
| 100K | RUDY | (44100, 2) float32 ✓ | 0.05 s | 0.909 / 0.893 | 0.744 / 0.724 | 0.048 (0.182) |
| 100K | density | (44100, 2) float32 ✓ | 0.01 s | 0.882 / 0.888 | 0.693 / 0.704 | 0.110 (0.182) |
| 1.5M | RUDY | (656100, 2) float32 ✓ | 0.89 s | 0.938 / 0.921 | 0.780 / 0.764 | 0.046 (0.166) |
| 1.5M | density | (656100, 2) float32 ✓ | 0.08 s | 0.908 / 0.915 | 0.721 / 0.735 | 0.167 (0.166) |

Every output is finite. `combined_congestion` and `fraction_above`
accept the outputs reshaped to `(ny, nx, 2)`, and `spearman_corr`
accepts torch tensors directly.

**These accuracy numbers mean nothing about real performance, and RUDY's
are circular.** The synthetic labels are themselves a bounding-box
wire-spreading proxy (Phase 9), close kin to RUDY. They differ in
details (pins at cell centres, gcell-quantised boxes, 1/span spreading),
but a high RUDY correlation here mostly reflects that shared
construction. The density baseline is only as good as the synthetic
labels' dependence on density. Real ground truth is needed before these
numbers go in a results table. RUDY's `frac > 0.9` (0.046-0.048 vs label
0.17-0.19) shows the uncalibrated absolute scale from choice 3.

### What the gate found in `src/metrics.py`

1. **`kendall_tau` cannot run on a CircuitNet-scale design.** It builds
   n×n arrays. Measured in watchdogged subprocesses on random inputs:

   | gcells | peak memory | time |
   |---|---|---|
   | 2,000 | 0.26 GiB | 0.1 s |
   | 4,000 | 0.62 GiB | 0.2 s |
   | 8,000 | 2.31 GiB | 0.9 s |
   | 15,376 (the 35K design) | **killed, over 5.5 GiB** | — |

   The gate therefore reports Kendall on a fixed, seeded 2,000-gcell
   subsample. Fixing it properly needs an O(n log n) implementation
   (Knight's algorithm, mergesort-based inversion counting). **Not done.**
   It would change a shared metric, so it's the user's call.
2. **Tie handling in `spearman_corr` turned out not to matter here.**
   `_rank` breaks ties arbitrarily, which could bias Spearman for outputs
   with many exact ties. Measured: only 0.1-2.5% of baseline predictions
   are tied, since the die-wide clock bbox gives every gcell nonzero
   RUDY. Textbook average-rank Spearman agrees with `spearman_corr` to 3
   decimals in every row above. This could change on real designs with
   large empty regions; the gate's `avg_rank_spearman` is there to
   re-check.

## Unresolved

1. **Accuracy of both baselines is unmeasured** without real congestion
   ground truth, and RUDY-vs-synthetic-label numbers are circular (above).
2. **RUDY's wire width p** (and therefore its absolute scale) isn't
   derivable from the graph. For thresholded metrics it needs either tech
   data or a fitted per-channel scale. Only rank metrics are meaningful
   as-is.
3. **Both baselines need the full gcell grid** (`grid_shape`) and refuse
   a Phase 11 partition subgraph (assertion). They're cheap enough to run
   on the full graph: 0.9 s for RUDY at 1.5M cells.
4. **`kendall_tau` at scale**; see above.
5. Not implemented: pin-RUDY and other RUDY variants (e.g. CircuitNet's
   feature set). Only the paper's RUDY was asked for.

## Commands to reproduce

```bash
cd ~/Documents/Congestion-detection-gnn
source .venv/bin/activate
python tests/test_baselines.py            # ~3 s
python tests/phase12_baselines_gate.py    # ~40 s; uses data/processed/synth_{35000,100000,1500000}.pt
                                          # if present, else regenerates them (seed 0)
```
