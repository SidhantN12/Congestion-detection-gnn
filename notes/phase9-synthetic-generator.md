# Phase 9 (Task 1) — Synthetic CircuitNet-scale graph generator

## Why, and the constraints it was built under

The next work is engineering for CircuitNet-scale designs (35K-1.5M cells).
On this machine none of the real inputs exist: no OpenROAD, no CircuitNet
download, and `data/raw/` and `data/processed/` are empty (the Phase 4-8
cached graphs and the checkpoint are lost and can't be regenerated without
OpenROAD). `src/synthetic.py` produces graphs with the same structure at
any size, using no real data.

**Host has changed since Phase 0**: work now runs on a MacBook Air (Apple M1,
**8 GB RAM**, 8 cores, macOS 14 / Darwin 23.6), not WSL. The venv packages
match `requirements.txt` exactly (checked: Python 3.11.16, torch 2.9.1,
torch_geometric 2.8.0.post1, numpy 2.4.6, matplotlib 3.11.1). scipy,
torch_cluster and torch_scatter are **not** installed, and none were added.

## What was built

### `src/synthetic.py`: generator, not a second graph builder

`generate_placement(n_cells, seed)` emits a `placement` dict and a
`congestion` dict in **exactly the key/shape schema** that
`extract_placement.py` and `extract_congestion.py` write to `.npz`. Those
dicts go through the **real `build_graph.build_graph`**. So the node
features (cell 5, net 3, gcell 4), the 6 relation types and their
directions all come from the same code that built the gcd graphs, not a
reimplementation that could drift.

The one substitution is the `cell-near-cell` kNN (see below).
`generate_graph(n_cells, seed)` is the one-call wrapper. The CLI prints
and optionally saves stats and the graph.

| Property | How it is generated | Where the default comes from |
|---|---|---|
| Units | site 380x2800 DBU, gcell pitch 4200 DBU | nangate45 as used by the gcd pipeline; 4200 = gcd's 17x17 grid pitch (Phase 5) |
| Cell widths | logic cells 2-8 sites (P = .25/.30/.20/.10/.08/.07), flops 17 sites | **assumed** nangate45-like; no liberty/LEF available to measure |
| Flop fraction | 10% | **assumed** |
| Die | square, total cell area / 0.70 utilization | **assumed** 0.70 (matches Phase 6's `PLACE_DENSITY`) |
| GCell count | die / fixed 4200-DBU pitch, so it grows with area like a real GRT grid | follows from the two rows above |
| Nets | 631/606 x cells | gcd's measured net/cell ratio (Phase 4) |
| Net degree body | P(2..6 pins) = .25/.33/.20/.08/.04 | shaped to the project spec ("most nets 3-4 pins") |
| Net degree tail | 10% of nets, power law P(d) ~ d^-2.5 on [7, 500] | **assumed** exponent |
| Hub nets | max(4, cells/25K) nets, log-uniform in [1000, max(2000, cells/100)] pins | **assumed**, sized to "a small number with thousands" |
| Clock net | one pre-CTS net on every flop | a placement-stage (pre-CTS) DEF has exactly this |
| Net locality | driver at a random point on a Z-order curve, sinks at Laplace(scale = 2 x degree) offsets along it | **assumed**; this makes net bboxes local |
| Hub/clock locality | uniformly random cells across the die | global nets span the die |
| Placement | Gaussian mixture of cells/4000 "module" clusters (lognormal sizes, sigma set for about 1.0 peak utilization per cluster) + 20% uniform background, tails reflected into the die, snapped to site/row grid | **assumed**; not legalized |
| Labels | RUDY-style proxy: each net's bbox wirelength spread over the gcells its bbox covers, per direction; capacity = 90th percentile of demand | **a proxy, not router output**; it exists only so `y_ratio` isn't degenerate for pipeline and scale testing |

**Nothing here is calibrated against real CircuitNet statistics.** No
CircuitNet data is available locally, and I didn't put any CircuitNet
figures into the code that I couldn't check. Every "assumed" row is a
keyword argument of `generate_placement`, so it can be re-calibrated later
if real statistics become available.

### `build_graph.py`: minimal refactor (behaviour unchanged by default)

The inline dense kNN became `knn_dense(x, y, k)`, and `build_graph` gained
an optional `knn_fn=knn_dense` argument. With no argument the code runs
exactly as before, same arithmetic. This was needed because `knn_dense`
builds full n x n `diff_x`, `diff_y` and `dist_sq` float64 matrices: at
35K cells that is 3 x 35,000² x 8 B ≈ 29 GB. That figure is arithmetic from
the code, not a measurement; Phase 10 measures this directly. It is the
first thing in the current pipeline that breaks at scale.

### `synthetic.knn_tiled`: exact kNN in O(n·k) memory

Points are bucketed into a t x t tile grid of about 256 points per tile.
Each tile's queries are checked against a (2r+1)² window of tiles. Any
point outside the window is at least r x min(tile_w, tile_h) away, so a
query whose k-th candidate lies within that bound has its exact answer.
Queries that fail the bound retry with r+1. It uses only numpy and needs
no scipy or torch_cluster.

### `src/memtrack.py`: honest peak memory on macOS

This uses `proc_pid_rusage(RUSAGE_INFO_V4).ri_lifetime_max_phys_footprint`
via `ctypes` (standard library). It is the kernel's high-water mark of
physical footprint, **including compressed and swapped pages**, and is the
same number Activity Monitor shows. `ru_maxrss` / `/usr/bin/time`'s max
RSS was measured to be wrong in both directions here:
- **Undercounts under pressure.** For the 1.5M-cell run, max RSS was
  1,273,839,616 B while peak footprint was 2,286,855,040 B. The difference
  was pages macOS had compressed.
- **Overcounts shared libraries.** For the 35K run, max RSS was
  820,133,888 B vs footprint 407,637,504 B, because RSS counts torch's
  mapped dylibs.
- Sanity check: a 500 MB `np.ones` allocation moved footprint by
  +500 MB; `ru_maxrss` showed 387 MB.

On Linux it falls back to VmHWM / `ru_maxrss`. Each scale runs in its own
process, so the peaks don't contaminate each other.

## How it was verified (`tests/test_synthetic.py`, ~7 s)

- **kNN exactness vs the original `knn_dense`**: sorted neighbour
  distances match exactly (max abs diff **0.0**) on three clustered,
  site-snapped placements (n = 3000, 3000, 6000). These have many exact
  distance ties, so neighbour *sets* can legitimately differ while
  distances can't. On continuous uniform points in a non-square extent,
  the neighbour **sets are identical** for n = 4000 (256 and 16 points per
  tile, the latter forcing many r>1 retries), n = 50 and n = 9 (window
  covers all tiles).
- **Schema**: exactly the 6 expected edge types; feature widths 5/3/4;
  `y_ratio` is (n_gcell, 2); `grid_shape` product = n_gcell; all features
  finite.
- **Invariants** (20K cells): `rev_pin` is exactly `pin` flipped; every
  net has ≥1 pin; no duplicate (net, cell) pins; every cell is in exactly
  one gcell; every gcell has ≥2 adjacency edges.
- **Determinism**: seed 0 twice gives identical graphs; seed 1 gives a
  different one.
- **Model compatibility**: the existing `CongestionGNN` runs a forward pass
  and outputs (8836, 2) with **155,010 params**, the same count Phase 7
  reported.

## Gate results (measured, seed 0, each scale in a fresh process)

Full output: `notes/results/phase9_gate_output.txt`. Raw per-scale stats
including the full degree histograms: `notes/results/phase9_synth_*.json`.

```
=== Nodes ===
                                   35K          100K          500K          1.5M
cell                            35,000       100,000       500,000     1,500,000
net                             36,444       104,125       520,627     1,561,881
gcell                           15,376        44,100       219,024       656,100
total                           86,820       248,225     1,239,651     3,717,981
gcell grid (ny x nx)           124x124       210x210       468x468       810x810

=== Edges ===
cell-pin-net                   167,397       467,991     2,377,958     7,277,098
net-rev_pin-cell               167,397       467,991     2,377,958     7,277,098
cell-in-gcell                   35,000       100,000       500,000     1,500,000
gcell-contains-cell             35,000       100,000       500,000     1,500,000
cell-near-cell                 280,000       800,000     4,000,000    12,000,000
gcell-adjacent-gcell            61,008       175,560       874,224     2,621,160
total                          745,802     2,111,542    10,630,140    32,175,356

=== Net degree (pins per net) ===
max                              3,572        10,176        50,128       150,085
mean                              4.59          4.49          4.57          4.66
median                               3             3             3             3
% nets with 1 pin                 3.3%          3.2%          3.2%          3.2%
% nets with 2 pins               28.5%         28.5%         28.5%         28.4%
% nets with 3-4 pins             48.8%         49.0%         49.0%         48.9%
# nets >= 1000 pins                  5             5            21            61
pins on >=1000-pin nets           5.4%          3.4%          4.5%          6.0%

=== Spatial clustering (cells per gcell) ===
mean                              2.28          2.27          2.28          2.29
var/mean (Poisson=1)              2.59          2.49          3.30          3.37
% empty gcells                   26.3%         25.2%         30.4%         31.2%
max gcell cell_density            7.24          7.60         12.97         11.22
cells with 0 pins                  298           945         4,194        11,462

=== Construction time and memory (macOS ri_lifetime_max_phys_footprint) ===
placement gen (s)                 0.04          0.12          1.05          3.79
build_graph (s)                   1.34          3.62         21.13         65.37
baseline (MiB)                     252           253           251           250
peak (MiB)                         404           405           880          2150
peak - baseline (MiB)              152           153           630          1900
graph tensors (MiB)                 13            37           184           557
```

"Baseline" is the process footprint after imports (torch, PyG), just
before generation starts. "Graph tensors" is the exact sum of `nbytes`
over every tensor in the final `HeteroData`.

Log-log degree plot (PMF and CCDF, all four scales):
`notes/figures/phase9_net_degree_loglog.png`.

**Reading the results:**
- **Heavy tail present at every scale.** The median is 3 pins and 78% of
  nets have 2-4 pins, yet the max degree grows with design size to
  150,085 pins at 1.5M cells (the clock net). 61 nets have ≥1000 pins and
  carry 6.0% of all pin edges. The one-hop neighbourhood of a single hub
  net is therefore a large slice of the graph, which is the batching
  hazard.
- **Spatially clustered, not uniform.** For uniform (Poisson) scatter,
  cells-per-gcell variance/mean would be about 1; it is 2.5-3.4 here.
- **Against the "~3M nodes, 20-30M edges" target for the largest
  designs**, the 1.5M-cell graph comes out slightly over on both: 3.72M
  nodes and 32.2M edges. The extra comes from the fixed 4200-DBU gcell
  pitch (656K gcells) and a mean of 4.66 pins/net. Both are adjustable
  (`utilization`, the degree body/tail); I didn't tune them to hit a
  number.
- **Construction peak at 1.5M is 1.9 GB above baseline, 2.1 GB total**,
  for 557 MiB of final tensors. It fits on this 8 GB machine.

## Unresolved / limitations (stated, not hidden)

1. **Not calibrated to CircuitNet.** Degree shape, clustering strength,
   cell-width mix and hub sizing are assumptions exposed as parameters.
   The synthetic graphs match the *structural properties asked for*;
   whether they match CircuitNet's specific distributions is unmeasured.
2. **Placement is not legalized.** Cells overlap. Max gcell `cell_density`
   reaches 7-13 (site-area occupancy; a legal placement is ≤ ~1 per
   gcell), and 25-31% of gcells are empty. Modern global placers spread
   cells toward a target density, so real placements are probably *less*
   density-clustered than this. The clustering knobs are
   `cluster_peak_util` and `background_frac`. An earlier version also
   piled clipped Gaussian tails onto the die boundary (edge gcells
   averaged 8.6 cells vs 2.1 in the interior); this was caught and fixed
   by reflecting the tails. After the fix, edge gcells average 1.4-1.7 vs
   2.3 interior at 35K.
3. **Degree-distribution gap at 500-1000 pins**, visible in the plot. The
   power-law tail is truncated at 500 and hubs start at 1000, so no nets
   fall in between. This is a generator artifact and harmless for the
   scalability work, but it isn't realistic.
4. **~3.2% of nets have 1 pin.** These come from deduplicating local nets
   whose sinks landed on the driver's cell. The real pipeline also keeps
   1-pin nets (it drops only 0-pin nets), so this is not a schema
   violation. 0.8% of cells have no pins; gcd's tapcells show the same
   pattern (Phase 4).
5. **Labels are a RUDY proxy**, not global-router congestion. They are
   fine for scale and pipeline tests and meaningless for claims about
   model accuracy.
6. The split of `build_graph` time (65 s at 1.5M) between kNN and the rest
   was **not measured**. It's the obvious first thing to profile if
   construction time matters.

## Commands to reproduce

```bash
cd ~/Documents/Congestion-detection-gnn
source .venv/bin/activate

python tests/test_synthetic.py                      # correctness, ~7 s

for n in 35000 100000 500000 1500000; do            # ~2 min total on M1
  python src/synthetic.py --cells $n --stats-json notes/results/phase9_synth_$n.json
done
python tests/phase9_synthetic_gate.py                # table + log-log plot

# a single graph, saved for later use (gitignored location):
python src/synthetic.py --cells 100000 --out data/processed/synth_100k.pt
```
