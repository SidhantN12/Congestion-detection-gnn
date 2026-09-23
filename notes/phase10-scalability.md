# Phase 10 (Task 2) — Scalability measurement: where the current approach breaks

## Question

At CircuitNet scale (35K-1.5M cells), which parts of the existing
pipeline stop fitting in memory, and where is the 12 GB crossover for a
full-graph `CongestionGNN` forward pass? All graphs are Phase 9 synthetic
graphs (`src/synthetic.py`, seed 0). No real benchmark data is involved.

## What was built

- **`src/scalability.py`**: the measurement harness.
  - `prepare` saves one synthetic graph per scale to `data/processed/`
    (gitignored).
  - `run --mode M --cells ...` measures each point in a **fresh
    subprocess**, because peak memory is a per-process high-water mark.
  - The parent polls the child's physical footprint every 20 ms and
    SIGKILLs it above `--budget-gib` (5.5 GiB here). That keeps the 8 GB
    laptop usable. A kill is recorded as "exceeded budget", which is
    itself a measured fact: the peak was > 5.5 GiB.
  - Each point is **3 repeats; the median is reported** and all peaks are
    kept (see "run-to-run spread" below for why).
- **Modes measured:**
  - `build_tiled`: generate + `build_graph(knn_fn=knn_tiled)`.
  - `build_dense`: generate + the **original** `build_graph()` with its
    n x n kNN.
  - `fwd_nograd`: inference, `eval()` + `no_grad`.
  - `fwd_grad`: forward with autograd on.
  - `fwd_bwd`: forward + backward of a scalar loss, i.e. a training step
    without the optimizer update.
  - `rank_loss`: `train.pairwise_ranking_loss` forward + backward on n
    gcells.
  - The `fwd_*` modes warm up on a tiny graph first, then load the graph,
    then take the baseline. They also record the peak *after loading*;
    **no run had its peak set by graph loading** (checked automatically).
- **`src/scalability_report.py`** (`python src/scalability.py report`)
  builds the table, the fits, the held-out checks and the chart, using
  only the results JSON.
- Memory metric: `ri_lifetime_max_phys_footprint` (Phase 9's
  `memtrack.py`), which includes compressed/swapped pages. torch used 4
  intra-op threads (its default on this M1).

## Measured results (median of 3 fresh processes; GiB = 2^30 B)

Raw data: `notes/results/phase10_scalability.json`. Report:
`notes/results/phase10_report.txt`.

| cells | nodes | edges | graph tensors (MiB) | build time (s) | build peak (GiB) | inference peak (GiB) | inference time (s) | train fwd peak (GiB) | train fwd+bwd peak (GiB) | fwd+bwd time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 24,902 | 221,262 | 4 | 0.32 | 0.31 | 0.28 | 0.08 | 0.28 | 0.28 | 0.12 |
| 20,000 | 49,661 | 426,186 | 7 | 0.78 | 0.37 | 0.30 | 0.14 | 0.33 | 0.32 | 0.25 |
| 35,000 | 86,820 | 745,802 | 13 | 1.41 | 0.39 | 0.37 | 0.29 | 0.39 | 0.38 | 0.46 |
| 50,000 | 124,264 | 1,072,090 | 19 | 2.18 | 0.43 | 0.40 | 0.41 | 0.45 | 0.48 | 0.90 |
| 100,000 | 248,225 | 2,111,542 | 37 | 3.75 | 0.39 | 0.62 | 0.95 | 0.79 | 1.15 | 1.82 |
| 200,000 | 495,867 | 4,224,540 | 73 | 8.71 | 0.60 | 1.05 | 2.47 | 2.27 | 2.26 | 4.66 |
| 350,000 | 868,103 | 7,408,386 | 129 | 17.08 | 0.81 | 1.88 | 5.50 | 3.89 | 3.89 | 9.20 |
| 500,000 | 1,239,651 | 10,630,140 | 184 | 22.70 | 0.86 | 2.64 | 8.07 | 5.45 | 5.45 | 15.56 |
| 750,000 | 1,859,270 | 15,986,408 | 277 | 33.22 | 1.16 | 3.83 | 11.32 | **>5.5 killed** | **>5.5 killed** | - |
| 1,000,000 | 2,479,498 | 21,316,812 | 370 | 44.20 | 1.48 | 5.01 | 16.72 | not run | not run | - |
| 1,500,000 | 3,717,981 | 32,175,356 | 557 | 70.56 | 2.13 | **>5.5 killed** | - | not run | not run | - |

Peaks are whole-process values and include about 0.25 GiB of
torch/PyG/model baseline. "Not run": `--stop-on-fail` skips larger sizes
once a smaller one exceeded the budget. `.pt` file sizes equal the
graph-tensor sizes to the MiB (in the report file).

### The original O(n²) components (measured until killed)

| component | measured points | fit (a + c·n²) | killed at | largest that fits 12 GiB | at CircuitNet sizes [EXTRAPOLATED] |
|---|---|---|---|---|---|
| `build_graph` dense kNN | 2K-12K cells; 12K = 4.55 GiB | c = 32.1 B/cell², R² = 1.0000, max residual 0.2% | 15,000 cells | **~19,800 cells** | 37 GiB at 35K cells; ~67,000 GiB at 1.5M |
| `train.py` pairwise ranking loss | 1K-6K gcells; 6K = 3.29 GiB | c = 90.3 B/gcell², R² = 0.993, max residual 21.7% (worst at the smallest n) | 8,000 gcells | **~11,800 gcells** | 20 GiB at 15,376 gcells (the 35K design); ~36,000 GiB at 656,100 gcells (the 1.5M design) |

32 B/cell² is consistent with four n x n float64 arrays being alive at
once in `knn_dense` (`diff_x`, `diff_y`, and the temporaries/`dist_sq` of
the squared sum). This is inferred from the code, not profiled.

## Where it breaks: fits and the 12 GB crossover

Linear fits `peak = a + b·cells` use only the measured points with
**≥ 200K cells**. Below that, noise dominates (next section). Each fit
was checked by refitting without its largest measured point and
predicting that point.

| mode | points used | slope | R² | held-out error | **12 GiB crossover** | predicted at 1.5M cells |
|---|---|---|---|---|---|---|
| training step (fwd+bwd) | 200K, 350K, 500K | 11.15 KiB/cell | 0.9998 | +1.4% | **~1.11M cells (~23.7M edges)** | 16.1 GiB |
| training fwd only | 200K, 350K, 500K | 11.12 KiB/cell | 0.9999 | +1.0% | ~1.12M cells | 16.1 GiB |
| inference (no_grad) | 200K-1M (5 pts) | 5.15 KiB/cell | 0.9992 | +1.9% | ~2.4M cells (beyond CircuitNet) | 7.5 GiB |
| graph construction (tiled) | 200K-1.5M (6 pts) | 1.22 KiB/cell | 0.9916 | -6.5% | ~10M cells | 2.1 GiB (measured) |

Edge counts use the measured 21.27 edges/cell at ≥200K cells.

**Largest graph that fits in 12 GB, by component:**
- Original dense graph builder: **~20K cells.** It breaks before the
  smallest CircuitNet design.
- `train.py` ranking loss: **~11.8K gcells.** It breaks on every CircuitNet
  design; the 35K design already has 15.4K gcells.
- Full-graph **training step: ~1.1M cells.** Designs in the upper part of
  the 35K-1.5M range (~1.1M-1.5M cells) don't fit.
- Full-graph **inference: ~2.4M cells.** The whole CircuitNet range fits;
  1.5M is predicted at 7.5 GiB.
- Tiled construction: ~10M cells. Not a bottleneck.

Chart (for the slide): `notes/figures/phase10_memory_vs_design_size.png`.
Measured points are solid; the extrapolations are dashed and start at the
last measured point. The 12 GB line, the 5.5 GiB measurement cap and the
killed runs are all marked.

## How much to trust these numbers

- **Everything above 5.5 GiB is extrapolated**, on an 8 GB machine that
  cannot measure it. The training crossover (1.11M) is a 2.2x
  extrapolation beyond the largest measured training point (500K). It
  rests on a fit that is linear to R² 0.9998, predicts a held-out point
  within 1.4%, and matches the model's structure: every tensor in the
  forward pass is sized by node or edge count, and both are linear in
  cells here. Consistent with that, the inference fit predicts 7.5 GiB at
  1.5M cells, and that run was indeed killed above 5.5 GiB.
- **Run-to-run spread is real**, not measurement error. The high-water
  mark is kernel-exact. The likely cause (a hypothesis, not verified) is PyTorch's multithreaded
  allocation/free timing.
  - Within one batch of 3, the worst spread was 34% (`fwd_bwd` at 50K)
    and 25% at ≥200K (`fwd_grad` at 200K: 1.76 / 2.27 / 2.33 GiB).
  - Between separate batches it was larger: `fwd_grad` at 100K measured
    1.18-1.25 GiB in an exploratory batch and 0.78-0.80 GiB in the final
    one.
  - This is why the fits exclude <200K cells and every point is a median
    of 3. At ≥350K the spread was ≤ 4% for every forward mode.
- **The backward pass doesn't raise the peak.** `fwd_bwd` ≈ `fwd_grad` at
  200K-500K (e.g. 5.448 vs 5.453 GiB at 500K): the peak is at the end of
  the forward pass, where all activations are alive. So training memory
  is governed by the retained forward activations, about 2.2x inference
  (11.15 vs 5.15 KiB/cell).
- **CPU, not GPU.** These are CPU footprints. A CUDA run adds a context
  and caching-allocator overhead and may use different scatter kernels;
  **none of that was measured here** (no GPU available). Read the 12 GB
  line as "this much tensor memory", not as a guarantee for a specific
  GPU.
- **Synthetic graphs.** Memory depends on node and edge counts, and the
  synthetic 1.5M graph has 32.2M edges, slightly above the "20-30M" figure
  for real designs (Phase 9). If real designs have fewer edges per cell,
  the crossover moves right roughly in proportion; edges/cell is the
  number to check first when real data arrives.
- Times are single passes on 4 threads. The large-scale ones may include
  some macOS memory-compression overhead; that wasn't separated out.

## Unresolved

1. **Which tensors dominate** the 5.15 KiB/cell inference cost and the
   11.15 KiB/cell training cost was not profiled; for example, whether
   it's the per-edge message tensors of the `cell-near-cell` relation (12M
   edges at 1.5M cells) or the `HeteroConv` per-relation outputs. This
   should be profiled before choosing a fix (neighbour sampling,
   partitioning, or cheaper relations).
2. The hub nets (up to 150K pins) don't show up as a memory problem in
   full-graph mode. Their cost appears under **neighbour sampling /
   mini-batching**, which this phase didn't measure.
3. The ranking-loss crossover is measured in gcells, not cells. Converting
   it using the synthetic ~2.28 cells/gcell (≈27K cells) depends on the
   gcell pitch assumption from Phase 9.

## Commands to reproduce

```bash
cd ~/Documents/Congestion-detection-gnn
source .venv/bin/activate

python src/scalability.py prepare --cells 10000 20000 35000 50000 100000 200000 350000 500000 750000 1000000 1500000
for m in fwd_nograd fwd_grad fwd_bwd; do
  python src/scalability.py run --mode $m --cells 10000 20000 35000 50000 100000 200000 350000 500000 750000 1000000 1500000 --stop-on-fail
done
python src/scalability.py run --mode build_tiled --cells 10000 20000 35000 50000 100000 200000 350000 500000 750000 1000000 1500000
python src/scalability.py run --mode build_dense --cells 2000 4000 6000 8000 10000 12000 15000 20000 35000 --stop-on-fail
python src/scalability.py run --mode rank_loss --cells 1000 2000 4000 6000 8000 10000 15376 --stop-on-fail
python src/scalability.py report     # table, fits, chart
```

End-to-end runtime wasn't timed as a whole. The per-run times in the
table (x3 repeats, plus graph loading for the forward modes) are the
bulk of it. Use `--budget-gib` to
raise the cap on a machine with more RAM; on a ≥16 GB machine the
1.5M-cell inference and the 750K+ training points would become directly
measurable instead of extrapolated.
