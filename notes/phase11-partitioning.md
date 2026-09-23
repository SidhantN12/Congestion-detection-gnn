# Phase 11 (Task 3) — Spatial partitioning with halos and degree capping

## Problem

Phase 10 measured a full-graph training step (forward + backward) at
11.15 KiB/cell. It was killed above 5.5 GiB at 750K cells, and the
extrapolation puts the 1.5M-cell design at 16.1 GiB, over a 12 GB budget.
`src/partition.py` splits one design into subgraphs that each fit a
configurable memory budget.

## What was built (`src/partition.py`)

- **Spatial partitions, not neighbour sampling.** The gcell grid is cut
  into px column strips of about equal *cell count*, then each strip into
  py row blocks of about equal cell count. Every partition is a contiguous
  rectangle of gcells, its **core**, and every gcell is core in exactly one
  partition. Congestion is predicted and supervised on core gcells only.
  Balancing by cell count rather than area matters because the placement
  is clustered: at 1.5M cells, the 16 partitions have 22K-65K core gcells
  but only 246K-282K nodes each.
- **L-hop halo.** CongestionGNN has 3 message-passing layers, so a gcell's
  output depends on every node within 3 hops *upstream* (edges followed
  dst → src). A partition is the 3-hop in-neighbourhood of its core
  gcells, plus every edge among those nodes. This reproduces the
  full-graph outputs exactly: a node at distance d < 3 has all its
  in-edges inside the set, which is all that layer 3-d needs.
- **Degree capping.** Without a cap, the halo passes through each core
  cell's nets and out to *every* pin of those nets. For the pre-CTS clock
  net (150,085 pins at 1.5M cells) and the 60 hub nets, that drags a large
  slice of the die into every partition.
  - Nets with full-graph degree > `degree_cap` (default **512**) are
    therefore not expanded. They keep only pins already inside the
    partition, sampled down to 512 (seeded) if there are more.
  - At 1.5M cells this caps **61 nets**: exactly the hubs and the clock,
    since the synthetic power-law tail stops at 500.
  - A capped net's aggregated embedding becomes an estimate over local
    pins, so outputs within its 2-hop reach are no longer exact. That
    deviation is measured below.
- **Memory budget.** `plan()` estimates a partition's training-step peak
  as `0.170 GiB + 534.7 B × edges`. Those coefficients come from a
  least-squares fit to Phase 10's fwd+bwd peaks at 200K/350K/500K cells
  (predicted/actual 1.007, 0.992, 1.003). Starting from the smallest k
  the whole-graph estimate allows, it refines the k × k grid until the
  largest partition's estimate fits `--budget-gib`.
- **A full-graph reference that fits this laptop.** Checking partitions
  needs full-graph outputs at 1.5M cells, but Phase 10 showed
  `model(data)` doesn't fit there (killed above 5.5 GiB).
  `reference_forward` computes the same maths one relation at a time,
  aggregating edges in chunks: SAGE mean
  `lin_l(mean_j x_j) + lin_r(x_i)`, HeteroConv sum, residual + LayerNorm
  + ReLU. It never materialises an (edges × 64) message tensor. It
  matches `model(data)` **bit-for-bit** (max diff 0.0) on 5K- and
  20K-cell graphs (`tests/test_partition.py`), and at 1.5M cells it
  **peaked at 3.84 GiB in 21.2 s**.
- `scalability.py` gained a generic `run_cmd` (the Phase 10 watchdog,
  factored out; behaviour unchanged). `measure` uses it to run one
  training step per saved partition in a fresh process. The step is
  forward + backward with a Huber loss on core gcells.

**The model has random weights** (seed 0) everywhere in this phase.
No trained checkpoint exists (see Phase 9's notes on the lost data).
The equivalence checks don't depend on the weights. The *magnitude* of
the degree-cap error does: it is measured here for a random model only.

## Tests (`tests/test_partition.py`, ~5 s, 20K-cell graph where `model(data)` is ground truth)

```
[1] n=5000: max |model(data) - reference_forward| = 0.00e+00
[1] n=20000: max |model(data) - reference_forward| = 0.00e+00
[2] 3x3: 9 rectangular partitions, cells per partition 2,132-2,275
[3] halo, uncapped: max |stitched - model(data)| = 0.00e+00 over all 8,836 gcells
[4] no halo: 100.0% of boundary gcells and 16.1% of interior gcells deviate (max err 0.026)
[5] cap=64: 62 nets capped; all 3,250 gcells out of their 2-hop reach exact; 5,586 in reach: max err 3.95e-06 (output std 7.07e-02)
[6] density sums and stored gcell features identical for every core gcell
```

Check [5] also asserts that capped nets keep ≤ cap pins, and that
uncapped nets within 2 hops keep every pin.

## Gate results: 1.5M-cell synthetic graph (3,717,981 nodes, 32,175,356 edges)

Raw data: `notes/results/phase11_partition_1500000.json`.

### Plan: 2 GiB budget → 4×4 grid, 16 partitions

k = 3 was tried first: its partition 0 was estimated at 2.02 GiB, over
budget. Planning took 3.5 s; building and saving all 16 partitions took
4.8 s.

### Per-partition measured training step (fwd+bwd, one fresh process each)

| part | nodes | edges | core gcells | peak (GiB) | step time (s) |
|---|---|---|---|---|---|
| 000 | 255,560 | 2,152,301 | 47,871 | 1.237 | 2.09 |
| 001 | 251,973 | 2,162,011 | 33,534 | 1.198 | 1.87 |
| 002 | 270,429 | 2,244,942 | 50,301 | 1.265 | 1.97 |
| 003 | 281,734 | 2,286,964 | 65,124 | 1.316 | 2.05 |
| 004 | 259,458 | 2,228,201 | 34,905 | 1.250 | 1.98 |
| 005 | 268,822 | 2,298,036 | 24,881 | 1.357 | 2.22 |
| 006 | 275,259 | 2,321,924 | 37,053 | 1.385 | 2.19 |
| 007 | 277,064 | 2,300,030 | 48,151 | 1.349 | 2.13 |
| 008 | 251,533 | 2,190,979 | 32,112 | 1.278 | 1.99 |
| 009 | 254,899 | 2,257,646 | 25,488 | 1.233 | 2.11 |
| 010 | 246,075 | 2,151,715 | 22,464 | 1.214 | 2.10 |
| 011 | 255,788 | 2,178,398 | 36,576 | 1.325 | 3.06 |
| 012 | 279,747 | 2,260,004 | 62,952 | 1.370 | 2.08 |
| 013 | 248,215 | 2,141,864 | 34,648 | 1.219 | 1.72 |
| 014 | 269,995 | 2,239,270 | 43,920 | 1.309 | 1.95 |
| 015 | 273,765 | 2,246,900 | 56,120 | 1.339 | 1.97 |

- The largest partition (006) repeated 3 times gave 1.385, 1.374 and
  1.353 GiB. The planner estimated **1.33 GiB** for it, about 4% under
  the measured median of 1.37.
- **Halo overhead:** across all partitions, nodes total 4,220,316 (1.135x
  the full graph) and edges 35,661,185 (1.108x). Sequential steps summed
  to 33.5 s.

### Before / after

| | before: full graph | after: partitioned (max over partitions) |
|---|---|---|
| **1.5M cells**, training step | killed above 5.5 GiB (Phase 10); **16.1 GiB extrapolated** | **1.385 GiB measured** (16 partitions) |
| **500K cells**, training step | **5.45 GiB measured** (Phase 10) | **1.656 GiB measured**, worst of 3 repeats (4 partitions, `notes/results/phase11_partition_500000.json`) |
| 1.5M cells, largest partition **without** degree cap | — | 639,446 nodes / 3,934,465 edges (vs 281,734 / 2,321,924 with cap 512) |

At 500K, the 2 GiB plan chose 2×2. Measured peaks were 1.40-1.62 GiB
on a single run each, and 1.49-1.66 GiB across repeats of the largest
partition. The planner estimated 1.58 GiB for it.

### Boundary verification against the full-graph reference

Across all 16 partitions, core-gcell outputs were stitched back into one
656,100-gcell prediction and compared with `reference_forward`. 9,684
gcells are *boundary* gcells: 4-adjacent to another partition's core.

| variant | stored gcell features | density recomputed from partition's own cells/pins | outputs bit-identical to reference | boundary gcells within 1e-6 | max abs error | Spearman vs reference (H, V) |
|---|---|---|---|---|---|---|
| **3-hop halo, degree cap 512** | 0 mismatches | 0 mismatches | 64.2% | **100%** | 3.9e-07 | 1.0000, 1.0000 |
| 3-hop halo, no cap | 0 mismatches | 0 mismatches | **100%** | **100%** | **0.0** | 1.0000, 1.0000 |
| **no halo** (core gcells + their cells + their nets) | 0 mismatches | 0 mismatches | 53.2% | **0%** | 0.0435 | 0.9999, 0.9996 |

Reference output std is 0.082. Without a halo, boundary gcells' mean
absolute error is 5.1e-3, against 5.5e-6 for interior gcells.

**Reading this honestly:**
- **Stored and recomputed density features match with or without a
  halo.** That is true by construction here, not something the halo
  achieves: `build_graph` assigns each cell to exactly one gcell (by
  lower-left corner), so a gcell's cell/pin density depends only on cells
  inside that gcell, and those are always in its own partition. Both
  checks pass for all 3 variants; they confirm the partitioner never
  drops a core cell or pin.
- **What the halo protects is the model output.** Without a halo, **0%**
  of boundary gcells match the full graph. With the 3-hop halo and no
  cap, **every** gcell matches **bit-for-bit**.
- **Degree capping is the one approximation.** 35.8% of gcells are within
  2 hops of one of the 61 capped nets and differ from the reference;
  their max error is 3.9e-07, against an output std of 0.082. That is
  for random weights. How large the cap error is for a *trained* model is
  unmeasured.
- The cap is what keeps partitions small. Uncapped, the largest partition
  has 2.27x the nodes and 1.69x the edges, and uncapped verification took
  52 s vs 20 s.

## Unresolved

1. **`train.py`'s pairwise ranking loss still doesn't fit, even per
   partition.** Partitions own 22K-65K core gcells. Phase 10's fit
   (90.3 B/gcell², killed at 8K gcells) extrapolates to about 357 GiB for
   65K gcells. Partitioning fixes the GNN's memory, not this loss; it
   needs sampled pairs (e.g. a fixed number of pairs per gcell). Not done.
2. **No multi-partition training loop.** This phase measures one training
   step per partition. An epoch loop (iterate partitions, accumulate
   gradients or step per partition) is not written.
3. **Degree-cap error on a trained model is unknown**; see above.
   `--degree-cap` is configurable. 512 is a choice that, in the synthetic
   distribution, caps exactly the ≥1000-pin nets.
4. **The memory estimate is linear in edges**, fitted on whole synthetic
   designs. It was within about 4% on these partitions, but it is an
   estimate: the planner guarantees the *estimate* fits, not the measured
   peak. Leave headroom (here, 2 GiB budget vs 1.39 GiB measured).
5. Partitions only change the GCell grid split. The halo recomputes 13.5%
   extra nodes per epoch at 1.5M cells (16 parts); a larger k trades
   memory for more halo overhead. Only k = 4 (1.5M) and k = 2 (500K)
   were measured.

## Commands to reproduce

```bash
cd ~/Documents/Congestion-detection-gnn
source .venv/bin/activate

python tests/test_partition.py                                        # ~5 s

# needs data/processed/synth_1500000.pt (Phase 10: scalability.py prepare --cells 1500000)
python src/partition.py plan      --cells 1500000 --budget-gib 2      # plan + save partitions
python src/partition.py reference --cells 1500000                     # 3.8 GiB, ~21 s
python src/partition.py verify    --cells 1500000                     # 3 variants, ~90 s
python src/partition.py measure   --cells 1500000                     # 16 fresh-process steps

python src/partition.py plan    --cells 500000 --budget-gib 2         # before/after at 500K
python src/partition.py measure --cells 500000
```
