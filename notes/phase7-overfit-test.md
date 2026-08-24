# Phase 7 — Overfit test

## Architecture (`src/model.py`)

`CongestionGNN`:
- **Type-specific input encoders**: one `Linear` per node type (cell: 5 ->
  64, net: 3 -> 64, gcell: 4 -> 64), projecting each type's own feature
  dimension into a shared 64-d latent space, followed by ReLU.
- **3 layers of relation-specific message passing**: `torch_geometric.nn.
  HeteroConv`, wrapping one `SAGEConv(64, 64)` per edge type (6 relations
  - `cell-pin-net`, `net-rev_pin-cell`, `cell-in-gcell`,
  `gcell-contains-cell`, `cell-near-cell`, `gcell-adjacent-gcell`), summed
  where a node type receives from multiple relations. Each layer adds a
  residual connection and per-type `LayerNorm` before the next.
- **MLP head on gcell nodes only**: `Linear(64,64) -> ReLU -> Linear(64,2)`,
  predicting `[horizontal_ratio, vertical_ratio]` per gcell - the exact
  two values `extract_congestion.py` produces as ground truth.

155,010 total parameters for this design's graph sizes (606-607 cells,
631-632 nets, 289 gcells on the `tight` layer-cap grid).

## Loss (`src/train.py`)

`L = L_reg + lambda * L_rank`, `lambda = 0.5`:
- `L_reg`: `smooth_l1_loss` (Huber) between predicted and actual
  `[horizontal_ratio, vertical_ratio]`.
- `L_rank`: a pairwise hinge ranking loss over all O(n^2) gcell pairs
  within the sample (n=289 here, so ~83k pairs - trivial on CPU): for
  every pair where the true values differ, penalise the model if its
  predicted difference doesn't have the same sign (by at least a margin).
  This directly targets what the gate measures (rank correlation) rather
  than only chasing absolute magnitude.

## Training

Sample: `bc50_bw20_s1` (Phase 6, blockage centered at 50% of die height,
20% width, seed 1) - chosen as a representative, moderately-severe,
non-degenerate case (not the mildest or most extreme in the dataset).
Adam, lr=1e-3, single-sample full-batch gradient descent (no
minibatching - the whole point of this phase is memorising one graph).

```
epoch     0  loss=0.5102  spearman_H=-0.30  spearman_V=0.30   (worse than random - untrained)
epoch   100  loss=0.0217  spearman_H=0.70   spearman_V=0.62
epoch   200  loss=0.0123  spearman_H=0.86   spearman_V=0.88
epoch   300  loss=0.0042  spearman_H=0.93   spearman_V=0.96
Final        loss=0.0013  spearman_H=0.9702 spearman_V=0.9812
```

Training stopped automatically once both channels held >=0.95 for 50
further epochs (confirms it's a stable plateau, not a noisy one-off
crossing).

## Gate result

**PASS**: Spearman correlation 0.9702 (horizontal) and 0.9812 (vertical),
both above the 0.95 threshold. Loss curve and predicted-vs-actual
heatmap pair (plus a signed diff panel): `data/raw/phase7_overfit.png`.
Predicted and ground-truth heatmaps are visually near-identical,
including the horizontal congestion band from the sample's blockage
(rows 6-9); residual errors are small and scattered (max abs diff 0.21
out of a ~1.6 max value), consistent with a rank-correlation-focused
loss rather than pixel-perfect MSE reconstruction.

No debugging of the upstream pipeline was needed - the model reached the
target on the first real training run, which is itself a (limited, single-
sample) point of confidence that Phases 2-6's data is structurally sound:
a model can't rank-correlate 0.97+ against a target built from broken or
randomly-shuffled features.

## Commands to reproduce

```bash
source ~/congestion-gnn/.venv/bin/activate
cd "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project"
python src/train.py data/processed/bc50_bw20_s1_graph.pt --epochs 1500
```
