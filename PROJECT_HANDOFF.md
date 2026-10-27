# Routing congestion prediction — project handoff

Updated 28 October 2026. Read this first, then `notes/`; phase notes are the
authoritative record. Final-year B.Tech AI & ML project, SIT Pune.

## Standing constraints

- Commit author: Sidhant Nair <rsidhantnair@gmail.com>. No assistance attribution
  or Co-Authored-By trailers in commits.
- Python: one assignment per line; no tuple or multiple assignment in new code.
- Never invent measurements. Distinguish measured results, historical results,
  synthetic experiments, and unperformed work.
- No further corpus sweeps, resplitting or test-driven tuning.

## Current experiment

Twelve real OpenROAD nangate45 graphs in `data/processed/corpus/`, three
variants each of dynamic_node, tinyRocket, ibex and aes. Graph/data/checkpoint
files are gitignored and must travel separately from code.

- Train: dynamic_node + tinyRocket (6 samples).
- Validate: ibex (3). Test: aes (3). No design overlaps.
- Uniform metal2–5 routing cap; `SKIP_INCREMENTAL_REPAIR=1`,
  `OPT_POST_GRT_WNS=1`. Label-quality utilisation choices were fixed before
  training. See `notes/phase14-real-corpus.md` for the final recipe and table.
- `configs/corpus_final.toml`: deterministic CPU fp32, no partitioning,
  Huber plus ranking loss, early stopping on validation Spearman.
- Completed 100 epochs, selected zero-based epoch 79. Do not rerun training
  merely to improve the result. `runs/corpus_final/best.pt` is the selected model;
  `data/processed/corpus_final_checkpoint.pt` is the deployed copy.
- JSONL/config/split are retained under `notes/results/gate4_*`.

Held-out mean Spearman H/V from the measured evaluation:

| Method | H | V |
|---|---:|---:|
| Capacity-only | 0.1096 | 0.0956 |
| Density-only | 0.7805 | 0.7270 |
| RUDY | 0.8593 | 0.8402 |
| CongestionGNN | 0.7570 | 0.7308 |

RUDY wins; density is close to the GNN. Claim the validated extraction pipeline
and measured memory engineering, not predictive superiority or unperformed
cross-technology transfer. The split has only one test design. Seed and
utilisation vary together; tinyRocket's utilisation range is narrow;
dynamic_node's response is non-monotonic. All final samples have zero overflow.

## Evaluation and UI

`scripts/evaluate_real_corpus.py` evaluates the fixed checkpoint and three
baselines fitted only on training graphs. `scripts/gate4_artifacts.py` creates
the training curve and complete per-sample/channel table. Spearman uses
midranks; Kendall flags sampling; hotspots use true ratio >= 0.9 and top-|S|
predictions with fractional credit for cutoff ties. Include precision, recall,
AUC, MAE and RMSE. Never report top-20 hit rate.

Run from the repository root:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and choose an aes variant for held-out results.
The API loads `corpus/api_manifest.json` and the real checkpoint, hiding the
fixture banner. `reference_router_seconds` is the historical measured 122 s;
`sample_router_seconds` preserves the individual flow's time. Inference is a
median of 25 forward calls after warm-up, excluding graph construction/loading.

See `notes/gate4-real-training.md` for completed checks, visual evidence,
reproduction commands and remaining limitations.

## Archive and deliverables

Verified local archive: `/Users/sidhant/Downloads/phase14_final_20261027.tar.gz`,
322,027,233 bytes; SHA-256
`ecc795c836255c9c8f858166ad33b89daa97b27b0733bcf3eaa9b7f29b2626f4`.
Contains 2,155 entries (2,025 regular files). Local corpus matches it exactly.
Droplet deletion status has not been confirmed; do not claim it was deleted.

Report working copy and poster replacement text are in `deliverables/`.
Original report files in Downloads are unchanged. Editable poster location is
unknown; report pagination needs a Word review. The Gate 4 code/data checks do
not imply those publication/layout tasks are complete.

The earlier natural-congestion utilisation measurement is invalidated by the
incremental-reroute bug and cannot justify the cap quantitatively. Historical
synthetic memory results are not real-corpus accuracy or GPU measurements.
