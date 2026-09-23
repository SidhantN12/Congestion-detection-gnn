# Phase 13 (Task 5) — Training infrastructure

## What was built

`src/train.py` keeps the Phase 7 overfit CLI unchanged
(`python src/train.py <graph.pt> ...`; `--help` checked). It also gains a
config-driven trainer:

```bash
python src/train.py --config configs/synthetic_smoke.toml [--set train.epochs=3] [--resume auto|<ckpt>]
```

| Requirement | Implementation |
|---|---|
| **Design-level split** | `src/dataset.py: split_designs` shuffles *designs* (seeded) into train/val/test by fraction, or takes explicit `split.val_designs` / `test_designs` lists. Every sample follows its design; an assertion rejects any design in two splits. The split is written to `<run>/split.json`. |
| **Config system** | `src/config.py`: TOML (stdlib `tomllib`, no new dependency) onto typed dataclasses with defaults. Unknown keys, wrong types and invalid choices are **errors**. `--set section.key=value` overrides are parsed as TOML. The resolved config and its hash go to `<run>/config.resolved.json`. |
| **Checkpoint + resume** | `last.pt` every epoch and `best.pt` on val improvement, written atomically (tmp + `os.replace`). Checkpoints hold model, Adam and GradScaler state, epoch and step counters, early-stopping state, RNG states, and the config and hash. `--resume auto` **refuses a checkpoint whose config hash differs**; only `train.epochs` may change. They also carry the keys `app/viewer.py` reads, so `best.pt` loads in the viewer. |
| **Gradient accumulation** | `train.accum_steps` micro-batches per optimizer step, where a micro-batch is one graph or one partition. Loss is divided by the actual group size, so a short last group at epoch end is averaged correctly. |
| **Mixed precision, guarded** | `train.amp` = auto / off / bf16 / fp16. CUDA: fp16 + GradScaler or bf16. CPU: `auto` → fp32; `fp16` → fp32 with a logged reason; `bf16` is **probed** (forward + backward on a copy of the model) and used only if it runs and stays finite, otherwise fp32. The decision is logged; it never crashes. |
| **Early stopping on val Spearman** | Monitors the mean over val samples of (Spearman_H + Spearman_V)/2. Each sample's partition outputs are stitched back into the full design first (`metrics.spearman_corr`). `early_stopping.patience`, `min_delta`. Not loss. |
| **Structured metrics log** | `<run>/metrics.jsonl`, one JSON object per event: `start` (config hash, device, AMP decision, split, counts, threads), `epoch`, `resume`, `early_stop`, `final` (best epoch, test metrics from `best.pt`). It is appended across resumes. |

Also added:
- **Partitioned training.** With `partition.enabled`, samples are split with
  the Phase 11 planner (`budget_gib`, `degree_cap`) and cached. Loss is
  computed on each partition's core gcells.
- **Sampled ranking loss.** Phase 10 measured the original O(n²)
  `pairwise_ranking_loss` running out of memory beyond about 12K gcells.
  `sampled_ranking_loss` uses the same hinge on `train.rank_pairs` random
  pairs; with 2M pairs on n = 300 it matches the full loss within 0.05%.
- **Synthetic multi-sample designs.** `synthetic.generate_placement` gained
  `variant`: the same cells and netlist, with every cell displaced by
  Gaussian noise (sigma 2 gcells) from an independent RNG. It moved 99.7%
  of cells on a 5K-cell design, and its horizontal labels correlated 0.81
  with variant 0's. That is the within-design similarity a sample-level
  split would leak. **Variant 0 is bit-identical to the old generator**,
  checked against the saved Phase 10 graphs at 35K and 100K cells.

## A reproducibility bug found and fixed during the gate

The first gate run's "resumed" and "uninterrupted" runs **did not match**.
The control run already differed at epoch 0 (val Spearman 0.5800 vs
0.5786), so the resume logic wasn't the cause. Diagnosis, all measured:

- Within one process, forward + backward on the same partition is
  bit-identical across repeated calls.
- Three identical 1-epoch runs **in separate processes** with the default
  4 torch threads: max weight difference **6.0e-05 and 1.9e-04**; val
  Spearman 0.5789 / 0.5788 / 0.5790.
- The same three runs with `torch.set_num_threads(1)`: max weight
  difference **0.0**.

So multithreaded CPU reductions sum in a run-dependent order. The unit
test had passed only because its two runs happened to share a process.
The fix is `train.deterministic` (**default true**): 1 intra-op thread
plus `torch.use_deterministic_algorithms(True, warn_only=True)`. Cost:
epoch time 3.2-3.4 s vs 2.2-2.3 s on the gate config (about 45%). Setting
it to false restores speed and gives up exact reproducibility and exact
resume.

## Tests (`tests/test_train.py`, ~15 s, temp directory only)

```
[1] config: bad keys/types/choices rejected, overrides parsed, hash ignores only train.epochs
[2] split by design: {'train': ['synth000', 'synth001', 'synth003'], 'val': ['synth004'], 'test': ['synth002']}; each of 10 samples follows its design
[3] sampled ranking loss 0.57519 vs full O(n^2) 0.57549 (rel diff 5.17e-04)
[4] 2 epochs + resume to 4 == 4 straight: weights bit-identical, Adam state identical, val metrics identical for all 4 epochs
[5] resume with changed train.lr refused
[6] accum_steps=3 over 39 units: 13 optimizer steps/epoch (= ceil(39/3))
[7] lr=1e-30, patience=2: early stop at epoch 2 after 3 epochs (val spearman stayed 0.063997)
[8] amp=auto: amp=auto on cpu -> fp32
[8] amp=off: amp=off
[8] amp=fp16: amp=fp16 requested on cpu -> fp32 (float16 autocast not used on CPU)
[8] amp=bf16: amp=bf16 on cpu: probe forward+backward OK -> bfloat16 autocast
[8] bf16 epoch on cpu: train loss 0.1205, val spearman 0.7308 (finite)
PASS
```

`test_synthetic.py`, `test_partition.py` and `test_baselines.py` were
rerun after the generator change and all pass.

## Gate: end-to-end run (`configs/synthetic_smoke.toml`)

The config specifies 8 synthetic designs (3K-6K cells) x 3 placements =
24 samples. The split by design is 5 / 2 / 1, giving 15 / 6 / 3 samples.
Partitioning is on, with a deliberately tiny 0.2 GiB budget so these
small designs still get partitioned (mostly 2x2), for **70 training
units**. Other settings: accumulation 2, early-stopping patience 3,
CPU, fp32 (`amp=auto`), deterministic.

The gate ran three separate processes:
1. **Step 1:** `--set train.epochs=3`, i.e. train 3 epochs, checkpoint, and exit.
2. **Step 2:** a new process with `--resume auto`, continuing to the
   config's 6 epochs.
3. **Control:** the same config for 6 epochs, uninterrupted, under
   another name.

```
start synthetic_smoke [1bc20763f3de] device=cpu amp=auto on cpu -> fp32; designs train/val/test = 5/2/1, samples 15/6/3, 70 train units
epoch   0  train loss 0.0870 (huber 0.0834, rank 0.0072)  val spearman 0.5788 (H 0.4606 V 0.6971)  *best*  3.4s
epoch   1  train loss 0.0642 (huber 0.0580, rank 0.0123)  val spearman 0.6913 (H 0.6758 V 0.7069)  *best*  3.2s
epoch   2  train loss 0.0625 (huber 0.0561, rank 0.0128)  val spearman 0.7417 (H 0.7314 V 0.7520)  *best*  3.2s
final: best epoch 2 val spearman 0.7417; test spearman 0.7510 (H 0.7406 V 0.7614) on 3 samples
--- new process ---
resumed from runs/synthetic_smoke/last.pt: next epoch 3, best val spearman 0.7417 @ 2
epoch   3  train loss 0.0542 (huber 0.0445, rank 0.0194)  val spearman 0.7629 (H 0.7530 V 0.7728)  *best*  3.2s
epoch   4  train loss 0.0402 (huber 0.0293, rank 0.0218)  val spearman 0.8148 (H 0.8060 V 0.8236)  *best*  3.2s
epoch   5  train loss 0.0355 (huber 0.0259, rank 0.0193)  val spearman 0.8264 (H 0.8176 V 0.8353)  *best*  3.2s
final: best epoch 5 val spearman 0.8264; test spearman 0.8604 (H 0.8518 V 0.8690) on 3 samples
```

- **Resume check:** the control run's 6 epochs print identical numbers.
  Its final weights are **bit-identical** to the resumed run's, and all 6
  epochs' val metrics are equal.
- **Committed log:** `notes/results/phase13_synthetic_smoke_metrics.jsonl`
  (10 records: start, 3 x epoch, final, resume, 3 x epoch, final),
  alongside the resolved config and the split.
- **Each epoch record contains:** `epoch`, `micro_steps` / `optimizer_steps`
  (280 / 140 after epoch 3, consistent with accumulation 2), `lr`, `train`
  {loss, huber, rank, units}, `val` {spearman, spearman_h, spearman_v,
  huber, per_design, n_samples}, `improved`, `best_val_spearman`,
  `best_epoch`, `epochs_without_improvement`, `train_time_s`,
  `epoch_time_s`, `peak_mem_bytes` (process lifetime peak: 312 MB).

**Val and test Spearman here are against synthetic RUDY-like proxy labels**
(Phase 9), on 1-2 val designs and 1 test design. They show the loop learns
something, not model quality. Val rose every epoch, so early stopping did
not fire in the gate; it is exercised in test [7].

## Unresolved / limitations

1. **Only CPU was run.** The CUDA branches (fp16 + GradScaler, bf16 on
   CUDA, `.to(device)`) are written but **untested**; no GPU here.
   `use_deterministic_algorithms` on CUDA may also need
   `CUBLAS_WORKSPACE_CONFIG` set, which is also untested.
2. **Epoch-level checkpoints only.** A crash mid-epoch loses that epoch's
   progress. Step-level checkpoints would need the within-epoch position
   saved too.
3. **A `final` record is written whenever a process ends** (including step 1
   above, which evaluated test after 3 epochs). A test evaluation per
   process is harmless here but would be wasteful on large test sets.
4. **Units are loaded from disk every step** (partition cache or whole
   graph), with no prefetching. That's fine at this size and not measured
   at benchmark scale.
5. **No learning-rate schedule**, only constant LR (+ weight decay).
6. **`kendall_tau` isn't in the per-epoch log**; it is O(n²) (Phase 12).
7. **bf16 on CPU was only checked to run and stay finite** (one epoch, val
   Spearman 0.7308). Its speed and accuracy vs fp32 weren't measured.

## Commands to reproduce

```bash
cd ~/Documents/Congestion-detection-gnn
source .venv/bin/activate
python tests/test_train.py

rm -rf runs/synthetic_smoke runs/synthetic_smoke_straight
python src/train.py --config configs/synthetic_smoke.toml --set train.epochs=3   # train + checkpoint
python src/train.py --config configs/synthetic_smoke.toml --resume auto          # resume to 6
python src/train.py --config configs/synthetic_smoke.toml --set 'name="synthetic_smoke_straight"'  # control
cat runs/synthetic_smoke/metrics.jsonl
```
