# Gate 4 — real-corpus training and evaluation

One fixed run on the final corpus. RUDY outperforms the GNN; density-only is close. No test-driven tuning or further corpus sweeps were performed.

## Provenance and training

- Archive SHA-256: `ecc795c836255c9c8f858166ad33b89daa97b27b0733bcf3eaa9b7f29b2626f4`.
- Archive: 322,027,233 bytes; 2,155 entries, including 2,025 regular files. All twelve local graphs and the training manifest matched byte for byte.
- Git was pulled before training: already up to date at `ac6d9fb`.
- Split: dynamic_node + tinyRocket train (six samples), ibex validation (three), aes test (three). Split checks passed.
- Config: `configs/corpus_final.toml`; deterministic CPU fp32, one thread, learning rate 0.001, accumulation 2, ranking weight 0.5, 20,000 ranking pairs, gradient clip 1, maximum 200 epochs, patience 20.
- Completed 100 epochs; best zero-based epoch 79; early stop at epoch 99. Summed epoch wall time 308.944 s (excludes startup/final evaluation). Peak process physical footprint 603,788,480 bytes.
- Checkpoint SHA-256: `987e86d115c1fe3c8a2ecfe2a1e3bea23a68ad1af779cb4c3df1666abe7baf91`. Local deployment: `data/processed/corpus_final_checkpoint.pt`; training state remains in `runs/corpus_final/`.

## Held-out results

Complete required table (all 24 sample/method/channel rows, including MAE, RMSE, recall and Kendall sampling flags): [Markdown](results/gate4_results.md), [CSV](results/gate4_results.csv). The table below is only the per-channel aggregate.

Training curve: [PNG](figures/gate4_training_curve.png), [PDF](figures/gate4_training_curve.pdf).

Unweighted mean over the three aes variants, separately for H and V. Spearman uses midranks. Kendall is exact for every evaluated map (`kendall_sampled: false`). Hotspots use true ratio >= 0.9 and top-|S| predictions with fractional tie credit; precision equals recall by construction. No top-20 hit-rate metric is reported.

| Method | Channel | Spearman | Kendall | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| CongestionGNN | H | 0.7570 | 0.5679 | 0.1927 | 0.1927 | 0.8130 |
| CongestionGNN | V | 0.7308 | 0.5501 | 0.6291 | 0.6291 | 0.8319 |
| RUDY | H | 0.8593 | 0.6803 | 0.3549 | 0.3549 | 0.8758 |
| RUDY | V | 0.8402 | 0.6690 | 0.7144 | 0.7144 | 0.8887 |
| Density-only | H | 0.7805 | 0.5861 | 0.2204 | 0.2204 | 0.8302 |
| Density-only | V | 0.7270 | 0.5448 | 0.6342 | 0.6342 | 0.8311 |
| Capacity-only | H | 0.1096 | 0.0831 | 0.0700 | 0.0700 | 0.5337 |
| Capacity-only | V | 0.0956 | 0.0742 | 0.3261 | 0.3261 | 0.5464 |

Density and capacity least-squares fits use only the six training graphs. RUDY is unfitted. All methods use identical test graphs and metric code.

## Inference timing

One warm-up, then median of 25 forward calls per graph, one CPU thread. Graph construction and disk I/O are excluded. The 122-second router reference is historical, not the runtime of each aes sample. API `reference_router_seconds` is 122 as required. `sample_router_seconds` separately preserves each archived sample-specific duration; the UI explicitly labels the historical reference.

| Design / variant | GNN median ms | Warm-up / timed calls |
|---|---:|---|
| aes__u35_s2 | 148.895 | 1 / 25 |
| aes__u40_s3 | 144.610 | 1 / 25 |
| aes__u45_s1 | 142.921 | 1 / 25 |
| dynamic_node__u40_s1 | 133.731 | 1 / 25 |
| dynamic_node__u55_s2 | 125.748 | 1 / 25 |
| dynamic_node__u65_s3 | 121.057 | 1 / 25 |
| ibex__u50_s1 | 148.746 | 1 / 25 |
| ibex__u65_s2 | 146.387 | 1 / 25 |
| ibex__u75_s3 | 147.371 | 1 / 25 |
| tinyRocket__u60_s1 | 300.954 | 1 / 25 |
| tinyRocket__u63_s2 | 303.129 | 1 / 25 |
| tinyRocket__u66_s3 | 300.662 | 1 / 25 |

## Interpretation and limitations

RUDY leads on both channels across the aggregate ranking and hotspot metrics. Density-only exceeds GNN horizontally and is close vertically. The weak capacity baseline rules out capacity alone as an explanation, but there is no demonstrated GNN advantage over RUDY and no ablation isolating connectivity. The contribution is the validated extraction pipeline, reproducible evaluation, and separately measured memory engineering. Synthetic CPU memory studies do not establish real million-cell accuracy or GPU memory requirements.

Four designs give only one held-out test netlist. Seed and utilisation vary together. tinyRocket variants cover only 60/63/66%; the last has IQR 0.1667 after both allowed step-downs. Quantisation makes IQR discrete. dynamic_node responds non-monotonically to utilisation. No final sample overflows; development overflow at higher tinyRocket pressure is a separate observation.

The earlier 78%-utilisation natural-congestion measurement was invalidated by the incremental-reroute bug. Do not use it quantitatively to justify the cap. The uniform cap and per-design label-quality rule describe the final experiment.

## Integration and reproduction

API startup now uses the real metadata and graph directory when a checkpoint exists, and fails if its metadata is missing. Predictions come from the checkpoint; metrics include per-channel hotspot precision, recall and ROC-AUC. Stub mode remains available without a checkpoint.

```sh
.venv/bin/python scripts/prepare_real_api.py /Users/sidhant/Downloads/phase14_final_20261027.tar.gz
.venv/bin/python scripts/evaluate_real_corpus.py
.venv/bin/python tests/test_api_gate4.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Measured outputs: `notes/results/gate4_evaluation.json`, resolved config, split, training JSONL, and API gate JSON. The report working copy is in `deliverables/`; original Downloads copies are unchanged. The editable poster location and droplet deletion status still need confirmation. Report pagination and table-of-contents fields require review in Word; no document renderer is installed locally.

## Verification completed

- `tests/test_metrics.py`: PASS (midranks, ties, hotspot precision/recall and AUC).
- `tests/test_baselines.py`: PASS (RUDY reference agreement, density/capacity isolation).
- `tests/test_api_gate4.py`: PASS across all twelve real designs, direct checkpoint-output equality, held-out metric agreement, unknown-design 404 and static frontend response.
- New Python files passed compilation and the one-assignment-per-line AST check; `git diff --check` passed.
- Report ZIP/XML integrity passed, and no `[TBD]` results cells remain. Visual pagination has not been checked.

## Gate 4 completion checks (28 October 2026)

Environment: `UV_HTTP_TIMEOUT=180 uv pip install --python .venv/bin/python -r requirements.txt`
checked all eight pinned requirements successfully. `uv pip check` reported all
69 installed packages compatible. The tied-case metric test passed: midrank
Spearman 0.884627 agrees with the independent implementation; order invariance
passed. Split overlap still raises both the validator ValueError and the
splitter AssertionError. No retraining was performed during checklist completion.

`gate4_artifacts.py` renders all 100 epochs from the preserved JSONL, with
training loss and separate H/V validation Spearman and the selected epoch 79.
MAE/RMSE were added by evaluating the same fixed checkpoint and train-fitted
baselines. RUDY's raw demand proxy is not calibrated to the router's ratio:
its strong rankings coexist with worse absolute error. Do not interpret this
scale mismatch as poor hotspot ordering.

### Live browser gate and orientation correction

`tests/gate4_browser.py` launches an isolated headless installed Chrome using
`tests/chrome_gate.py` and the existing WebSocket dependency; no browser download
is needed. Start the API first, then run the gate. Its reference files are
restored from the verified local archive if absent.

The final corpus's dynamic_node/u40_s1 labels exactly equal the archived k5b
labels at float32 precision. This independently checked equality connects the
archived OpenROAD GUI dump to the sample shown in the real UI. No aes GUI dump
exists in the archive: aes is checked for the held-out UI gate, while this
identical dynamic_node reference supplies the requested OpenROAD comparison.

The gate discovered a remaining rendering bug: a negative `drawImage`
destination height extends the rectangle upward without reversing source
pixels. Before the fix, only 309/7,225 cell centres matched OpenROAD at the
correct y-coordinate; all 7,225 matched vertically reversed. The renderer now
uses an explicit translate/scale transform, consistent with its grid-to-canvas
and mouse coordinate helpers. Afterwards all 7,225/7,225 centres match the
reference colour with zero RGB error. An asymmetric two-row regression probe
also passes. The raw reference difference is 4.838709677379427e-8 in ratio units
(0.000005 percentage points when rounded), with no missing GUI cells.

Live browser assertions passed for aes/u35_s2: fixture banner hidden, TEST
(HELD OUT) label, displayed metrics agreeing with the saved evaluation, and
122-second router reference. No browser JavaScript errors were observed.

Evidence:
- [Held-out aes UI](figures/gate4_aes_ui.png)
- [Real dynamic_node truth UI](figures/gate4_dynamic_truth_ui.png)
- [OpenROAD reference comparison](figures/gate4_openroad_reference.png)
- `results/gate4_browser.json` and `results/gate4_browser_before_fix.json`

### UI access

Open http://127.0.0.1:8000 while the server is running. Select an aes variant
for held-out results, then choose Horizontal/Vertical/Max and Prediction/Truth/
Compare. The server was started during checklist completion. To restart it,
use the command above from the repository root. Refresh an already-open tab
to load the corrected renderer.

### Remaining limitations outside the Gate 4 checklist

The original editable poster is still not located; replacement text is supplied.
The updated DOCX has not been visually paginated in Word. Droplet deletion status
is unknown. These do not alter the measured Gate 4 training/evaluation/UI result.
