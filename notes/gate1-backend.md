# Gate 1 — Stub fixtures and backend

## Why this gate exists

The real training corpus (4 nangate45 designs, 15K-25K cells each, 3
placement variants, genuine OpenROAD post-route ground truth) is being
generated on another machine and hadn't landed yet. `data/raw/` and
`data/processed/` are empty. This gate builds the FastAPI backend and
its fixtures against stand-in data, so the frontend (Gate 2/3) and the
API contract can be built now and need no changes when the real corpus
and a trained checkpoint arrive (Gate 4) - only which files get
discovered changes.

## Stub fixtures (`scripts/generate_stub_fixtures.py`)

Three synthetic designs via `src/synthetic.py`'s `generate_placement` +
the real `build_graph.build_graph` (same code path the real pipeline
uses, per Phase 9), sized close to the real corpus:

| id | cells | seed | split | grid | gcells |
|---|---|---|---|---|---|
| `synth_train_20k` | 20,000 | 101 | train | 94x94 | 8,836 |
| `synth_val_25k` | 25,000 | 102 | val | 105x105 | 11,025 |
| `synth_test_30k` | 30,000 | 103 | test | 116x115 | 13,340 |

**Truth** is `synthetic.py`'s RUDY-style proxy label (`y_ratio`) -
already understood project-wide as a stand-in for router ground truth,
not router ground truth itself (Phase 9/12).

**Prediction is not a model forward pass.** An untrained model's output
carries no structure, which would make Gate 2's animations unjudgeable:
the bloom-in has nothing to settle on, the predicted/truth wipe shows
noise against structure with no way to tell whether the *animation* or
the *data* is broken, and the hotspot table points at meaningless
regions. Decided (with the user) to instead construct the fixture
prediction as **truth plus spatially-smoothed noise**, with the noise
scale calibrated per channel so Spearman lands in [0.6, 0.8] - enough
correlated structure to exercise every animation meaningfully, not so
much that it could pass for a real result.

- Noise: coarse (~12% resolution) white noise, bilinearly upsampled to
  the full grid via separable `np.interp` - smooth correlated blobs,
  not per-gcell speckle. Independent fields per channel (H, V), so they
  don't move in lockstep.
- Calibration: bisection on the noise multiplier `k`, evaluating
  `spearman_corr(clip(truth + k*noise, 0, None), truth)` against the
  target band.
- Achieved (printed by the script, not claimed as a result anywhere in
  the API or UI):

  ```
  synth_train_20k: Spearman H=0.769 V=0.732
  synth_val_25k:   Spearman H=0.753 V=0.755
  synth_test_30k:  Spearman H=0.785 V=0.735
  ```

**This is a fixture, not a measurement.** Every response that carries a
prediction or a metric also carries `model_source` (`"fixture"` today,
`"checkpoint"` once a trained model is found - see below). The frontend
(Gate 2) must show an unmissable banner whenever `model_source !=
"checkpoint"` and must never present the fixture's Spearman as a model
result; the banner's job is to disappear automatically the moment a real
checkpoint exists, with no other code change.

**Written to `data/stub/`, not `data/processed/`.** The real corpus
lands in `data/processed/` by scp and the Gate 4 trainer globs that
directory - stub and real graphs sharing a location would be a silent
way to end up training on synthetic fixtures. `data/stub/` is gitignored
like `data/raw/` and `data/processed/` (regenerate via the command
below, not expected to be in git).

Each design's graph (`data/stub/<id>_graph.pt`) is the real `HeteroData`
plus one extra attribute, `data["gcell"].y_pred_fixture` - the
calibrated fixture array. `data/stub/manifest.json` holds the
non-tensor metadata (cell count, grid shape, split, die dimensions in
DBU, macro outlines - empty; none exist anywhere in this pipeline, so
the field is honestly empty rather than fabricated).

## Backend (`api/`)

- `api/model_registry.py`: loaded once at startup (FastAPI `lifespan`).
  Looks for `data/processed/*checkpoint*.pt`; if found, loads it and
  sets `model_source = "checkpoint"`. Otherwise builds an **untrained**
  `CongestionGNN` from a stub graph's own node-feature dims and edge
  types (not a hard-coded schema), seeded (`torch.manual_seed(0)`) so
  stub-mode runs are repeatable across restarts, and sets `model_source
  = "fixture"`.
- `api/designs.py` (`DesignStore`): manifest + graph cache (3 small
  designs, trivially cheap to keep in memory), and the derived
  quantities every route needs.
  - **Latency honesty, revised**: an initial version timed one forward
    pass per request with `perf_counter`. That's real but noisy: across
    the 3 stub designs, first-call cold cost dominated the spread enough
    that per-design ranges overlapped (153-199ms vs 201-213ms), which is
    useless for a demo badge that needs one stable number. Fixed via
    `benchmark_latency()`: one untimed warm-up call, then
    `LATENCY_BENCHMARK_RUNS` (25) timed calls, reporting the **median**.
    Computed once per design and cached for the process's lifetime -
    `prediction_maps()` reads the cached figure rather than re-running
    the benchmark per request. The cost is real regardless of whether
    the weights are trained, since it's the same architecture doing the
    same arithmetic on the same graph; what differs by `model_source` is
    only *which values get shown* as the prediction (checkpoint: the
    forward pass's own output; fixture: the pre-baked fixture array) -
    `inference_seconds` is the genuine benchmarked cost either way.
  - **Kendall's tau, conditional exactness, threshold revised**: an
    initial draft set the exact/sampled cutoff at 10,000 gcells. Phase
    12 measured the existing O(n^2) `kendall_tau` fine at 8,000 gcells
    and OOM-killed above ~15,376 - 10,000 was leaving exactness on the
    table for no reason, including for some of the real 15K-25K-cell
    designs (whose grids are projected to land around 6,500-11,000
    gcells). Raised to **14,000**, and `select_kendall()` (a standalone
    function in `designs.py`, independently unit-tested in
    `tests/test_api_gate1.py` since none of the stub grids exceed
    14,000) wraps the exact attempt in `try`/`except MemoryError`,
    falling back to the fixed-seed 2,000-gcell sample if the exact
    computation fails even under the threshold - the `try/except` is the
    real guarantee, the threshold is just there to avoid attempting a
    computation already known to be hopeless. With the new threshold,
    all three stub designs (8,836 / 11,025 / 13,340 gcells) compute
    Kendall exactly.
  - **Hotspots**: ranked by the combined channel (`max(H, V)`) against
    *truth*, top-k. `rank_delta = predicted_rank - true_rank` (full-grid
    ranks, not just within the top-k): positive means the model ranks
    that region less urgent than it truly is, negative means it
    over-ranks it.
- `api/main.py`: the 6 routes from the brief, plus a static mount of
  `web/` (empty until Gate 2; FastAPI is fine mounting an empty existing
  directory). `reference_router_seconds` is `null` on every response
  that carries it - filled from real ORFS flow timings in Gate 4, never
  invented before then.

### Endpoints

```
GET /designs
GET /designs/{id}
GET /designs/{id}/prediction
GET /designs/{id}/truth
GET /designs/{id}/hotspots?k=20
GET /designs/{id}/metrics
```

All verified against the real running server (`uvicorn api.main:app`),
not only the test client - see commands below.

## Gate result (`tests/test_api_gate1.py`, via FastAPI's `TestClient`)

Every endpoint checked for every stub design: correct shapes, correct
value ranges, 404 on an unknown design id, `kendall_sampled` matching
the 14,000-gcell threshold (plus a standalone check of `select_kendall`
exercising the sampled-above-threshold and MemoryError-fallback paths
directly, since no stub grid is large enough to hit either otherwise),
hotspot ranks monotonic and starting at 1.

**Measured inference latency per design** (the gate's actual ask - the
median of 25 timed forward passes through the loaded model after a
warm-up call, on an M1 MacBook Air, CPU):

```
design                  gcells    latency_ms  spearman_h  spearman_v  kendall_sampled
synth_train_20k           8836        141.70       0.769       0.732            False
synth_val_25k             11025        187.74       0.753       0.755            False
synth_test_30k            13340        217.23       0.785       0.735            False
```

(`spearman_h`/`spearman_v` here are the fixture's own calibrated values,
reported for the gate's own record-keeping - not something the UI will
present as a model result, per the `model_source` discipline above.)

## Dependencies added

`fastapi==0.162.0`, `uvicorn==0.53.0`, `httpx==0.28.1` (the last only for
`fastapi.testclient.TestClient` in the gate script). Pinned in
`requirements.txt`, installed via `uv pip install`.

## Unresolved / limitations

1. **Fixture calibration is a single global target band** ([0.6, 0.8])
   applied independently per channel per design: it isn't tuned to look
   like any particular kind of real model error (e.g. systematic
   under-prediction at hotspots), just to have plausible rank structure.
   Revisit if Gate 2/3 screenshots reveal the fixture looking
   unconvincing in some specific way.
2. **`httpx`/`starlette.testclient` prints a deprecation warning**
   (`install httpx2 instead`) on this httpx/starlette pairing. Cosmetic,
   doesn't affect the gate; not chased further since it's a test-only
   dependency.
3. **die dimensions are raw DBU**, not microns - the synthetic
   placement path doesn't carry a `dbu_per_micron` figure the way
   `extract_congestion.py`'s real output does (Phase 2). Fine for Gate 1
   (die outlines just need consistent units for layout math); worth
   converting once Gate 4's real designs are in.
4. **Macro outlines are always `[]`** - no macro extraction exists
   anywhere in this pipeline yet, real or synthetic. Honestly empty, not
   a Gate 1 gap to fix; the frontend must handle an empty list rather
   than assuming at least one macro.

## Commands to reproduce

```bash
cd ~/Documents/Congestion-detection-gnn
source .venv/bin/activate

python scripts/generate_stub_fixtures.py     # writes data/stub/*
python tests/test_api_gate1.py               # gate check + latency table

uvicorn api.main:app --reload --port 8000    # run it for real
curl http://127.0.0.1:8000/designs
```
