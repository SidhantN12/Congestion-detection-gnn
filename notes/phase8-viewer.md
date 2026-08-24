# Phase 8 — Minimal viewer

## What it is

`app/viewer.py`, a Streamlit app: sample selector, predicted vs ground
truth heatmaps on a shared colour scale, a signed difference panel,
Spearman/Kendall metrics (both added to `src/metrics.py`, implemented by
hand rather than via scipy - same reasoning as Phase 7's `spearman_corr`:
small, simple, avoids a new pinned dependency), and a ranked top-20
hotspot table with click-to-zoom (selecting a row shows a zoomed-in
ground-truth/predicted pair around that GCell).

Model and graphs are cached via `@st.cache_resource` /`@st.cache_data`
respectively, per the spec. `predict()` is deliberately **not** cached -
caching it would make the "re-predicts in under a second" gate
meaningless (it'd just be timing a dict lookup on repeat views of the
same sample).

## An important caveat, stated plainly in the app itself

The model loaded is the **Phase 7 checkpoint** - trained via the
single-sample overfit test, not on the full 45-sample Phase 6 dataset
(there is no "train a real model" step in the 8-phase plan; Phase 7 was
explicitly a memorisation sanity check, not model development). The
viewer's sample selector lets you switch to *any* of the 45 samples, and
predictions on samples other than the training one are expected to
generalise imperfectly - confirmed when testing sample switching:
Spearman dropped from 0.97/0.98 (the training sample) to ~0.47-0.50 on
an arbitrary other sample. The app says this explicitly in its own
caption rather than presenting the numbers without context.

## Testing (and one real gap)

Couldn't click through a real browser in this environment, so verified
two other ways:

1. **`streamlit.testing.v1.AppTest`** (`tests/test_viewer_app.py`) -
   Streamlit's own headless testing framework, which actually executes
   `app/viewer.py` and can simulate widget interactions, rather than a
   hand-rolled approximation of the app's logic. Confirmed: the app runs
   with no exceptions on initial load, switching the sample selector
   reruns cleanly with no exceptions, all 4 metrics render, and the
   app's own displayed timing caption is present.
2. **Manual HTTP check** - launched the real server
   (`streamlit run app/viewer.py`), confirmed `/_stcore/health` returns
   `ok` and the root page returns HTTP 200.

**Gap found**: `AppTest`'s dataframe element (`at.dataframe[0]`) only
exposes `key`/`proto`/`root`/`run`/`type`/`value` - no method to simulate
a row-selection click. This is a real limitation of Streamlit's testing
framework (dataframe `on_select` is newer and apparently not yet wired
into `AppTest`'s simulation API), not something in this app to fix.
Verified the click-to-zoom *logic* directly instead: the zoom-window
boundary-clipping math (`max(0, x-pad)` / `min(nx, x+pad+1)`) was
exercised standalone against all four grid corners plus a center point -
every case produces a valid, non-empty slice, so it won't crash
regardless of which hotspot a user clicks. The actual click interaction
itself is unverified by automation; if you want to confirm it visually,
the server is left running at `http://localhost:8501`.

## Gate result

**Re-prediction timing (the actual gate metric): 26.8ms** - the model
forward pass itself, as displayed by the app's own caption, on both the
training sample and after switching to a different one. Comfortably
under the 1-second threshold. (`AppTest`'s simulated rerun took ~1s
total in the test harness, but that includes the testing framework's own
script-re-execution overhead - matplotlib figure generation,
Streamlit's delta-tracking, etc. - not real prediction latency; the
in-app timing caption is what the gate is actually asking about.)

## Commands to reproduce

```bash
source ~/congestion-gnn/.venv/bin/activate
cd "/mnt/c/Users/Sidhant/OneDrive/Documents/Python/Btech Project"

# The checkpoint lives under data/processed/, which is gitignored (like
# the rest of data/processed - regenerable, not tracked) - regenerate it
# first if it's not already there:
python src/train.py data/processed/bc50_bw20_s1_graph.pt --epochs 1500 \
  --save-checkpoint data/processed/phase7_model_checkpoint.pt

# Automated gate check:
python tests/test_viewer_app.py

# Run it for real:
streamlit run app/viewer.py
# then open http://localhost:8501
```

## Angular (deferred, per discussion)

The plan called for Streamlit; a later request asked about an Angular
frontend instead. Decided to ship the Streamlit viewer above first (per
the original plan, lower risk) and treat Angular as a stretch goal
afterward - it would need a separate backend API (FastAPI, per that
discussion) since Angular can't import the Python model directly the way
Streamlit does. Not started; revisit if there's time after Phase 8's
gate is otherwise satisfied.
