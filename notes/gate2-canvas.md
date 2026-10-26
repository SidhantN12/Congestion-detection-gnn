# Gate 2 — Canvas and core animations

## What was built

`web/` — plain HTML/CSS/JS, served as static files by the FastAPI app
(`api/main.py`'s existing `StaticFiles` mount at `/`, from Gate 1). No
build step: `index.html` loads `js/app.js` as a native ES module
(`<script type="module">`), which `import`s `js/renderer.js` and
`js/colormap.js` by relative path - the browser resolves these directly,
no bundler, no npm.

- `web/index.html` - structure: fixture banner, top bar with the
  inference/router latency badge, canvas pane (die rendering + a hint
  line for the keyboard shortcuts), right side panel (design selector,
  channel toggle, mode toggle, legend, metrics, ranked hotspot table).
- `web/css/style.css` - dark theme, responsive down to narrow widths
  (side panel drops below the canvas under 860px).
- `web/js/colormap.js` - pure colour functions, no DOM/canvas
  dependency: `congestionColor(value, domainMax)` (sequential cool-blue
  -> cyan -> amber -> red at the 1.0 overflow threshold -> a darker red
  beyond it, hand-picked stops rather than a library rainbow colormap)
  and `diffColor(diffValue, domainMaxAbs)` (diverging blue/neutral/red
  for Difference mode). Also exports CSS gradient strings for the
  legend, built by sampling the *same* functions the canvas uses, so the
  legend can never visually drift from what's actually drawn.
- `web/js/renderer.js` - `GridRenderer`: owns the visible canvas and one
  named offscreen buffer per layer (`"prediction"`, `"truth"`,
  `"difference"`), each exactly `nx x ny` pixels (one pixel per gcell).
  Drawing is always "write an `ImageData` buffer, then `drawImage` it
  scaled onto the visible canvas" - this decouples per-frame cost from
  grid size entirely (a 300x300 grid costs the same draw call as 90x90),
  which is what makes the bloom-in animation (every gcell recoloured
  every frame) and zoom/pan affordable at any of this project's grid
  sizes without special-casing a dimension anywhere.
- `web/js/app.js` - state, data loading, the three animations, UI
  wiring.

## Visual design

Dark background throughout. The congestion ramp is deep blue
(uncongested) -> cyan -> amber -> red at the 1.0 overflow threshold ->
a darker, more saturated red beyond it - hand-picked piecewise-linear
RGB stops (`colormap.js`), not a library rainbow colormap, per the
brief's "perceptually reasonable, not raw rainbow." The legend draws the
same ramp (sampled from the identical function) with the 1.0 threshold
marked as a labelled tick. A faint per-gcell grid overlay draws whenever
cells are large enough on screen to be legible (skipped below ~2px/cell
to avoid a moiré wash at extreme zoom-out). Macro outlines are
implemented (`drawMacroOutlines`, DBU-space rectangles converted through
the same grid<->canvas mapping) but never exercised yet - no macro
extraction exists anywhere in this pipeline, real or synthetic, so
`macro_outlines` is always `[]` (Gate 1). The frontend handles the empty
list correctly rather than assuming at least one macro.

## The three animations

1. **Bloom-in** (`startBloomAnimation` in `app.js`). On load (and on a
   design switch), the active mode's heatmap animates from flat/zero to
   its true values over 1200ms, ease-out. Per-gcell stagger: each cell's
   start delay is proportional to its own normalised value (up to 600ms
   for the single highest-value cell), so low-congestion cells bloom in
   almost immediately while hotspots are still held back - every cell's
   local ease-out duration is sized so all of them land exactly on their
   true value at the same t=1200ms, regardless of delay. This is what
   makes hotspots "settle last" per the brief, without ever overshooting
   or landing early.
2. **Predicted-vs-truth wipe** (Compare mode). A draggable vertical
   divider over one canvas, left = prediction layer, right = truth layer,
   via `ctx.clip()` on each layer's draw call. Drag tracks the pointer
   (fixed during testing - see below). `S` triggers a 3-leg auto-sweep
   (full left -> full right -> centre, ~0.9s/leg, eased) as the fallback
   for a live demo; pressing `S` also switches into Compare mode first if
   not already there, so one keystroke works as a demo macro.
3. **Click-to-zoom**. Clicking a hotspot table row animates the canvas
   viewport (not a CSS transform - an actual sub-rectangle of the
   gcell-resolution buffer, via `drawImage`'s source rect) to a window
   around that gcell over 500ms ease-out, sized as a fixed *fraction* of
   the grid's own dimensions (not a fixed cell count) so it zooms to a
   proportionally similar view regardless of grid size, with a label
   showing predicted/true/rank-delta. Hovering a row highlights that
   gcell on the canvas; hovering the canvas does the inverse lookup
   (pixel -> gcell -> matching row, if the hovered gcell is one of the
   20 ranked hotspots) and highlights the row. `Esc` zooms back out.

`prefers-reduced-motion: reduce` is checked once at startup
(`window.matchMedia`) and threaded through all three: bloom skips
straight to final values, the zoom viewport snaps instead of easing, and
the auto-sweep becomes a single instant jump. **Not verified live in a
browser** - there's no way to toggle the OS/browser reduced-motion
setting from within this session's browser automation, so this is
implemented-per-spec but unconfirmed, stated plainly rather than
claimed as tested.

## Honesty in the UI

- The fixture banner (`model_source != "checkpoint"`) is unmissable
  (full-width, amber, top of page) and its visibility is driven entirely
  by the API's `model_source` field - it will disappear with no code
  change once Gate 4's checkpoint exists.
- The metrics panel always shows Spearman and Kendall (H and V), and
  appends "Kendall computed exactly over every gcell" or "...on a
  2,000-gcell sample" depending on `kendall_sampled` - never presents a
  sampled figure as if it were exact.
- In fixture mode, the metrics panel also states plainly that the shown
  numbers describe the Gate 1 fixture, not a trained model.
- The design selector's split is shown as a coloured tag: "TRAIN (seen
  during training)" vs "VAL/TEST (held out)" - a viewer can always tell
  whether a prediction is on data the model (once real) has seen.
- The latency badge's inference figure is Gate 1's benchmarked median
  (never recomputed per paint); the router figure reads
  `reference_router_seconds` and shows literal "pending" styling when
  it's `null`, never a placeholder number.

## Bugs found and fixed while testing in a real browser

Per the project's own standing instruction to verify UI changes in an
actual browser rather than trust the code - three real bugs surfaced,
none of them visible from reading the code alone:

1. **Compare-mode drag tracked the wrong coordinate space.** The divider
   fraction was computed against the full canvas element's width, but
   the rendered die is letterboxed within `destRect` (die aspect ratio
   != canvas aspect ratio, which is normal - the canvas pane is wide and
   short, the die is near-square). Dragging therefore moved the divider
   to the wrong position whenever the canvas wasn't exactly the die's
   aspect ratio, i.e. almost always. Fixed by computing the fraction
   against `destRect`, not the canvas element, in device-pixel space.
2. **The die was rendered upside down.** `notes/phase2-congestion-extraction.md`
   established that gcell y=0 is the die's physical *bottom*
   (OpenROAD's origin='lower' convention) - but the offscreen buffer was
   being blitted with data row 0 as image row 0, which is the *top* by
   normal canvas convention. Caught by hovering a hotspot at
   `y` near the grid's maximum and seeing its highlight box render near
   the canvas's *bottom* instead of its top. Fixed with a single
   `gridYToCanvasY`/`canvasYToGridY` flip pair in `renderer.js`
   (`drawInto` now blits with a negative destination height; grid lines,
   macro outlines, hotspot highlight boxes and the hover hit-test all
   route through the same pair), so the displayed orientation now
   matches the convention the rest of the project already established
   rather than introducing a second, silently-inconsistent one.
3. **A race on rapid design switching.** `loadDesign` had no guard
   against a second call starting before the first one's five requests
   resolved; whichever finished last won, even if it was for the
   design no longer selected - caught by dispatching a design change
   programmatically right after the initial page-load request was still
   in flight, which left the dropdown and the rendered heatmap pointing
   at two different designs. Fixed with a monotonic `loadToken`: a
   `loadDesign` call checks its own token against the current one after
   its (now-parallelised, previously serial) five fetches resolve, and
   discards its result if a newer call has since started.

All three were caught via `claude-in-chrome` (real Chrome, real pointer
events, real network timing) against the Gate 1 API - not inferred from
reading the code, and not something a shape/type check would have
caught, since every response was individually well-formed.

## Verification

- Every control exercised in a live browser: design switch, channel
  toggle (H/V/max, confirmed visibly distinct patterns and a shared
  colour domain across prediction/truth), mode toggle (prediction,
  truth, compare, difference - confirmed distinct rendering per mode,
  including the diverging legend swap for Difference), Compare drag (post-fix),
  keyboard auto-sweep (`S`), click-to-zoom with its label and the
  highlight box landing on the correct cell (post-orientation-fix),
  `Esc` to zoom out, row hover <-> canvas region hover linking in both
  directions.
- Zero console errors across the full session (checked with
  `read_console_messages` after every interaction, not just at the end).
- All interactions re-verified after each fix, on the real `uvicorn`
  server (not a mock), against Gate 1's stub designs.

## Unresolved / limitations

1. `prefers-reduced-motion` is implemented but not live-verified (see
   above).
2. Macro outline rendering is implemented but never exercised against
   real geometry (none exists yet).
3. The bloom-in's exact mid-animation appearance wasn't captured in a
   screenshot (ease-out math means most non-hotspot cells are already
   >85% of the way to their final colour by the animation's midpoint,
   making a "clearly mid-flight" screenshot hard to time reliably from
   outside the page) - verified instead by confirming the canvas starts
   empty immediately after data load and reaches the correct final
   state, with the animation code path exercised on every design switch
   with no errors.
4. Gate 3 (difference-mode polish as a cross-fade rather than a cut,
   channel cross-fade, hotspot pulse) is explicitly deferred per the
   brief - Difference mode itself works today, just as an instant cut on
   mode switch rather than an animated transition.

## Commands to reproduce

```bash
cd ~/Documents/Congestion-detection-gnn
source .venv/bin/activate

python scripts/generate_stub_fixtures.py   # if data/stub/ isn't already populated
uvicorn api.main:app --reload --port 8000
# open http://127.0.0.1:8000
```
