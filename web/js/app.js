import { GridRenderer } from "./renderer.js";
import { congestionColor } from "./colormap.js";
import { diffColor } from "./colormap.js";
import { OVERFLOW_THRESHOLD } from "./colormap.js";
import { sequentialGradientCss } from "./colormap.js";
import { divergingGradientCss } from "./colormap.js";

const canvas = document.getElementById("main-canvas");
const renderer = new GridRenderer(canvas);

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const BLOOM_TOTAL_MS = 1200;
const BLOOM_MAX_DELAY_FRACTION = 0.5;
const ZOOM_TOTAL_MS = 500;
const SWEEP_LEG_MS = 900;

const state = {
  designs: [],
  modelSource: "fixture",
  currentDesignId: null,
  detail: null,
  prediction: null,
  truth: null,
  hotspots: [],
  hotspotByCoord: new Map(),
  channel: "max",
  mode: "prediction",
  wipeFraction: 0.5,
  destRect: null,
  predCombined: null,
  truthCombined: null,
  diffCombined: null,
  domainMax: 1.0,
  domainMaxAbsDiff: 1.0,
  zoomed: false,
  zoomedHotspot: null,
  hoveredKey: null,
  dragging: false,
};

let activeAnimationHandle = null;

function cancelActiveAnimation() {
  if (activeAnimationHandle !== null) {
    cancelAnimationFrame(activeAnimationHandle);
    activeAnimationHandle = null;
  }
}

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} -> HTTP ${response.status}`);
  }
  return response.json();
}

function easeOutCubic(x) {
  const inverse = 1 - x;
  return 1 - inverse * inverse * inverse;
}

function combinedChannel(horizontal, vertical, channel) {
  const n = horizontal.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (channel === "horizontal") {
      out[i] = horizontal[i];
    } else if (channel === "vertical") {
      out[i] = vertical[i];
    } else {
      out[i] = Math.max(horizontal[i], vertical[i]);
    }
  }
  return out;
}

function arrayMax(arr) {
  let result = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > result) {
      result = arr[i];
    }
  }
  return result;
}

function arrayMaxAbs(arr) {
  let result = 0;
  for (let i = 0; i < arr.length; i++) {
    const absValue = Math.abs(arr[i]);
    if (absValue > result) {
      result = absValue;
    }
  }
  return result;
}

// ---- data loading ----

async function init() {
  const designs = await fetchJSON("/designs");
  state.designs = designs;
  const firstDesign = designs[0];
  state.modelSource = firstDesign.model_source;
  updateFixtureBanner();
  populateDesignSelect(designs);
  wireControls();
  window.addEventListener("resize", () => drawCurrentMode());
  await loadDesign(firstDesign.id);
}

function populateDesignSelect(designs) {
  const select = document.getElementById("design-select");
  select.innerHTML = "";
  for (const design of designs) {
    const option = document.createElement("option");
    option.value = design.id;
    const cellCountLabel = design.cell_count.toLocaleString();
    option.textContent = `${design.name} (${cellCountLabel} cells, ${design.split})`;
    select.appendChild(option);
  }
  select.addEventListener("change", () => {
    loadDesign(select.value);
  });
}

function updateFixtureBanner() {
  const banner = document.getElementById("fixture-banner");
  banner.hidden = state.modelSource === "checkpoint";
}

let currentLoadToken = 0;

async function loadDesign(id) {
  currentLoadToken += 1;
  const loadToken = currentLoadToken;

  const detailPromise = fetchJSON(`/designs/${id}`);
  const predictionPromise = fetchJSON(`/designs/${id}/prediction`);
  const truthPromise = fetchJSON(`/designs/${id}/truth`);
  const hotspotsPromise = fetchJSON(`/designs/${id}/hotspots?k=20`);
  const metricsPromise = fetchJSON(`/designs/${id}/metrics`);

  const detail = await detailPromise;
  const prediction = await predictionPromise;
  const truth = await truthPromise;
  const hotspotsResponse = await hotspotsPromise;
  const metrics = await metricsPromise;

  // A newer loadDesign call started (and may already have finished)
  // while these requests were in flight - discard this stale response
  // rather than let a slow request for an old design overwrite a faster
  // one for the design actually selected now.
  if (loadToken !== currentLoadToken) {
    return;
  }

  cancelActiveAnimation();
  state.currentDesignId = id;
  state.zoomed = false;
  state.zoomedHotspot = null;
  document.getElementById("zoom-label").hidden = true;

  state.detail = detail;
  state.prediction = prediction;
  state.truth = truth;
  state.hotspots = hotspotsResponse.hotspots;
  state.modelSource = prediction.model_source;
  updateFixtureBanner();

  const hotspotByCoord = new Map();
  for (const row of state.hotspots) {
    hotspotByCoord.set(`${row.x},${row.y}`, row);
  }
  state.hotspotByCoord = hotspotByCoord;

  const ny = detail.grid_shape[0];
  const nx = detail.grid_shape[1];
  renderer.setGridShape(ny, nx);

  updateDesignMeta(detail);
  updateLatencyBadge(prediction);
  updateMetricsPanel(metrics);
  updateHotspotTable(state.hotspots);

  rebuildBuffers();

  if (state.mode === "compare" || reducedMotion) {
    drawCurrentMode();
  } else {
    startBloomAnimation();
  }
}

// ---- derived data / buffers ----

function rebuildBuffers() {
  const horizontalPred = state.prediction.horizontal;
  const verticalPred = state.prediction.vertical;
  const horizontalTruth = state.truth.horizontal;
  const verticalTruth = state.truth.vertical;

  state.predCombined = combinedChannel(horizontalPred, verticalPred, state.channel);
  state.truthCombined = combinedChannel(horizontalTruth, verticalTruth, state.channel);

  const n = state.predCombined.length;
  const diffCombined = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    diffCombined[i] = state.predCombined[i] - state.truthCombined[i];
  }
  state.diffCombined = diffCombined;

  const predMax = arrayMax(state.predCombined);
  const truthMax = arrayMax(state.truthCombined);
  state.domainMax = Math.max(predMax, truthMax, OVERFLOW_THRESHOLD + 0.001);
  state.domainMaxAbsDiff = Math.max(arrayMaxAbs(state.diffCombined), 1e-3);

  const sequentialColorFn = (v) => congestionColor(v, state.domainMax);
  const diffColorFn = (v) => diffColor(v, state.domainMaxAbsDiff);

  renderer.buildBuffer("prediction", state.predCombined, sequentialColorFn);
  renderer.buildBuffer("truth", state.truthCombined, sequentialColorFn);
  renderer.buildBuffer("difference", state.diffCombined, diffColorFn);

  updateLegend();
}

// ---- drawing ----

function drawCurrentMode() {
  if (!state.detail) {
    return;
  }
  const canvasSize = renderer.resizeToDisplaySize();
  renderer.clear(canvasSize.width, canvasSize.height);

  const dieAspect = state.detail.die_w_dbu / state.detail.die_h_dbu;
  const destRect = renderer.computeDestRect(canvasSize.width, canvasSize.height, dieAspect);
  state.destRect = destRect;

  if (state.mode === "compare") {
    canvas.style.cursor = "ew-resize";
    const dividerX = destRect.x + state.wipeFraction * destRect.width;
    const leftClip = {
      x: destRect.x, y: destRect.y,
      width: dividerX - destRect.x, height: destRect.height,
    };
    const rightClip = {
      x: dividerX, y: destRect.y,
      width: destRect.x + destRect.width - dividerX, height: destRect.height,
    };
    renderer.drawInto("prediction", destRect, leftClip);
    renderer.drawInto("truth", destRect, rightClip);
    renderer.drawGridLines(destRect);
    renderer.drawMacroOutlines(destRect, state.detail.macro_outlines, state.detail.die_w_dbu, state.detail.die_h_dbu);
    drawDivider(dividerX, destRect);
    drawCompareLabels(destRect);
  } else {
    canvas.style.cursor = "crosshair";
    renderer.drawInto(state.mode, destRect, null);
    renderer.drawGridLines(destRect);
    renderer.drawMacroOutlines(destRect, state.detail.macro_outlines, state.detail.die_w_dbu, state.detail.die_h_dbu);
  }

  drawHotspotHighlights(destRect);
}

function drawDivider(dividerX, destRect) {
  const ctx = renderer.ctx;
  ctx.save();
  ctx.strokeStyle = "#6ea8ff";
  ctx.lineWidth = 2;
  ctx.shadowColor = "rgba(110, 168, 255, 0.7)";
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(dividerX, destRect.y);
  ctx.lineTo(dividerX, destRect.y + destRect.height);
  ctx.stroke();

  ctx.shadowBlur = 0;
  ctx.fillStyle = "#6ea8ff";
  const gripY = destRect.y + destRect.height / 2;
  ctx.beginPath();
  ctx.ellipse(dividerX, gripY, 7, 16, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawCompareLabels(destRect) {
  const ctx = renderer.ctx;
  ctx.save();
  ctx.font = "12px -apple-system, sans-serif";
  ctx.fillStyle = "rgba(230, 233, 240, 0.85)";
  ctx.fillText("PREDICTED", destRect.x + 10, destRect.y + 18);
  const truthLabelWidth = ctx.measureText("TRUTH").width;
  ctx.fillText("TRUTH", destRect.x + destRect.width - truthLabelWidth - 10, destRect.y + 18);
  ctx.restore();
}

function drawHotspotHighlights(destRect) {
  const ctx = renderer.ctx;
  for (const row of state.hotspots) {
    const key = `${row.x},${row.y}`;
    const isHovered = key === state.hoveredKey;
    const isSelected = state.zoomedHotspot
      && row.x === state.zoomedHotspot.x
      && row.y === state.zoomedHotspot.y;
    if (!isHovered && !isSelected) {
      continue;
    }
    const rect = renderer.gridRectToCanvasRect(row.x, row.y, row.x + 1, row.y + 1, destRect);
    if (rect.x + rect.width < destRect.x || rect.x > destRect.x + destRect.width) {
      continue;
    }
    ctx.save();
    ctx.strokeStyle = isSelected ? "#f2b134" : "#e6e9f0";
    ctx.lineWidth = isSelected ? 2.5 : 1.5;
    const pad = 2;
    ctx.strokeRect(rect.x - pad, rect.y - pad, rect.width + pad * 2, rect.height + pad * 2);
    ctx.restore();
  }
}

// ---- bloom-in animation ----

function startBloomAnimation() {
  cancelActiveAnimation();
  const mode = state.mode;
  const trueValues = mode === "difference" ? state.diffCombined : getModeTrueValues(mode);
  if (!trueValues) {
    drawCurrentMode();
    return;
  }

  const colorFn = mode === "difference"
    ? (v) => diffColor(v, state.domainMaxAbsDiff)
    : (v) => congestionColor(v, state.domainMax);
  const normalizeFn = mode === "difference"
    ? (v) => Math.min(Math.abs(v) / state.domainMaxAbsDiff, 1)
    : (v) => Math.min(v / state.domainMax, 1);

  const n = trueValues.length;
  const frameBuffer = new Float64Array(n);
  const startTime = performance.now();

  function frame(now) {
    const elapsed = now - startTime;
    for (let i = 0; i < n; i++) {
      const trueValue = trueValues[i];
      const normalized = normalizeFn(trueValue);
      const delay = normalized * BLOOM_MAX_DELAY_FRACTION * BLOOM_TOTAL_MS;
      if (elapsed <= delay) {
        frameBuffer[i] = 0;
        continue;
      }
      const duration = BLOOM_TOTAL_MS - delay;
      const localProgress = Math.min((elapsed - delay) / duration, 1);
      const eased = easeOutCubic(localProgress);
      frameBuffer[i] = trueValue * eased;
    }
    renderer.buildBuffer(mode, frameBuffer, colorFn);
    drawCurrentMode();

    if (elapsed < BLOOM_TOTAL_MS) {
      activeAnimationHandle = requestAnimationFrame(frame);
    } else {
      renderer.buildBuffer(mode, trueValues, colorFn);
      drawCurrentMode();
      activeAnimationHandle = null;
    }
  }

  activeAnimationHandle = requestAnimationFrame(frame);
}

function getModeTrueValues(mode) {
  if (mode === "prediction") {
    return state.predCombined;
  }
  if (mode === "truth") {
    return state.truthCombined;
  }
  return null;
}

// ---- click-to-zoom ----

function computeZoomViewport(gx, gy) {
  const nx = renderer.nx;
  const ny = renderer.ny;
  const fraction = 0.16;
  const minHalfCells = 6;
  const halfWidth = Math.max(minHalfCells, Math.round((nx * fraction) / 2));
  const halfHeight = Math.max(minHalfCells, Math.round((ny * fraction) / 2));
  const fullWidth = Math.min(nx, halfWidth * 2);
  const fullHeight = Math.min(ny, halfHeight * 2);

  let x0 = gx - halfWidth;
  x0 = Math.max(0, Math.min(nx - fullWidth, x0));
  let y0 = gy - halfHeight;
  y0 = Math.max(0, Math.min(ny - fullHeight, y0));

  return { x0, y0, x1: x0 + fullWidth, y1: y0 + fullHeight };
}

function animateViewport(targetViewport, onDone) {
  cancelActiveAnimation();
  const startViewport = {
    x0: renderer.viewport.x0, y0: renderer.viewport.y0,
    x1: renderer.viewport.x1, y1: renderer.viewport.y1,
  };

  if (reducedMotion) {
    renderer.viewport = targetViewport;
    drawCurrentMode();
    if (onDone) {
      onDone();
    }
    return;
  }

  const startTime = performance.now();

  function frame(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / ZOOM_TOTAL_MS, 1);
    const eased = easeOutCubic(progress);
    renderer.viewport = {
      x0: startViewport.x0 + (targetViewport.x0 - startViewport.x0) * eased,
      y0: startViewport.y0 + (targetViewport.y0 - startViewport.y0) * eased,
      x1: startViewport.x1 + (targetViewport.x1 - startViewport.x1) * eased,
      y1: startViewport.y1 + (targetViewport.y1 - startViewport.y1) * eased,
    };
    drawCurrentMode();

    if (progress < 1) {
      activeAnimationHandle = requestAnimationFrame(frame);
    } else {
      activeAnimationHandle = null;
      if (onDone) {
        onDone();
      }
    }
  }

  activeAnimationHandle = requestAnimationFrame(frame);
}

function zoomToHotspot(row) {
  state.zoomed = true;
  state.zoomedHotspot = { x: row.x, y: row.y };
  const target = computeZoomViewport(row.x, row.y);
  animateViewport(target);

  const label = document.getElementById("zoom-label");
  label.hidden = false;
  label.innerHTML = `
    <div><strong>(${row.x}, ${row.y})</strong></div>
    <div>predicted: ${row.predicted.toFixed(3)}</div>
    <div>true: ${row.true.toFixed(3)}</div>
    <div>Δrank: ${row.rank_delta}</div>
  `;
}

function zoomOut() {
  if (!state.zoomed) {
    return;
  }
  state.zoomed = false;
  state.zoomedHotspot = null;
  document.getElementById("zoom-label").hidden = true;
  animateViewport({ x0: 0, y0: 0, x1: renderer.nx, y1: renderer.ny });
  highlightTableRow(null);
}

// ---- compare/wipe drag + keyboard auto-sweep ----

function startSweep() {
  cancelActiveAnimation();
  if (state.mode !== "compare") {
    setMode("compare");
  }

  if (reducedMotion) {
    state.wipeFraction = state.wipeFraction > 0.5 ? 0.0 : 1.0;
    drawCurrentMode();
    return;
  }

  const legs = [1.0, 0.0, 0.5];
  const startFraction = state.wipeFraction;
  let legIndex = 0;
  let legStart = startFraction;
  const startTime = performance.now();

  function frame(now) {
    const elapsed = now - startTime - legIndex * SWEEP_LEG_MS;
    const progress = Math.min(Math.max(elapsed / SWEEP_LEG_MS, 0), 1);
    const eased = easeOutCubic(progress);
    const target = legs[legIndex];
    state.wipeFraction = legStart + (target - legStart) * eased;
    drawCurrentMode();

    if (progress >= 1) {
      legStart = target;
      legIndex += 1;
    }
    if (legIndex < legs.length) {
      activeAnimationHandle = requestAnimationFrame(frame);
    } else {
      activeAnimationHandle = null;
    }
  }

  activeAnimationHandle = requestAnimationFrame(frame);
}

function onCanvasPointerDown(event) {
  if (state.mode !== "compare") {
    return;
  }
  cancelActiveAnimation();
  state.dragging = true;
  updateWipeFromPointer(event);
}

function onCanvasPointerMove(event) {
  if (state.dragging) {
    updateWipeFromPointer(event);
    return;
  }
  handleHoverAt(event);
}

function onCanvasPointerUp() {
  state.dragging = false;
}

function updateWipeFromPointer(event) {
  if (!state.destRect) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const pointerXDevice = (event.clientX - rect.left) * dpr;
  const destRect = state.destRect;
  const fraction = (pointerXDevice - destRect.x) / destRect.width;
  state.wipeFraction = Math.min(1, Math.max(0, fraction));
  drawCurrentMode();
}

function handleHoverAt(event) {
  if (!state.destRect) {
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const px = (event.clientX - rect.left) * dpr;
  const py = (event.clientY - rect.top) * dpr;
  const cell = renderer.canvasPointToGrid(px, py, state.destRect);
  if (!cell) {
    highlightTableRow(null);
    return;
  }
  const key = `${cell.gx},${cell.gy}`;
  const row = state.hotspotByCoord.get(key);
  if (row) {
    highlightTableRow(key);
  } else {
    highlightTableRow(null);
  }
}

// ---- UI wiring ----

function wireControls() {
  canvas.addEventListener("pointerdown", onCanvasPointerDown);
  canvas.addEventListener("pointermove", onCanvasPointerMove);
  window.addEventListener("pointerup", onCanvasPointerUp);

  document.getElementById("channel-toggle").addEventListener("click", (event) => {
    const button = event.target.closest(".toggle-btn");
    if (!button) {
      return;
    }
    setActiveButton("channel-toggle", button);
    state.channel = button.dataset.channel;
    rebuildBuffers();
    drawCurrentMode();
  });

  document.getElementById("mode-toggle").addEventListener("click", (event) => {
    const button = event.target.closest(".toggle-btn");
    if (!button) {
      return;
    }
    setActiveButton("mode-toggle", button);
    setMode(button.dataset.mode);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "s" || event.key === "S") {
      startSweep();
    } else if (event.key === "Escape") {
      zoomOut();
    }
  });
}

function setActiveButton(groupId, activeButton) {
  const group = document.getElementById(groupId);
  for (const button of group.querySelectorAll(".toggle-btn")) {
    button.classList.toggle("is-active", button === activeButton);
  }
}

function setMode(mode) {
  state.mode = mode;
  updateLegend();

  const group = document.getElementById("mode-toggle");
  for (const button of group.querySelectorAll(".toggle-btn")) {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  }

  if (mode === "prediction" || mode === "truth" || mode === "difference") {
    if (!reducedMotion) {
      startBloomAnimation();
      return;
    }
  }
  drawCurrentMode();
}

// ---- panels ----

function updateDesignMeta(detail) {
  const container = document.getElementById("design-meta");
  const splitClass = `split-tag split-tag-${detail.split}`;
  const splitLabel = detail.split === "train" ? "TRAIN (seen during training)" : `${detail.split.toUpperCase()} (held out)`;
  const nyText = detail.grid_shape[0];
  const nxText = detail.grid_shape[1];
  container.innerHTML = `
    <span class="${splitClass}">${splitLabel}</span>
    <div>${detail.cell_count.toLocaleString()} cells &middot; grid ${nyText}&times;${nxText}</div>
  `;
}

function updateLatencyBadge(prediction) {
  const inferenceEl = document.getElementById("inference-value");
  const inferenceMs = prediction.inference_seconds * 1000;
  inferenceEl.textContent = `${inferenceMs.toFixed(1)} ms`;

  const routerEl = document.getElementById("router-value");
  if (prediction.reference_router_seconds === null || prediction.reference_router_seconds === undefined) {
    routerEl.textContent = "pending";
    routerEl.classList.add("latency-badge-pending");
  } else {
    const routerMs = prediction.reference_router_seconds * 1000;
    routerEl.textContent = `${routerMs.toFixed(0)} ms`;
    routerEl.classList.remove("latency-badge-pending");
  }
}

function formatMetricValue(value) {
  return value.toFixed(3);
}

function updateMetricsPanel(metrics) {
  const container = document.getElementById("metrics-panel");
  const h = metrics.channels.horizontal;
  const v = metrics.channels.vertical;
  const kendallNote = h.kendall_sampled || v.kendall_sampled
    ? "Kendall computed on a 2,000-gcell sample (grid exceeds the exact-computation threshold)."
    : "Kendall computed exactly over every gcell.";
  const fixtureNote = state.modelSource !== "checkpoint"
    ? "These numbers describe the Gate 1 fixture (truth + calibrated noise), not a trained model."
    : "";

  container.innerHTML = `
    <div class="metric-row"><span>Spearman H</span><span>${formatMetricValue(h.spearman)}</span></div>
    <div class="metric-row"><span>Spearman V</span><span>${formatMetricValue(v.spearman)}</span></div>
    <div class="metric-row"><span>Kendall H</span><span>${formatMetricValue(h.kendall)}</span></div>
    <div class="metric-row"><span>Kendall V</span><span>${formatMetricValue(v.kendall)}</span></div>
    <div class="metric-row"><span>MAE H</span><span>${formatMetricValue(h.mae)}</span></div>
    <div class="metric-row"><span>MAE V</span><span>${formatMetricValue(v.mae)}</span></div>
    <div class="metric-row"><span>RMSE H</span><span>${formatMetricValue(h.rmse)}</span></div>
    <div class="metric-row"><span>RMSE V</span><span>${formatMetricValue(v.rmse)}</span></div>
    <div class="metric-row"><span>Hotspot precision / recall H</span><span>${formatMetricValue(h.hotspot_precision)} / ${formatMetricValue(h.hotspot_recall)}</span></div>
    <div class="metric-row"><span>Hotspot precision / recall V</span><span>${formatMetricValue(v.hotspot_precision)} / ${formatMetricValue(v.hotspot_recall)}</span></div>
    <div class="metric-row"><span>ROC-AUC H</span><span>${formatMetricValue(h.hotspot_auc)}</span></div>
    <div class="metric-row"><span>ROC-AUC V</span><span>${formatMetricValue(v.hotspot_auc)}</span></div>
    <div class="metric-note">${kendallNote}</div>
    ${fixtureNote ? `<div class="metric-note">${fixtureNote}</div>` : ""}
  `;
}

function updateHotspotTable(hotspots) {
  const tbody = document.getElementById("hotspot-tbody");
  tbody.innerHTML = "";
  hotspots.forEach((row, index) => {
    const tr = document.createElement("tr");
    const key = `${row.x},${row.y}`;
    tr.dataset.key = key;
    const rankDeltaClass = row.rank_delta > 0 ? "rank-delta-pos" : row.rank_delta < 0 ? "rank-delta-neg" : "";
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td>${row.x},${row.y}</td>
      <td>${row.predicted.toFixed(2)}</td>
      <td>${row.true.toFixed(2)}</td>
      <td class="${rankDeltaClass}">${row.rank_delta > 0 ? "+" : ""}${row.rank_delta}</td>
    `;
    tr.addEventListener("mouseenter", () => {
      state.hoveredKey = key;
      drawCurrentMode();
    });
    tr.addEventListener("mouseleave", () => {
      if (state.hoveredKey === key) {
        state.hoveredKey = null;
        drawCurrentMode();
      }
    });
    tr.addEventListener("click", () => {
      if (state.zoomedHotspot && state.zoomedHotspot.x === row.x && state.zoomedHotspot.y === row.y) {
        zoomOut();
        return;
      }
      zoomToHotspot(row);
      highlightTableRow(key, true);
    });
    tbody.appendChild(tr);
  });
}

function highlightTableRow(key, isSelection) {
  state.hoveredKey = isSelection ? state.hoveredKey : key;
  const tbody = document.getElementById("hotspot-tbody");
  for (const tr of tbody.querySelectorAll("tr")) {
    tr.classList.toggle("is-hovered", tr.dataset.key === key && !isSelection);
    tr.classList.toggle("is-selected", state.zoomedHotspot
      && tr.dataset.key === `${state.zoomedHotspot.x},${state.zoomedHotspot.y}`);
  }
  if (!isSelection) {
    drawCurrentMode();
  }
}

function updateLegend() {
  const container = document.getElementById("legend");
  const isDifference = state.mode === "difference";
  const gradientCss = isDifference
    ? divergingGradientCss(state.domainMaxAbsDiff)
    : sequentialGradientCss(state.domainMax);

  const ramp = document.createElement("div");
  ramp.className = "legend-ramp";
  ramp.style.background = gradientCss;

  const scale = document.createElement("div");
  scale.className = "legend-scale";

  if (isDifference) {
    scale.innerHTML = `
      <span>under-predict (−${state.domainMaxAbsDiff.toFixed(2)})</span>
      <span>agree (0)</span>
      <span>over-predict (+${state.domainMaxAbsDiff.toFixed(2)})</span>
    `;
  } else {
    const thresholdFraction = OVERFLOW_THRESHOLD / state.domainMax;
    const thresholdPercent = Math.min(100, Math.max(0, thresholdFraction * 100));
    scale.innerHTML = `
      <span>0</span>
      <span class="legend-threshold" style="left:${thresholdPercent}%">1.0 overflow</span>
      <span>${state.domainMax.toFixed(2)}</span>
    `;
  }

  container.innerHTML = "";
  container.appendChild(ramp);
  container.appendChild(scale);
}

init();
