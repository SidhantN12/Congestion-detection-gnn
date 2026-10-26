// Congestion colour ramp: deep blue (uncongested) -> cyan -> amber -> red
// at the 1.0 overflow threshold -> a darker, more saturated red beyond it.
// Hand-picked stops rather than a library colormap (jet/rainbow), per the
// brief's "perceptually reasonable, not raw rainbow" requirement.

export const OVERFLOW_THRESHOLD = 1.0;

const RAMP_STOPS = [
  { t: 0.0, rgb: [8, 20, 56] },
  { t: 0.35, rgb: [23, 83, 180] },
  { t: 0.55, rgb: [34, 196, 194] },
  { t: 0.78, rgb: [240, 176, 52] },
  { t: 1.0, rgb: [220, 50, 55] },
];

const OVERFLOW_STOPS = [
  { t: 1.0, rgb: [220, 50, 55] },
  { t: 1.6, rgb: [120, 10, 30] },
];

function lerp(a, b, f) {
  return a + (b - a) * f;
}

function lerpRgb(rgbA, rgbB, f) {
  const r = lerp(rgbA[0], rgbB[0], f);
  const g = lerp(rgbA[1], rgbB[1], f);
  const b = lerp(rgbA[2], rgbB[2], f);
  return [r, g, b];
}

function sampleStops(stops, t) {
  if (t <= stops[0].t) {
    return stops[0].rgb;
  }
  const lastIndex = stops.length - 1;
  if (t >= stops[lastIndex].t) {
    return stops[lastIndex].rgb;
  }
  for (let i = 0; i < lastIndex; i++) {
    const left = stops[i];
    const right = stops[i + 1];
    if (t >= left.t && t <= right.t) {
      const span = right.t - left.t;
      const localF = span > 0 ? (t - left.t) / span : 0;
      return lerpRgb(left.rgb, right.rgb, localF);
    }
  }
  return stops[lastIndex].rgb;
}

// value: raw congestion ratio. domainMax: the current design's max value
// (>= OVERFLOW_THRESHOLD), used to scale the overflow segment - never a
// hard-coded cap, since real designs' max congestion is unknown ahead of
// time.
export function congestionColor(value, domainMax) {
  const safeMax = Math.max(domainMax, OVERFLOW_THRESHOLD + 1e-6);
  let rgb;
  if (value <= OVERFLOW_THRESHOLD) {
    const normalized = value / OVERFLOW_THRESHOLD;
    rgb = sampleStops(RAMP_STOPS, normalized);
  } else {
    const overflowFraction = (value - OVERFLOW_THRESHOLD) / (safeMax - OVERFLOW_THRESHOLD);
    const t = 1.0 + Math.min(overflowFraction, 1.0) * 0.6;
    rgb = sampleStops(OVERFLOW_STOPS, t);
  }
  return rgb;
}

// Diverging scale for difference mode: blue = under-prediction
// (predicted < true), red = over-prediction, pale neutral = agreement.
const DIFF_NEGATIVE = [60, 110, 220];
const DIFF_NEUTRAL = [40, 44, 54];
const DIFF_POSITIVE = [220, 70, 60];

export function diffColor(diffValue, domainMaxAbs) {
  const safeMax = Math.max(domainMaxAbs, 1e-6);
  const clamped = Math.max(-safeMax, Math.min(safeMax, diffValue));
  const f = clamped / safeMax;
  if (f >= 0) {
    return lerpRgb(DIFF_NEUTRAL, DIFF_POSITIVE, f);
  }
  return lerpRgb(DIFF_NEUTRAL, DIFF_NEGATIVE, -f);
}

export function rgbToCss(rgb) {
  const r = Math.round(rgb[0]);
  const g = Math.round(rgb[1]);
  const b = Math.round(rgb[2]);
  return `rgb(${r}, ${g}, ${b})`;
}

// A CSS linear-gradient string for the legend ramp, sampling the same
// function the canvas uses so the legend can never visually drift from
// the data it's labelling.
export function sequentialGradientCss(domainMax) {
  const stepCount = 32;
  const parts = [];
  for (let i = 0; i <= stepCount; i++) {
    const frac = i / stepCount;
    const value = frac * domainMax;
    const rgb = congestionColor(value, domainMax);
    const pct = (frac * 100).toFixed(2);
    parts.push(`${rgbToCss(rgb)} ${pct}%`);
  }
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

export function divergingGradientCss(domainMaxAbs) {
  const stepCount = 32;
  const parts = [];
  for (let i = 0; i <= stepCount; i++) {
    const frac = i / stepCount;
    const value = (frac * 2 - 1) * domainMaxAbs;
    const rgb = diffColor(value, domainMaxAbs);
    const pct = (frac * 100).toFixed(2);
    parts.push(`${rgbToCss(rgb)} ${pct}%`);
  }
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}
