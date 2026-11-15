/** Pure helpers behind TimeSeriesPlotComponent. No DOM. */

/** One per-frame trace. `null` or NaN marks a frame with no value. */
export interface TimeSeries {
  label: string;
  values: ArrayLike<number | null>;
  /** Line colour; defaults to the tab10 cycle. */
  color?: string;
  /** Dashed line (used for the y axis of per-axis signals). */
  dashed?: boolean;
}

/** Visible frame window, [start, end). */
export interface FrameWindow {
  start: number;
  end: number;
}

/** A frame segment, [start, end), as the extract request takes it. */
export type FrameSegment = [number, number];

export const MIN_WINDOW_FRAMES = 20;

function finite(v: number | null | undefined): v is number {
  return v != null && Number.isFinite(v);
}

/**
 * Min and max of each pixel column over the frames it covers.
 *
 * Plotting min/max per column keeps a one-frame spike visible however many
 * frames share a pixel, which is the point of this plot. Columns with no
 * finite value are NaN.
 */
export function decimate(
  values: ArrayLike<number | null>,
  window: FrameWindow,
  columns: number,
): { min: Float64Array; max: Float64Array } {
  const min = new Float64Array(columns).fill(NaN);
  const max = new Float64Array(columns).fill(NaN);
  const span = window.end - window.start;
  if (columns <= 0 || span <= 0) return { min, max };
  const end = Math.min(window.end, values.length);
  for (let f = Math.max(0, window.start); f < end; f++) {
    const v = values[f];
    if (!finite(v)) continue;
    const c = Math.min(
      columns - 1,
      Math.floor(((f - window.start) / span) * columns),
    );
    if (!(v >= min[c])) min[c] = v;
    if (!(v <= max[c])) max[c] = v;
  }
  return { min, max };
}

/** Finite [min, max] over all series inside the window, padded 5 %. */
export function valueRange(
  series: TimeSeries[],
  window: FrameWindow,
): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
    const end = Math.min(window.end, s.values.length);
    for (let f = Math.max(0, window.start); f < end; f++) {
      const v = s.values[f];
      if (!finite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo)) return [0, 1];
  if (lo === hi) return [lo - 1, hi + 1];
  const pad = (hi - lo) * 0.05;
  return [lo - pad, hi + pad];
}

/** Frame under a horizontal position in a plot area of `width` px. */
export function frameAtX(x: number, width: number, window: FrameWindow): number {
  if (width <= 0) return window.start;
  const t = Math.min(Math.max(x / width, 0), 1);
  const frame = Math.floor(window.start + t * (window.end - window.start));
  return Math.min(frame, window.end - 1);
}

/** Horizontal position of the centre of a frame. */
export function xOfFrame(frame: number, width: number, window: FrameWindow): number {
  return ((frame + 0.5 - window.start) / (window.end - window.start)) * width;
}

/**
 * Zoom the window by `factor` (<1 zooms in) around `anchor`, keeping the
 * anchor frame under the pointer and the window inside [0, nFrames).
 */
export function zoomWindow(
  window: FrameWindow,
  anchor: number,
  factor: number,
  nFrames: number,
): FrameWindow {
  const span = window.end - window.start;
  const minSpan = Math.min(MIN_WINDOW_FRAMES, nFrames);
  const nextSpan = Math.round(
    Math.min(Math.max(span * factor, minSpan), nFrames),
  );
  const ratio = span > 0 ? (anchor - window.start) / span : 0.5;
  let start = Math.round(anchor - ratio * nextSpan);
  start = Math.min(Math.max(start, 0), nFrames - nextSpan);
  return { start, end: start + nextSpan };
}

/** Shift the window by `frames`, clamped to the video. */
export function panWindow(
  window: FrameWindow,
  frames: number,
  nFrames: number,
): FrameWindow {
  const span = window.end - window.start;
  const start = Math.min(Math.max(window.start + frames, 0), nFrames - span);
  return { start, end: start + span };
}

/** Normalised segment from two frames picked in any order (end exclusive). */
export function segmentBetween(a: number, b: number): FrameSegment {
  return [Math.min(a, b), Math.max(a, b) + 1];
}

// Confidence colours: good is quiet, bad stands out. Values at or above 0.9
// are near-white on the white plot; below that the colour runs through
// orange to deep red, so low-confidence stretches are what the eye finds.
const CONFIDENCE_STOPS: [number, [number, number, number]][] = [
  [0.0, [127, 0, 0]],
  [0.3, [200, 16, 32]],
  [0.5, [240, 59, 32]],
  [0.7, [253, 141, 60]],
  [0.85, [254, 217, 142]],
  [0.9, [244, 244, 244]],
  [1.0, [250, 250, 250]],
];

/** Colour for a confidence in [0, 1] as `rgb(r, g, b)`: near-white when high, red when low. */
export function confidenceColor(t: number): string {
  const v = Math.min(Math.max(Number.isFinite(t) ? t : 0, 0), 1);
  let i = 0;
  while (i < CONFIDENCE_STOPS.length - 2 && v > CONFIDENCE_STOPS[i + 1][0]) i++;
  const [t0, a] = CONFIDENCE_STOPS[i];
  const [t1, b] = CONFIDENCE_STOPS[i + 1];
  const f = t1 > t0 ? (v - t0) / (t1 - t0) : 0;
  const c = a.map((x, k) => Math.round(x + (b[k] - x) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}



/** matplotlib tab10, in ethograph's order (MULTIDIM_COLORS). */
export const TRACE_COLORS = [
  '#1f77b4',
  '#d62728',
  '#2ca02c',
  '#ff7f0e',
  '#9467bd',
  '#8c564b',
  '#e377c2',
  '#7f7f7f',
  '#bcbd22',
  '#17becf',
];

/** About `target` evenly spaced round values (1, 2, 5 × 10^k) covering [lo, hi]. */
export function niceTicks(lo: number, hi: number, target = 5): number[] {
  if (!(hi > lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  const step = niceStep((hi - lo) / Math.max(target, 1));
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return ticks;
}

export function niceStep(raw: number): number {
  if (!(raw > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(raw));
  const f = raw / power;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * power;
}

/** Time tick label that adapts to the tick spacing, as in ethograph's TimeAxisItem. */
export function formatTime(seconds: number, spacing: number): string {
  const sign = seconds < 0 ? '-' : '';
  const a = Math.abs(seconds);
  const minutes = Math.floor(a / 60);
  const s = a % 60;
  if (spacing >= 60) return `${sign}${minutes}:${String(Math.floor(s)).padStart(2, '0')}`;
  if (spacing >= 1) {
    if (minutes > 0) return `${sign}${minutes}:${s.toFixed(0).padStart(2, '0')}`;
    return `${sign}${Math.round(s)}s`;
  }
  if (minutes > 0) {
    const digits = spacing >= 0.1 ? 1 : 2;
    return `${sign}${minutes}:${s.toFixed(digits).padStart(3 + digits, '0')}`;
  }
  if (spacing >= 0.1) return `${sign}${s.toFixed(1)}s`;
  return `${sign}${(a * 1000).toFixed(0)}ms`;
}

/** Zoom a value range by `factor` around `anchor`. */
export function zoomRange(
  range: [number, number],
  anchor: number,
  factor: number,
): [number, number] {
  return [anchor + (range[0] - anchor) * factor, anchor + (range[1] - anchor) * factor];
}

/**
 * Largest per-frame change (deg) still read as one continuous turn across ±180.
 * A bigger change is a real jump (often a tracking error) and is drawn.
 */
export const MAX_WRAP_STEP_DEG = 60;

/**
 * True when prev → next crosses the ±180° seam with a small real change
 * (e.g. 178 → -179 is a 3° turn), as opposed to a large jump.
 */
export function crossesWrap(prev: number, next: number): boolean {
  const delta = next - prev;
  return Math.abs(delta) > 180 && 360 - Math.abs(delta) <= MAX_WRAP_STEP_DEG;
}

/** Ticks on the confidence colour bar; 0.9 marks where the colours turn white. */
export const COLORBAR_TICKS = [0, 0.25, 0.5, 0.75, 0.9, 1];
/** Ticks when the bar is too short for all of them. */
export const COLORBAR_TICKS_SHORT = [0, 0.5, 0.9];
