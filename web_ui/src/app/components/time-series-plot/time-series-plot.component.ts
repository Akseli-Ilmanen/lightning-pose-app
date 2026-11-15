import {
  afterNextRender,
  ChangeDetectionStrategy,
  Component,
  computed,
  DestroyRef,
  effect,
  ElementRef,
  inject,
  input,
  model,
  output,
  signal,
  viewChild,
} from '@angular/core';
import {
  decimate,
  formatTime,
  FrameSegment,
  FrameWindow,
  frameAtX,
  niceStep,
  niceTicks,
  panWindow,
  segmentBetween,
  TimeSeries,
  TRACE_COLORS,
  valueRange,
  confidenceColor,
  crossesWrap,
  COLORBAR_TICKS,
  COLORBAR_TICKS_SHORT,
  xOfFrame,
  zoomRange,
  zoomWindow,
} from './time-series-math';

/** Pointer travel (px) before a press becomes a drag instead of a click. */
const DRAG_THRESHOLD_PX = 4;

// pyqtgraph-on-white look, as in ethograph's BasePlot.
const PLOT_BG = '#ffffff';
const AXIS = '#3c3c3c';
const TIME_MARKER = '#ff0000';
const REGION_FILL = 'rgba(0, 0, 255, 0.16)';
const REGION_EDGE = 'rgba(0, 0, 255, 0.55)';
const SINGLE_TRACE = '#000000';
const FONT = '11px ui-sans-serif, system-ui, sans-serif';

const PAD_TOP = 8;
const PAD_RIGHT = 10;
/** Right gutter of the heatmap: gap, bar, ticks, labels, title. */
const COLORBAR_GUTTER = 78;
/** Margins for plots stacked on one time axis: room for row labels and a colour bar in every row. */
export const STACKED_GUTTERS = { left: 96, right: COLORBAR_GUTTER };
const COLORBAR_GAP = 10;
const COLORBAR_WIDTH = 12;
const X_AXIS_H = 22;

interface Layout {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Per-frame traces on a canvas, styled after ethograph's pyqtgraph plots
 * (white plot, tick axes with time in seconds, red time marker).
 *
 * - Click: emits `seek` with the frame under the pointer.
 * - Drag: marks a segment (two-way `segment`).
 * - Wheel: zooms time around the pointer. Ctrl+wheel, or the wheel over the
 *   value axis, zooms values only; the wheel over the time axis zooms time only.
 * - Right-drag: zooms each axis on its own (left/right time, up/down values).
 * - Shift+wheel, Alt+drag or middle-drag pans. Double-click or **A** autoscales.
 *
 * `lines` draws one line per series. When frames outnumber pixels, each pixel
 * shows the min–max of its frames so a one-frame spike stays visible.
 * `heatmap` draws one row per series with values in [0, 1] (confidence),
 * near-white when confident and red when not; each pixel shows the lowest
 * value it covers, so a dip is never averaged away.
 */
@Component({
  selector: 'app-time-series-plot',
  templateUrl: './time-series-plot.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block' },
})
export class TimeSeriesPlotComponent {
  series = input.required<TimeSeries[]>();
  nFrames = input.required<number>();
  currentFrame = input(0);
  mode = input<'lines' | 'heatmap'>('lines');
  /** Canvas height in px, axes included. */
  height = input(180);
  unit = input('');
  /** Value-axis title, e.g. "Speed". The unit is appended. */
  yLabel = input('');
  /** Frames per second. When set, the time axis is in seconds. */
  fps = input<number | null>(null);
  /** Break lines at jumps over 180 (angles in degrees that wrap at ±180). */
  wrapAngles = input(false);
  ariaLabel = input('Prediction time series');
  /** Marked segment [start, end), or null. */
  segment = model<FrameSegment | null>(null);
  seek = output<number>();

  private host = inject<ElementRef<HTMLElement>>(ElementRef);
  private destroyRef = inject(DestroyRef);
  private baseCanvas = viewChild.required<ElementRef<HTMLCanvasElement>>('base');
  private overlayCanvas =
    viewChild.required<ElementRef<HTMLCanvasElement>>('overlay');

  protected width = signal(0);
  /**
   * Zoomed frames [start, end); null shows the whole video. Two-way, so
   * stacked plots can share one time axis (zoom one, all follow).
   */
  xWindow = model<FrameWindow | null>(null);
  /** The zoomed window, or the whole video when unset or out of range. */
  protected window = computed<FrameWindow>(() => {
    const n = Math.max(this.nFrames(), 1);
    const w = this.xWindow();
    if (!w || w.start < 0 || w.end > n || w.end <= w.start) return { start: 0, end: n };
    return w;
  });
  /**
   * Fixed left/right margins (px). Stacked plots pass the same values so their
   * time axes line up; otherwise the margins depend on the mode.
   */
  gutters = input<{ left: number; right: number } | null>(null);
  /**
   * Centre on the current frame, as ethograph's centre-playback mode: the time
   * marker stays in the middle and the traces slide under it, keeping the zoom
   * span. The window may run past the start or end of the recording. Panning
   * gestures scrub the video instead of moving the window.
   */
  follow = model(false);
  /** Frames on screen: the zoomed window, or the same span centred on the current frame. */
  protected view = computed<FrameWindow>(() => {
    const w = this.window();
    if (!this.follow()) return w;
    const span = w.end - w.start;
    const start = this.currentFrame() - Math.floor(span / 2);
    return { start, end: start + span };
  });
  /**
   * Fixed value range [min, max]; null autoscales to what is visible.
   * Two-way, so a parent can set it from min/max fields and keep it per signal.
   * Ctrl+wheel and right-drag on y set it; A / double-click clear it.
   */
  yLimits = model<[number, number] | null>(null);
  /** Value range on screen: the fixed limits, or the autoscaled range. */
  readonly yRange = computed<[number, number]>(
    () => this.yLimits() ?? valueRange(this.series(), this.view()),
  );
  protected isZoomed = computed(
    () =>
      this.window().start > 0 ||
      this.window().end < this.nFrames() ||
      this.yLimits() != null,
  );
  private dragPreview = signal<FrameSegment | null>(null);
  private press: {
    x: number;
    y: number;
    frame: number;
    /** What the drag does: mark a segment, pan, or zoom each axis (right-drag). */
    action: 'segment' | 'pan' | 'zoom';
    /** Value under the pointer at press, the y anchor for right-drag zoom. */
    value: number;
    dragging: boolean;
    window: FrameWindow;
    /** Current frame at press, the start point for scrubbing in follow mode. */
    current: number;
    yRange: [number, number];
  } | null = null;

  protected layout = computed<Layout>(() => {
    const heatmap = this.mode() === 'heatmap';
    const fixed = this.gutters();
    const left = fixed?.left ?? (heatmap ? 96 : 64);
    // The heatmap keeps room on the right for its vertical colour bar.
    const right = fixed?.right ?? (heatmap ? COLORBAR_GUTTER : PAD_RIGHT);
    return {
      left,
      top: PAD_TOP,
      width: Math.max(this.width() - left - right, 1),
      height: Math.max(this.height() - PAD_TOP - X_AXIS_H, 1),
    };
  });


  protected colors = computed(() => {
    const series = this.series();
    return series.map(
      (s, i) =>
        s.color ?? (series.length === 1 ? SINGLE_TRACE : TRACE_COLORS[i % TRACE_COLORS.length]),
    );
  });

  /** Legend entries with each series' value at the current frame. */
  protected legend = computed(() => {
    const frame = this.currentFrame();
    const colors = this.colors();
    return this.series().map((s, i) => {
      const v = s.values[frame];
      return {
        label: s.label,
        color: colors[i],
        dashed: !!s.dashed,
        value: v != null && Number.isFinite(v) ? formatValue(v) : '–',
      };
    });
  });

  constructor() {
    afterNextRender(() => {
      const el = this.host.nativeElement.querySelector('.plot-area');
      if (!el) return;
      const observer = new ResizeObserver(() => this.width.set(el.clientWidth));
      observer.observe(el);
      this.width.set(el.clientWidth);
      this.destroyRef.onDestroy(() => observer.disconnect());
    });
    // Canvas drawing is an imperative side effect: the data layer redraws when
    // data or zoom changes, the overlay (time marker, segment) on every frame.
    effect(() => this.drawBase());
    effect(() => this.drawOverlay());
  }

  // ---------------------------------------------------------------------------
  // Pointer interaction
  // ---------------------------------------------------------------------------

  private local(event: MouseEvent): { x: number; y: number } {
    const rect = this.overlayCanvas().nativeElement.getBoundingClientRect();
    const l = this.layout();
    return { x: event.clientX - rect.left - l.left, y: event.clientY - rect.top - l.top };
  }

  private frameAt(event: MouseEvent): number {
    const frame = frameAtX(this.local(event).x, this.layout().width, this.view());
    return Math.min(Math.max(frame, 0), Math.max(this.nFrames() - 1, 0));
  }

  /** Follow mode: move through the recording by `frames` instead of panning. */
  private scrub(from: number, frames: number) {
    const last = Math.max(this.nFrames() - 1, 0);
    this.seek.emit(Math.min(Math.max(from + frames, 0), last));
  }

  private valueAt(event: MouseEvent): number {
    const [lo, hi] = this.yRange();
    const t = 1 - this.local(event).y / this.layout().height;
    return lo + t * (hi - lo);
  }

  protected onPointerDown(event: PointerEvent) {
    if (this.nFrames() <= 0) return;
    let action: 'segment' | 'pan' | 'zoom';
    if (event.button === 2) action = 'zoom';
    else if (event.button === 1 || (event.button === 0 && event.altKey)) action = 'pan';
    else if (event.button === 0) action = 'segment';
    else return;
    if (action !== 'segment') event.preventDefault();
    (event.target as Element).setPointerCapture?.(event.pointerId);
    this.press = {
      x: event.clientX,
      y: event.clientY,
      frame: this.frameAt(event),
      action,
      value: this.valueAt(event),
      dragging: false,
      window: this.window(),
      yRange: this.yRange(),
      current: this.currentFrame(),
    };
  }

  protected onPointerMove(event: PointerEvent) {
    const press = this.press;
    if (!press) return;
    const dx = event.clientX - press.x;
    const dy = event.clientY - press.y;
    if (!press.dragging && Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return;
    press.dragging = true;
    if (press.action === 'zoom') {
      // pyqtgraph right-drag: right/left zooms time in/out, up/down zooms values
      // in/out, each axis on its own, anchored where the drag started.
      const fx = Math.exp(-dx / 150);
      const fy = Math.exp(dy / 150);
      const n = this.nFrames();
      this.xWindow.set(zoomWindow(press.window, press.frame, fx, n));
      if (this.mode() === 'lines') {
        this.yLimits.set(zoomRange(press.yRange, press.value, fy));
      }
      return;
    }
    if (press.action === 'pan') {
      const l = this.layout();
      const span = press.window.end - press.window.start;
      const frames = Math.round((-dx / l.width) * span);
      if (this.follow()) this.scrub(press.current, frames);
      else this.xWindow.set(panWindow(press.window, frames, this.nFrames()));
      if (this.mode() === 'lines') {
        const [lo, hi] = press.yRange;
        const shift = (dy / l.height) * (hi - lo);
        this.yLimits.set([lo + shift, hi + shift]);
      }
      return;
    }
    this.dragPreview.set(segmentBetween(press.frame, this.frameAt(event)));
  }

  protected onPointerUp() {
    const press = this.press;
    this.press = null;
    if (!press || press.action !== 'segment') return;
    if (press.dragging) {
      this.segment.set(this.dragPreview());
      this.dragPreview.set(null);
    } else {
      this.seek.emit(press.frame);
    }
  }

  protected onWheel(event: WheelEvent) {
    const n = this.nFrames();
    if (n <= 0) return;
    event.preventDefault();
    const zoomIn = event.deltaY < 0;
    // Over an axis, the wheel zooms only that axis (as in pyqtgraph).
    const { x, y } = this.local(event);
    const overYAxis = x < 0;
    const overXAxis = y > this.layout().height;
    if (
      this.mode() === 'lines' &&
      (overYAxis || ((event.ctrlKey || event.metaKey) && !overXAxis))
    ) {
      this.yLimits.set(
        zoomRange(this.yRange(), this.valueAt(event), zoomIn ? 0.8 : 1.25),
      );
      return;
    }
    const w = this.window();
    if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
      const delta = event.shiftKey ? event.deltaY : event.deltaX;
      const frames = Math.round(((w.end - w.start) * Math.sign(delta)) / 10);
      if (this.follow()) this.scrub(this.currentFrame(), frames);
      else this.xWindow.set(panWindow(w, frames, n));
      return;
    }
    this.xWindow.set(zoomWindow(w, this.frameAt(event), zoomIn ? 0.8 : 1.25, n));
  }

  protected onKeyDown(event: KeyboardEvent) {
    const n = this.nFrames();
    const w = this.window();
    const centre = Math.round((w.start + w.end) / 2);
    if (event.key === '+' || event.key === '=') {
      this.xWindow.set(zoomWindow(w, centre, 0.8, n));
    } else if (event.key === '-') {
      this.xWindow.set(zoomWindow(w, centre, 1.25, n));
    } else if (event.key === '0' || event.key === 'a' || event.key === 'A') {
      this.resetZoom();
    } else {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
  }

  /** Show the whole video and autoscale the value axis (pyqtgraph's "A"). */
  resetZoom() {
    this.xWindow.set(null);
    this.yLimits.set(null);
  }

  // ---------------------------------------------------------------------------
  // Drawing
  // ---------------------------------------------------------------------------

  private prepare(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
    const dpr = globalThis.devicePixelRatio || 1;
    const w = this.width();
    const h = this.height();
    if (canvas.width !== Math.round(w * dpr)) canvas.width = Math.round(w * dpr);
    if (canvas.height !== Math.round(h * dpr)) canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return ctx;
  }

  private drawBase() {
    const series = this.series();
    const win = this.view();
    const mode = this.mode();
    const l = this.layout();
    const colors = this.colors();
    const [lo, hi] = this.yRange();
    const fps = this.fps();
    const wrap = this.wrapAngles();
    const yTitle = [this.yLabel(), this.unit() ? `(${this.unit()})` : '']
      .filter(Boolean)
      .join(' ');
    if (this.width() <= 0) return;
    const ctx = this.prepare(this.baseCanvas().nativeElement);
    if (!ctx) return;

    ctx.fillStyle = PLOT_BG;
    ctx.fillRect(0, 0, this.width(), this.height());
    ctx.font = FONT;
    ctx.fillStyle = AXIS;
    ctx.strokeStyle = AXIS;
    ctx.lineWidth = 1;

    this.drawTimeAxis(ctx, l, win, fps);
    const columns = Math.floor(l.width);

    if (mode === 'heatmap') {
      const rows = Math.max(series.length, 1);
      const rowH = l.height / rows;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      series.forEach((s, r) => {
        const { min } = decimate(s.values, win, columns);
        for (let c = 0; c < columns; c++) {
          if (Number.isNaN(min[c])) continue;
          ctx.fillStyle = confidenceColor(min[c]);
          ctx.fillRect(l.left + c, l.top + r * rowH, 1.5, Math.ceil(rowH));
        }
        if (rowH >= 9) {
          ctx.fillStyle = AXIS;
          ctx.fillText(truncate(s.label, 14), l.left - 6, l.top + r * rowH + rowH / 2);
        }
      });
      this.drawFrame(ctx, l);
      this.drawColorBar(ctx, l);
      return;
    }

    const y = (v: number) => l.top + (1 - (v - lo) / (hi - lo)) * l.height;
    this.drawValueAxis(ctx, l, lo, hi, y, yTitle);

    ctx.save();
    ctx.beginPath();
    ctx.rect(l.left, l.top, l.width, l.height);
    ctx.clip();
    ctx.lineWidth = series.length > 6 ? 1.25 : 2;
    ctx.lineJoin = 'round';
    series.forEach((s, i) => {
      ctx.strokeStyle = colors[i];
      ctx.setLineDash(s.dashed ? [6, 4] : []);
      ctx.beginPath();
      if (wrap || win.end - win.start <= columns) {
        traceFrames(ctx, s.values, win, l, y, wrap);
      } else {
        traceColumns(ctx, decimate(s.values, win, columns), l, y);
      }
      ctx.stroke();
    });
    ctx.restore();
    this.drawFrame(ctx, l);
  }

  /**
   * Vertical confidence colour bar right of the heatmap, 1 at the top, with
   * numbered ticks (as pyqtgraph's ColorBarItem), so the colours read as values.
   */
  private drawColorBar(ctx: CanvasRenderingContext2D, l: Layout) {
    const x = l.left + l.width + COLORBAR_GAP;
    const h = l.height;
    for (let py = 0; py < h; py++) {
      ctx.fillStyle = confidenceColor(1 - py / Math.max(h - 1, 1));
      ctx.fillRect(x, l.top + py, COLORBAR_WIDTH, 1);
    }
    ctx.strokeStyle = AXIS;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, l.top + 0.5, COLORBAR_WIDTH - 1, h - 1);
    ctx.fillStyle = AXIS;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.beginPath();
    const right = x + COLORBAR_WIDTH;
    const ticks = h >= 140 ? COLORBAR_TICKS : COLORBAR_TICKS_SHORT;
    for (const v of ticks) {
      const py = Math.round(l.top + (1 - v) * (h - 1)) + 0.5;
      ctx.moveTo(right, py);
      ctx.lineTo(right + 4, py);
      ctx.fillText(v === 0 || v === 1 ? String(v) : v.toFixed(2), right + 6, py);
    }
    ctx.stroke();
    ctx.save();
    ctx.translate(this.width() - 6, l.top + h / 2);
    ctx.rotate(Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('likelihood', 0, 0);
    ctx.restore();
  }

  private drawFrame(ctx: CanvasRenderingContext2D, l: Layout) {
    ctx.strokeStyle = AXIS;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(l.left - 0.5, l.top);
    ctx.lineTo(l.left - 0.5, l.top + l.height + 0.5);
    ctx.lineTo(l.left + l.width, l.top + l.height + 0.5);
    ctx.stroke();
  }

  private drawTimeAxis(
    ctx: CanvasRenderingContext2D,
    l: Layout,
    win: FrameWindow,
    fps: number | null,
  ) {
    const scale = fps && fps > 0 ? fps : 1;
    const t0 = win.start / scale;
    const t1 = win.end / scale;
    const target = Math.max(2, Math.floor(l.width / 90));
    const step = fps ? niceStep((t1 - t0) / target) : Math.max(1, niceStep((t1 - t0) / target));
    const ticks = niceTicks(t0, t1, target);
    const baseY = l.top + l.height;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.beginPath();
    for (const t of ticks) {
      const x = Math.round(l.left + xOfFrame(t * scale - 0.5, l.width, win)) + 0.5;
      if (x < l.left - 1 || x > l.left + l.width + 1) continue;
      ctx.moveTo(x, baseY);
      ctx.lineTo(x, baseY + 5);
      ctx.fillText(fps ? formatTime(t, step) : String(Math.round(t)), x, baseY + 7);
    }
    ctx.stroke();
  }

  private drawValueAxis(
    ctx: CanvasRenderingContext2D,
    l: Layout,
    lo: number,
    hi: number,
    y: (v: number) => number,
    title: string,
  ) {
    const ticks = niceTicks(lo, hi, Math.max(2, Math.floor(l.height / 35)));
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.beginPath();
    for (const v of ticks) {
      const py = Math.round(y(v)) + 0.5;
      ctx.moveTo(l.left - 5, py);
      ctx.lineTo(l.left, py);
      ctx.fillText(formatValue(v), l.left - 7, py);
    }
    ctx.stroke();
    if (title) {
      ctx.save();
      ctx.translate(11, l.top + l.height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText(truncate(title, 28), 0, 0);
      ctx.restore();
    }
  }

  private drawOverlay() {
    const win = this.view();
    const l = this.layout();
    const frame = this.currentFrame();
    const segment = this.dragPreview() ?? this.segment();
    if (this.width() <= 0) return;
    const ctx = this.prepare(this.overlayCanvas().nativeElement);
    if (!ctx) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(l.left, l.top, l.width, l.height);
    ctx.clip();
    if (segment) {
      const x0 = l.left + xOfFrame(segment[0] - 0.5, l.width, win);
      const x1 = l.left + xOfFrame(segment[1] - 0.5, l.width, win);
      ctx.fillStyle = REGION_FILL;
      ctx.fillRect(x0, l.top, Math.max(x1 - x0, 1), l.height);
      ctx.fillStyle = REGION_EDGE;
      ctx.fillRect(x0, l.top, 1, l.height);
      ctx.fillRect(x1 - 1, l.top, 1, l.height);
    }
    if (frame >= win.start && frame < win.end) {
      ctx.fillStyle = TIME_MARKER;
      ctx.fillRect(Math.round(l.left + xOfFrame(frame, l.width, win)) - 1, l.top, 2, l.height);
    }
    ctx.restore();
  }
}

/**
 * One vertex per frame. Used when zoomed in (fewer frames than pixels), and
 * always for angles, so a seam crossing is never mistaken for a jump.
 *
 * With `wrap`, a step that crosses ±180° with a small real change (see
 * crossesWrap) runs to the edge and continues from the opposite edge. A large
 * step is drawn as a jump, because that is usually a tracking error.
 */
function traceFrames(
  ctx: CanvasRenderingContext2D,
  values: ArrayLike<number | null>,
  win: FrameWindow,
  l: Layout,
  y: (v: number) => number,
  wrap: boolean,
) {
  let prev: number | null = null;
  let prevX = 0;
  const from = Math.max(0, win.start - 1);
  const to = Math.min(values.length, win.end + 1);
  for (let f = from; f < to; f++) {
    const v = values[f];
    if (v == null || !Number.isFinite(v)) {
      prev = null;
      continue;
    }
    const x = l.left + xOfFrame(f, l.width, win);
    if (prev == null) {
      ctx.moveTo(x, y(v));
    } else if (wrap && crossesWrap(prev, v)) {
      const edge = prev > 0 ? 180 : -180;
      const unwrapped = v - Math.sign(v - prev) * 360;
      const t = (edge - prev) / (unwrapped - prev);
      const xc = prevX + t * (x - prevX);
      ctx.lineTo(xc, y(edge));
      ctx.moveTo(xc, y(-edge));
      ctx.lineTo(x, y(v));
    } else {
      ctx.lineTo(x, y(v));
    }
    prev = v;
    prevX = x;
  }
}

/** Min–max per pixel column (zoomed out: more frames than pixels). */
function traceColumns(
  ctx: CanvasRenderingContext2D,
  { min, max }: { min: Float64Array; max: Float64Array },
  l: Layout,
  y: (v: number) => number,
) {
  let drawing = false;
  for (let c = 0; c < min.length; c++) {
    if (Number.isNaN(min[c])) {
      drawing = false;
      continue;
    }
    const x = l.left + c + 0.5;
    if (!drawing) ctx.moveTo(x, y(max[c]));
    else ctx.lineTo(x, y(max[c]));
    ctx.lineTo(x, y(min[c]));
    drawing = true;
  }
}

function formatValue(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1).replace(/\.0$/, '');
  if (a >= 1) return v.toFixed(2).replace(/\.?0+$/, '');
  if (a === 0) return '0';
  return v.toPrecision(2);
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}
