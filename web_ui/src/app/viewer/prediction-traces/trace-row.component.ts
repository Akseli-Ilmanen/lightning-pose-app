import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  input,
  model,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import { RpcService } from '../../rpc.service';
import { ProjectInfoService } from '../../project-info.service';
import {
  STACKED_GUTTERS,
  TimeSeriesPlotComponent,
} from '../../components/time-series-plot/time-series-plot.component';
import {
  FrameSegment,
  FrameWindow,
  TimeSeries,
  TRACE_COLORS,
} from '../../components/time-series-plot/time-series-math';
import {
  DEFAULT_TRACE,
  PredictionTracesSettingsService,
} from './prediction-traces-settings.service';
import {
  BUILTIN_SIGNALS,
  ComputePoseSignalRequest,
  ComputePoseSignalResponse,
  ListPoseSignalsResponse,
  PoseSignalInfo,
  TraceSource,
} from './pose-signals';

const BUILTIN_IDS: readonly string[] = BUILTIN_SIGNALS.map((s) => s.id);

/** A series with the keypoint and axis it belongs to, for filtering. */
interface TaggedSeries {
  label: string;
  values: ArrayLike<number | null>;
  keypoint?: string | null;
  space?: string | null;
}

/**
 * One stacked row of the Prediction traces: its own signal, keypoints, axes and
 * y range, on the time axis, time marker and segment it shares with the other
 * rows. Its choices are row `index` in PredictionTracesSettingsService.
 */
@Component({
  selector: 'app-trace-row',
  imports: [TimeSeriesPlotComponent],
  templateUrl: './trace-row.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block' },
})
export class TraceRowComponent {
  /** Which saved row this is. */
  index = input.required<number>();
  sources = input.required<TraceSource[]>();
  nFrames = input.required<number>();
  fps = input<number | null>(null);
  currentFrame = input(0);
  likelihoodThreshold = input(0.9);
  /** The movement signals the server offers (listPoseSignals), or null. */
  movement = input<ListPoseSignalsResponse | null>(null);
  /** Rows after the first can be closed. */
  removable = input(false);
  /** Keep the time marker centred (shared by all rows). */
  follow = input(false);
  /** Left-drag marks the segment instead of panning. */
  markSegment = input(false);
  /** Zoomed time window, shared by all rows. */
  xWindow = model<FrameWindow | null>(null);
  /** Marked segment, shared by all rows. */
  segment = model<FrameSegment | null>(null);

  seek = output<number>();
  remove = output<void>();

  private rpc = inject(RpcService);
  private projectInfoService = inject(ProjectInfoService);
  private settings = inject(PredictionTracesSettingsService);

  protected readonly builtinSignals = BUILTIN_SIGNALS;
  protected readonly gutters = STACKED_GUTTERS;

  /** This row's saved choices. */
  private choice = computed(() => this.settings.traces()[this.index()] ?? DEFAULT_TRACE);

  /** Chosen model/view: the saved one when it is loaded, else the first. */
  protected sourceKey = computed(() => {
    const sources = this.sources();
    const saved = this.choice().sourceKey;
    return sources.some((s) => s.key === saved) ? saved : (sources[0]?.key ?? null);
  });
  protected source = computed(
    () => this.sources().find((s) => s.key === this.sourceKey()) ?? null,
  );

  protected signalId = computed(() => this.choice().signalId);
  protected movementSignals = computed(() => this.movement()?.signals ?? []);
  protected movementSignal = computed<PoseSignalInfo | null>(
    () => this.movementSignals().find((s) => s.id === this.signalId()) ?? null,
  );
  protected isBuiltin = computed(() => BUILTIN_IDS.includes(this.signalId()));

  /**
   * Arguments of the chosen movement signal: the saved ones (when they still
   * name keypoints in this file), else defaults.
   */
  protected params = computed<Record<string, unknown>>(() => {
    const info = this.movementSignal();
    const defaults = this.defaultParams(info);
    const saved = info ? this.choice().paramsBySignal[info.id] : undefined;
    if (!info || !saved) return defaults;
    const keypoints = this.source()?.keypoints ?? [];
    const out = { ...defaults };
    for (const p of info.params) {
      const v = saved[p.name];
      if (v == null) continue;
      if (p.kind === 'keypoint' && !keypoints.includes(String(v))) continue;
      if (p.kind === 'choice' && !(p.choices ?? []).includes(String(v))) continue;
      out[p.name] = v;
    }
    return out;
  });
  protected maskLowConfidence = computed(() => this.choice().maskLowConfidence);

  private result = signal<{ key: string; response: ComputePoseSignalResponse } | null>(
    null,
  );
  protected loading = signal(false);
  protected error = signal<string | null>(null);

  /** Identifies the request the current choices would send. */
  private requestKey = computed(() =>
    JSON.stringify([
      this.source()?.path ?? null,
      this.signalId(),
      this.params(),
      this.maskLowConfidence() ? this.likelihoodThreshold() : null,
    ]),
  );
  protected isStale = computed(
    () =>
      !this.isBuiltin() &&
      this.movementSignal() != null &&
      this.result()?.key !== this.requestKey(),
  );

  // Which keypoints and axes to draw ------------------------------------------

  /**
   * Keypoints to plot. Starts with only the first keypoint, so the first plot
   * is readable. Saved names this file lacks are skipped; if none are left,
   * the first keypoint is used.
   */
  protected selectedKeypoints = computed(() => {
    const all = this.source()?.keypoints ?? [];
    const saved = this.choice().selectedKeypoints;
    if (saved == null) return all.slice(0, 1);
    const kept = all.filter((k) => saved.includes(k));
    return kept.length > 0 || saved.length === 0 ? kept : all.slice(0, 1);
  });
  /** Axes to plot for per-axis signals (position, velocity). */
  protected spaceChoice = computed(() => this.choice().spaceChoice);

  protected setSpaceChoice(value: 'xy' | 'x' | 'y') {
    this.settings.updateTrace(this.index(), { spaceChoice: value });
  }

  /** Every series of the chosen signal, before keypoint/axis filtering. */
  private allSeries = computed<TaggedSeries[]>(() => {
    const source = this.source();
    if (!source) return [];
    const id = this.signalId();
    if (id === 'position') {
      return source.keypoints.flatMap((kp) =>
        (['x', 'y'] as const).flatMap((axis) => {
          const values = source.column(kp, axis);
          return values ? [{ label: `${kp} ${axis}`, values, keypoint: kp, space: axis }] : [];
        }),
      );
    }
    if (id === 'confidence') {
      return source.keypoints.flatMap((kp) => {
        const values = source.column(kp, 'likelihood');
        return values ? [{ label: kp, values, keypoint: kp }] : [];
      });
    }
    const result = this.result();
    return result && result.key === this.requestKey() ? result.response.series : [];
  });

  /** The signal has one series per keypoint: offer the Keypoints dropdown. */
  protected hasKeypoints = computed(() => this.allSeries().some((s) => s.keypoint));
  /** The signal has x and y series: offer the axis choice. */
  protected hasSpace = computed(() => this.allSeries().some((s) => s.space));

  protected series = computed<TimeSeries[]>(() => {
    const keypoints = this.source()?.keypoints ?? [];
    const selected = new Set(this.selectedKeypoints());
    const spaces = this.spaceChoice();
    const oneAxis = spaces !== 'xy';
    return this.allSeries()
      .filter((s) => !s.keypoint || selected.has(s.keypoint))
      .filter((s) => !s.space || !oneAxis || s.space === spaces)
      .map((s) => {
        const index = s.keypoint ? keypoints.indexOf(s.keypoint) : -1;
        return {
          label: oneAxis && s.keypoint ? s.keypoint : s.label,
          values: s.values,
          // One colour per keypoint whatever is selected; y is dashed.
          color: index >= 0 ? TRACE_COLORS[index % TRACE_COLORS.length] : undefined,
          dashed: s.space === 'y' && !oneAxis,
        };
      });
  });

  protected keypointMenuLabel = computed(() => {
    const n = this.selectedKeypoints().length;
    const all = this.source()?.keypoints.length ?? 0;
    return n === all
      ? 'All keypoints'
      : n === 1
        ? this.selectedKeypoints()[0]
        : `${n} of ${all} keypoints`;
  });

  protected isKeypointSelected(kp: string): boolean {
    return this.selectedKeypoints().includes(kp);
  }

  protected toggleKeypoint(kp: string, checked: boolean) {
    const all = this.source()?.keypoints ?? [];
    const next = new Set(this.selectedKeypoints());
    if (checked) next.add(kp);
    else next.delete(kp);
    this.settings.updateTrace(this.index(), {
      selectedKeypoints: all.filter((k) => next.has(k)),
    });
  }

  protected selectAllKeypoints(all: boolean) {
    this.settings.updateTrace(this.index(), {
      selectedKeypoints: all ? [...(this.source()?.keypoints ?? [])] : [],
    });
  }

  protected keypointColor(kp: string): string {
    const index = this.source()?.keypoints.indexOf(kp) ?? -1;
    return index >= 0 ? TRACE_COLORS[index % TRACE_COLORS.length] : 'transparent';
  }

  protected mode = computed(() =>
    this.signalId() === 'confidence' ? 'heatmap' : 'lines',
  );
  protected plotHeight = computed(() =>
    this.mode() === 'heatmap'
      ? Math.min(Math.max(this.series().length * 16 + 30, 80), 300)
      : 180,
  );
  protected yLabel = computed(
    () =>
      BUILTIN_SIGNALS.find((s) => s.id === this.signalId())?.label ??
      this.movementSignal()?.label ??
      '',
  );
  protected wrapAngles = computed(() => this.signalId() === 'head_direction');

  // Value-axis range ---------------------------------------------------------

  private plotView = viewChild(TimeSeriesPlotComponent);
  /** Fixed y range for the chosen signal (saved per signal), or null to autoscale. */
  protected yLimits = computed<[number, number] | null>(
    () => this.choice().yLimitsBySignal[this.signalId()] ?? null,
  );
  /** Range on screen, for the min/max fields (autoscaled values when auto). */
  protected shownYRange = computed(() => this.yLimits() ?? this.plotView()?.yRange() ?? null);

  protected setYLimits(range: [number, number] | null) {
    this.settings.setYLimits(this.index(), this.signalId(), range);
  }

  /** A typed min (0) or max (1). Ignored unless min < max. */
  protected onYLimitChange(which: 0 | 1, text: string) {
    const value = Number(text);
    const current = this.shownYRange();
    if (text.trim() === '' || !Number.isFinite(value) || !current) return;
    const next: [number, number] = which === 0 ? [value, current[1]] : [current[0], value];
    if (next[0] < next[1]) this.setYLimits(next);
  }

  protected onYAutoChange(auto: boolean) {
    if (auto) this.setYLimits(null);
    else {
      const current = this.shownYRange();
      if (current) this.setYLimits([current[0], current[1]]);
    }
  }

  protected formatLimit(v: number | undefined): string {
    if (v == null || !Number.isFinite(v)) return '';
    return String(Number(v.toPrecision(4)));
  }

  protected unit = computed(() => {
    const builtin = BUILTIN_SIGNALS.find((s) => s.id === this.signalId());
    if (builtin) return builtin.unit;
    return this.result()?.response.unit ?? this.movementSignal()?.unit ?? '';
  });
  protected plotLabel = computed(
    () => `${this.yLabel()} over time. Click to jump to a frame, drag to mark a segment.`,
  );

  constructor() {
    // Movement signals listed, a session or model/view loaded, or the row
    // reopened with a saved derived signal: (re)compute it.
    effect(() => {
      this.movementSignal();
      this.source();
      untracked(() => {
        this.saveParams();
        if (this.isStale()) this.replotIfDerived();
      });
    });
  }

  protected onSourceChange(key: string) {
    this.settings.updateTrace(this.index(), { sourceKey: key });
    this.replotIfDerived();
  }

  protected onSignalChange(id: string) {
    this.settings.updateTrace(this.index(), { signalId: id });
    this.error.set(null);
    this.saveParams();
    this.replotIfDerived();
  }

  protected onParamChange(name: string, value: string) {
    const info = this.movementSignal();
    if (!info) return;
    this.settings.setParams(this.index(), info.id, { ...this.params(), [name]: value });
    this.replotIfDerived();
  }

  protected onMaskChange(checked: boolean) {
    this.settings.updateTrace(this.index(), { maskLowConfidence: checked });
    this.replotIfDerived();
  }

  /**
   * Save the chosen signal's arguments (defaults included), so the Viewer can
   * draw the head-direction arrow from the settings even with the panel closed.
   */
  private saveParams() {
    const info = this.movementSignal();
    if (!info) return;
    const params = this.params();
    const saved = this.choice().paramsBySignal[info.id];
    if (JSON.stringify(saved) !== JSON.stringify(params)) {
      this.settings.setParams(this.index(), info.id, params);
    }
  }

  private replotIfDerived() {
    if (!this.isBuiltin() && this.movementSignal()) void this.plot();
  }

  /** Compute the chosen movement signal on the server. */
  async plot() {
    const source = this.source();
    const info = this.movementSignal();
    const projectKey = this.projectInfoService.projectContext()?.key;
    if (!source || !info || !projectKey) return;
    const key = this.requestKey();
    const request: ComputePoseSignalRequest = {
      projectKey,
      predictionPath: source.path,
      signal: info.id,
      params: this.params(),
      fps: this.fps(),
      confidenceThreshold: this.maskLowConfidence() ? this.likelihoodThreshold() : null,
    };
    this.loading.set(true);
    this.error.set(null);
    try {
      const response = (await this.rpc.call(
        'computePoseSignal',
        request,
      )) as ComputePoseSignalResponse;
      // Drop a response the user has already moved on from.
      if (key === this.requestKey()) this.result.set({ key, response });
    } catch (err) {
      const detail = (err as { error?: { detail?: unknown } })?.error?.detail;
      this.error.set(typeof detail === 'string' ? detail : `Could not compute ${info.label}.`);
    } finally {
      this.loading.set(false);
    }
  }

  protected keypointChoices = computed(() => this.source()?.keypoints ?? []);

  private defaultParams(info: PoseSignalInfo | null): Record<string, unknown> {
    const keypoints = this.source()?.keypoints ?? [];
    const out: Record<string, unknown> = {};
    let nextKeypoint = 0;
    for (const p of info?.params ?? []) {
      if (p.kind === 'keypoint') {
        out[p.name] = guessKeypoint(p.name, keypoints) ?? keypoints[nextKeypoint];
        nextKeypoint = Math.min(nextKeypoint + 1, keypoints.length - 1);
      } else if (p.default != null) {
        out[p.name] = p.default;
      }
    }
    return out;
  }

  protected paramValue(name: string): string {
    const v = this.params()[name];
    return v == null ? '' : String(v);
  }
}

/** For head direction, a keypoint named like left_ear / earL for the left param. */
function guessKeypoint(param: string, keypoints: string[]): string | undefined {
  const side = param.startsWith('left') ? 'l' : param.startsWith('right') ? 'r' : null;
  if (!side) return undefined;
  const word = side === 'l' ? 'left' : 'right';
  return (
    keypoints.find((k) => k.toLowerCase().includes(word)) ??
    keypoints.find((k) =>
      new RegExp(`(^|[_\\-\\s])${side}([_\\-\\s]|$)|${side.toUpperCase()}$`).test(k),
    )
  );
}
