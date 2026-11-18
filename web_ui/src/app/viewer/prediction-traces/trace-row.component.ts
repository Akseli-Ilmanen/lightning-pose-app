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
  /** The view (camera) the series comes from. */
  view: string;
}

/**
 * One stacked row of the Prediction traces: its own signal, views, keypoints,
 * axes and y range, on the time axis, time marker and segment it shares with the other
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

  /** Shown models, in the Viewer's order. */
  protected models = computed(() => [...new Set(this.sources().map((s) => s.modelKey))]);
  /** Chosen model: the saved one when it is shown, else the first. */
  protected modelKey = computed(() => {
    const models = this.models();
    const saved = this.choice().modelKey;
    return saved != null && models.includes(saved) ? saved : (models[0] ?? null);
  });
  /** The chosen model's predictions, one per view. */
  private modelSources = computed(() =>
    this.sources().filter((s) => s.modelKey === this.modelKey()),
  );
  protected views = computed(() => this.modelSources().map((s) => s.viewName));
  /**
   * Views to plot, ticked like the keypoints. Starts with every view, so one
   * keypoint can be compared across cameras without choosing one first. Saved
   * names this session lacks are skipped; if none are left, every view is used.
   */
  protected selectedViews = computed(() => {
    const all = this.views();
    const saved = this.choice().selectedViews;
    if (saved == null) return all;
    const kept = all.filter((v) => saved.includes(v));
    return kept.length > 0 || saved.length === 0 ? kept : all;
  });
  private selectedSources = computed(() => {
    const selected = new Set(this.selectedViews());
    return this.modelSources().filter((s) => selected.has(s.viewName));
  });
  /** Keypoints of the chosen model, over all its views. */
  protected keypointChoices = computed(() => [
    ...new Set(this.modelSources().flatMap((s) => s.keypoints)),
  ]);

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
    const keypoints = this.keypointChoices();
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

  /** Server responses by request key (one request per view). */
  private results = signal<Record<string, ComputePoseSignalResponse>>({});
  private pending = signal(0);
  /** Request keys sent and not yet answered. */
  private inFlight = new Set<string>();
  protected loading = computed(() => this.pending() > 0);
  protected error = signal<string | null>(null);

  /** Identifies the request the current choices would send for one view. */
  private requestKey(source: TraceSource): string {
    return JSON.stringify([
      source.path,
      this.signalId(),
      this.params(),
      this.maskLowConfidence() ? this.likelihoodThreshold() : null,
    ]);
  }
  /** Ticked views whose derived signal has not been computed with the current choices. */
  private missingSources = computed(() => {
    if (this.isBuiltin() || this.movementSignal() == null) return [];
    const results = this.results();
    return this.selectedSources().filter((s) => !(this.requestKey(s) in results));
  });
  protected isStale = computed(() => this.missingSources().length > 0);

  // Which keypoints and axes to draw ------------------------------------------

  /**
   * Keypoints to plot. Starts with only the first keypoint, so the first plot
   * is readable. Saved names this file lacks are skipped; if none are left,
   * the first keypoint is used.
   */
  protected selectedKeypoints = computed(() => {
    const all = this.keypointChoices();
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

  /** Every series of the chosen signal in the ticked views, before keypoint/axis filtering. */
  private allSeries = computed<TaggedSeries[]>(() => {
    const id = this.signalId();
    const results = this.results();
    // Position is masked here; derived signals are masked on the server.
    const threshold = this.maskLowConfidence() ? this.likelihoodThreshold() : null;
    return this.selectedSources().flatMap((source): TaggedSeries[] => {
      const view = source.viewName;
      if (id === 'position') {
        return source.keypoints.flatMap((kp) => {
          const likelihood = threshold == null ? null : source.column(kp, 'likelihood');
          return (['x', 'y'] as const).flatMap((axis) => {
            const column = source.column(kp, axis);
            if (!column) return [];
            const values =
              likelihood && threshold != null
                ? Array.from(column, (v, i) => (likelihood[i] < threshold ? null : v))
                : column;
            return [{ label: `${kp} ${axis}`, values, keypoint: kp, space: axis, view }];
          });
        });
      }
      if (id === 'confidence') {
        return source.keypoints.flatMap((kp) => {
          const values = source.column(kp, 'likelihood');
          return values ? [{ label: kp, values, keypoint: kp, view }] : [];
        });
      }
      return (results[this.requestKey(source)]?.series ?? []).map((s) => ({ ...s, view }));
    });
  });

  /** The signal has one series per keypoint: offer the Keypoints dropdown. */
  protected hasKeypoints = computed(() => this.allSeries().some((s) => s.keypoint));
  /** The signal has x and y series: offer the axis choice. */
  protected hasSpace = computed(() => this.allSeries().some((s) => s.space));

  /**
   * What a colour stands for. One view: the keypoint. Several views and one
   * keypoint (or a signal without keypoints): the view. Several of both: each
   * view and keypoint pair.
   */
  protected colorBy = computed<'keypoint' | 'view' | 'pair'>(() => {
    if (this.selectedViews().length <= 1) return 'keypoint';
    return !this.hasKeypoints() || this.selectedKeypoints().length <= 1 ? 'view' : 'pair';
  });

  protected series = computed<TimeSeries[]>(() => {
    const keypoints = this.keypointChoices();
    const views = this.views();
    const selected = new Set(this.selectedKeypoints());
    const spaces = this.spaceChoice();
    const oneAxis = spaces !== 'xy';
    const colorBy = this.colorBy();
    return (
      this.allSeries()
        .filter((s) => !s.keypoint || selected.has(s.keypoint))
        .filter((s) => !s.space || !oneAxis || s.space === spaces)
        .map((s) => ({
          s,
          kp: s.keypoint ? keypoints.indexOf(s.keypoint) : -1,
          view: views.indexOf(s.view),
        }))
        // A keypoint's views sit next to each other (the sort is stable).
        .sort((a, b) => a.kp - b.kp)
        .map(({ s, kp, view }) => {
          const name = oneAxis && s.keypoint ? s.keypoint : s.label;
          const index =
            colorBy === 'view' ? view : colorBy === 'pair' ? kp * views.length + view : kp;
          return {
            label:
              colorBy === 'keypoint'
                ? name
                : colorBy === 'view' && s.keypoint
                  ? // The one keypoint is named in the Keypoints dropdown.
                    s.space && !oneAxis
                    ? `${s.view} ${s.space}`
                    : s.view
                  : `${s.view} · ${name}`,
            values: s.values,
            // A colour keeps its meaning whatever is ticked; y is dashed.
            color: index >= 0 ? TRACE_COLORS[index % TRACE_COLORS.length] : undefined,
            dashed: s.space === 'y' && !oneAxis,
          };
        })
    );
  });

  protected viewMenuLabel = computed(() => {
    const n = this.selectedViews().length;
    const all = this.views().length;
    return n === all ? 'All views' : n === 1 ? this.selectedViews()[0] : `${n} of ${all} views`;
  });

  protected isViewSelected(view: string): boolean {
    return this.selectedViews().includes(view);
  }

  protected toggleView(view: string, checked: boolean) {
    const next = new Set(this.selectedViews());
    if (checked) next.add(view);
    else next.delete(view);
    this.setSelectedViews(this.views().filter((v) => next.has(v)));
  }

  protected selectAllViews(all: boolean) {
    this.setSelectedViews(all ? [...this.views()] : []);
  }

  private setSelectedViews(selectedViews: string[]) {
    this.settings.updateTrace(this.index(), { selectedViews });
    this.replotIfDerived();
  }

  protected viewColor(view: string): string {
    const index = this.views().indexOf(view);
    return index >= 0 ? TRACE_COLORS[index % TRACE_COLORS.length] : 'transparent';
  }

  protected keypointMenuLabel = computed(() => {
    const n = this.selectedKeypoints().length;
    const all = this.keypointChoices().length;
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
    const all = this.keypointChoices();
    const next = new Set(this.selectedKeypoints());
    if (checked) next.add(kp);
    else next.delete(kp);
    this.settings.updateTrace(this.index(), {
      selectedKeypoints: all.filter((k) => next.has(k)),
    });
  }

  protected selectAllKeypoints(all: boolean) {
    this.settings.updateTrace(this.index(), {
      selectedKeypoints: all ? [...this.keypointChoices()] : [],
    });
  }

  protected keypointColor(kp: string): string {
    const index = this.keypointChoices().indexOf(kp);
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
    const results = this.results();
    const computed = this.selectedSources()
      .map((s) => results[this.requestKey(s)])
      .find((r) => r != null);
    return computed?.unit ?? this.movementSignal()?.unit ?? '';
  });
  protected plotLabel = computed(
    () => `${this.yLabel()} over time. Click to jump to a frame, drag to mark a segment.`,
  );

  constructor() {
    // Movement signals listed, a session or model/view loaded, or the row
    // reopened with a saved derived signal: (re)compute it.
    effect(() => {
      this.movementSignal();
      this.selectedSources();
      untracked(() => {
        this.saveParams();
        if (this.isStale()) this.replotIfDerived();
      });
    });
  }

  protected onModelChange(modelKey: string) {
    this.settings.updateTrace(this.index(), { modelKey });
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

  /** Compute the chosen movement signal on the server, for each ticked view that lacks it. */
  async plot() {
    const info = this.movementSignal();
    const projectKey = this.projectInfoService.projectContext()?.key;
    const missing = this.missingSources();
    if (!info || !projectKey || missing.length === 0) return;
    // Forget results the current choices no longer ask for.
    const wanted = new Set(this.selectedSources().map((s) => this.requestKey(s)));
    this.results.update((r) =>
      Object.fromEntries(Object.entries(r).filter(([key]) => wanted.has(key))),
    );
    this.error.set(null);
    await Promise.all(
      missing.map(async (source) => {
        const key = this.requestKey(source);
        // Already asked for (a view ticked while another is still computing).
        if (this.inFlight.has(key)) return;
        this.inFlight.add(key);
        const request: ComputePoseSignalRequest = {
          projectKey,
          predictionPath: source.path,
          signal: info.id,
          params: this.params(),
          fps: this.fps(),
          confidenceThreshold: this.maskLowConfidence() ? this.likelihoodThreshold() : null,
        };
        this.pending.update((n) => n + 1);
        try {
          const response = (await this.rpc.call(
            'computePoseSignal',
            request,
          )) as ComputePoseSignalResponse;
          this.results.update((r) => ({ ...r, [key]: response }));
        } catch (err) {
          const detail = (err as { error?: { detail?: unknown } })?.error?.detail;
          this.error.set(
            typeof detail === 'string' ? detail : `Could not compute ${info.label}.`,
          );
        } finally {
          this.inFlight.delete(key);
          this.pending.update((n) => n - 1);
        }
      }),
    );
  }

  private defaultParams(info: PoseSignalInfo | null): Record<string, unknown> {
    const keypoints = this.keypointChoices();
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
