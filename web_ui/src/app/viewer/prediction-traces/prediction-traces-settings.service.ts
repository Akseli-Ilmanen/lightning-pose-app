import { computed, effect, Injectable, signal } from '@angular/core';
import { HeadingOverlay } from './pose-signals';

const STORAGE_KEY = 'prediction-traces.settings';

/** Up to this many stacked trace rows. */
export const MAX_TRACES = 3;

/** What one trace row shows. */
export interface TraceChoices {
  signalId: string;
  /** Keypoints to plot. null until the user changes it (then the first keypoint). */
  selectedKeypoints: string[] | null;
  spaceChoice: 'xy' | 'x' | 'y';
  maskLowConfidence: boolean;
  /** Preferred model/view, `${modelKey}/${viewName}`. */
  sourceKey: string | null;
  /** Arguments per movement signal, e.g. head direction's left/right keypoint. Per row. */
  paramsBySignal: Record<string, Record<string, unknown>>;
  /** Fixed y range per signal; absent means autoscale. Per row. */
  yLimitsBySignal: Record<string, [number, number]>;
}

/** What the user chose in the Prediction traces panel. */
export interface PredictionTracesChoices {
  /** Stacked rows, 1 to MAX_TRACES. */
  traces: TraceChoices[];
  extractCount: number;
  selection: 'kmeans' | 'uniform';
  /** Keep the time marker centred and slide the traces under it (ethograph's centre mode). */
  followPlayback: boolean;
}

export const DEFAULT_TRACE: TraceChoices = {
  signalId: 'position',
  selectedKeypoints: null,
  spaceChoice: 'xy',
  maskLowConfidence: false,
  sourceKey: null,
  paramsBySignal: {},
  yLimitsBySignal: {},
};

/** Arrow colour per row, so several head-direction rows can be told apart on the video. */
export const HEADING_COLORS = ['#ffd60a', '#22d3ee', '#f472b6'] as const;

export const DEFAULT_CHOICES: PredictionTracesChoices = {
  traces: [DEFAULT_TRACE],
  extractCount: 10,
  selection: 'kmeans',
  followPlayback: true,
};

/**
 * The Prediction traces choices, kept for the whole app and saved in the
 * browser. Closing the panel, opening another session or reloading the page
 * keeps them. (The marked segment and zoom are per video and live in the panel.)
 */
@Injectable({ providedIn: 'root' })
export class PredictionTracesSettingsService {
  readonly choices = signal<PredictionTracesChoices>(load());

  readonly traces = computed(() => this.choices().traces);
  readonly extractCount = computed(() => this.choices().extractCount);
  readonly selection = computed(() => this.choices().selection);
  readonly followPlayback = computed(() => this.choices().followPlayback);

  /**
   * One head-direction arrow per row showing head direction, with that row's
   * keypoints, source and colour. `modelKey` comes from the row's source; the
   * Viewer falls back to the first shown model.
   */
  readonly headings = computed<
    (Omit<HeadingOverlay, 'modelKey'> & { modelKey: string | null; row: number; color: string })[]
  >(() =>
    this.choices().traces.flatMap((row, i) => {
      if (row.signalId !== 'head_direction') return [];
      const p = row.paramsBySignal['head_direction'] ?? {};
      if (typeof p['left_keypoint'] !== 'string' || typeof p['right_keypoint'] !== 'string') {
        return [];
      }
      return [
        {
          row: i,
          color: HEADING_COLORS[i % HEADING_COLORS.length],
          modelKey: row.sourceKey?.split('/')[0] ?? null,
          left: p['left_keypoint'],
          right: p['right_keypoint'],
          cameraView: p['camera_view'] === 'bottom_up' ? ('bottom_up' as const) : ('top_down' as const),
        },
      ];
    }),
  );

  constructor() {
    effect(() => save(this.choices()));
  }

  update(patch: Partial<Omit<PredictionTracesChoices, 'traces'>>) {
    this.choices.update((c) => ({ ...c, ...patch }));
  }

  updateTrace(index: number, patch: Partial<TraceChoices>) {
    this.choices.update((c) => ({
      ...c,
      traces: c.traces.map((t, i) => (i === index ? { ...t, ...patch } : t)),
    }));
  }

  /** Add a row (Confidence when the last one is not, else Position). */
  addTrace() {
    this.choices.update((c) => {
      if (c.traces.length >= MAX_TRACES) return c;
      const last = c.traces[c.traces.length - 1] ?? DEFAULT_TRACE;
      const signalId = last.signalId === 'confidence' ? 'position' : 'confidence';
      return { ...c, traces: [...c.traces, structuredClone({ ...last, signalId })] };
    });
  }

  removeTrace(index: number) {
    this.choices.update((c) =>
      c.traces.length <= 1 ? c : { ...c, traces: c.traces.filter((_, i) => i !== index) },
    );
  }

  setParams(index: number, signalId: string, params: Record<string, unknown>) {
    const t = this.choices().traces[index];
    if (!t) return;
    this.updateTrace(index, { paramsBySignal: { ...t.paramsBySignal, [signalId]: params } });
  }

  setYLimits(index: number, signalId: string, range: [number, number] | null) {
    const t = this.choices().traces[index];
    if (!t) return;
    const yLimitsBySignal = { ...t.yLimitsBySignal };
    if (range) yLimitsBySignal[signalId] = range;
    else delete yLimitsBySignal[signalId];
    this.updateTrace(index, { yLimitsBySignal });
  }
}

/** Saved choices; settings saved before rows existed become row 1. */
function load(): PredictionTracesChoices {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    if (!parsed || typeof parsed !== 'object') return structuredClone(DEFAULT_CHOICES);
    const choices: PredictionTracesChoices = { ...DEFAULT_CHOICES, ...parsed };
    if (!Array.isArray(parsed.traces)) {
      const row: TraceChoices = { ...DEFAULT_TRACE };
      for (const key of Object.keys(DEFAULT_TRACE) as (keyof TraceChoices)[]) {
        if (key in parsed) (row as unknown as Record<string, unknown>)[key] = parsed[key];
      }
      choices.traces = [row];
    }
    // Saved when params / y limits were shared by all rows: each row starts from them.
    const shared = parsed as Record<string, unknown>;
    const sharedParams = isRecord(shared['paramsBySignal']) ? shared['paramsBySignal'] : {};
    const sharedYLimits = isRecord(shared['yLimitsBySignal']) ? shared['yLimitsBySignal'] : {};
    delete (choices as unknown as Record<string, unknown>)['paramsBySignal'];
    delete (choices as unknown as Record<string, unknown>)['yLimitsBySignal'];
    choices.traces = choices.traces
      .slice(0, MAX_TRACES)
      .map((t: Partial<TraceChoices>) => ({
        ...DEFAULT_TRACE,
        ...t,
        paramsBySignal: t.paramsBySignal ?? structuredClone(sharedParams),
        yLimitsBySignal: t.yLimitsBySignal ?? structuredClone(sharedYLimits),
      }));
    if (choices.traces.length === 0) choices.traces = [{ ...DEFAULT_TRACE }];
    return choices;
  } catch {
    return structuredClone(DEFAULT_CHOICES);
  }
}

function save(choices: PredictionTracesChoices) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(choices));
  } catch {
    // Storage blocked: choices still hold for this page.
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isRecord(v: unknown): v is Record<string, any> {
  return !!v && typeof v === 'object' && !Array.isArray(v);
}
