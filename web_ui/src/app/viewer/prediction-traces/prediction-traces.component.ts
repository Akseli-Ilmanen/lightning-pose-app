import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
  linkedSignal,
  output,
  signal,
} from '@angular/core';
import { RpcService } from '../../rpc.service';
import {
  MAX_TRACES,
  PredictionTracesSettingsService,
} from './prediction-traces-settings.service';
import {
  FrameSegment,
  FrameWindow,
} from '../../components/time-series-plot/time-series-math';
import { ListPoseSignalsResponse, SegmentExtractRequest, TraceSource } from './pose-signals';
import { TraceRowComponent } from './trace-row.component';

/**
 * Prediction time series under the Viewer's videos (GitHub issue #206).
 *
 * Up to three stacked rows (TraceRowComponent), each with its own signal, on
 * one shared time axis: zoom, time marker, centring and the marked segment are
 * shared. Click a trace to jump to a frame; drag to mark a segment and extract
 * frames from it.
 */
@Component({
  selector: 'app-prediction-traces',
  imports: [TraceRowComponent],
  templateUrl: './prediction-traces.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block' },
})
export class PredictionTracesComponent {
  /** Loaded predictions for the session: one per shown model and view. */
  sources = input.required<TraceSource[]>();
  nFrames = input.required<number>();
  fps = input<number | null>(null);
  currentFrame = input(0);
  likelihoodThreshold = input(0.9);
  /** A label file is chosen and the player is paused. */
  canExtract = input(false);
  extracting = input(false);

  seek = output<number>();
  extractSegment = output<SegmentExtractRequest>();

  private rpc = inject(RpcService);
  /** The user's choices; they outlive this panel (closing it, other sessions, reloads). */
  private settings = inject(PredictionTracesSettingsService);

  protected readonly maxTraces = MAX_TRACES;
  protected rows = this.settings.traces;
  /** The movement signals the server offers; fetched once for all rows. */
  protected movement = signal<ListPoseSignalsResponse | null>(null);

  /** Time zoom shared by all rows. Whole video again when the video changes. */
  protected xWindow = linkedSignal<number, FrameWindow | null>({
    source: this.nFrames,
    computation: () => null,
  });

  protected followPlayback = this.settings.followPlayback;

  protected setFollowPlayback(on: boolean) {
    this.settings.update({ followPlayback: on });
  }

  protected addTrace() {
    this.settings.addTrace();
  }

  protected removeTrace(index: number) {
    this.settings.removeTrace(index);
  }

  // Segment and extraction ---------------------------------------------------

  /** Marked segment [start, end), shared by all rows. Cleared when the video changes. */
  segment = linkedSignal<number, FrameSegment | null>({
    source: this.nFrames,
    computation: () => null,
  });
  /**
   * Marking a segment to extract from: left-drag marks instead of panning and
   * the segment controls show. Off again when the video changes.
   */
  marking = linkedSignal<number, boolean>({
    source: this.nFrames,
    computation: () => false,
  });

  protected startMarking() {
    this.marking.set(true);
  }

  protected stopMarking() {
    this.marking.set(false);
    this.segment.set(null);
  }

  protected extractCount = this.settings.extractCount;
  /** How frames are picked from the segment. */
  protected selection = this.settings.selection;

  protected setSelection(value: 'kmeans' | 'uniform') {
    this.settings.update({ selection: value });
  }

  protected segmentLength = computed(() => {
    const s = this.segment();
    return s ? s[1] - s[0] : 0;
  });
  protected coveragePercent = computed(() => {
    const len = this.segmentLength();
    return len > 0 ? (100 * this.extractCount()) / len : 0;
  });
  protected extractProblem = computed(() => {
    const len = this.segmentLength();
    const n = this.extractCount();
    if (!Number.isInteger(n) || n < 1) return 'Choose at least one frame.';
    if (n > len) return `The segment has only ${len} frames.`;
    return null;
  });

  constructor() {
    this.rpc
      .call('listPoseSignals', {})
      .then((r) => this.movement.set(r as ListPoseSignalsResponse))
      .catch(() => this.movement.set(null));
  }

  protected setSegmentStart() {
    const frame = this.currentFrame();
    const end = this.segment()?.[1] ?? this.nFrames();
    this.segment.set(frame < end ? [frame, end] : [frame, frame + 1]);
  }

  protected setSegmentEnd() {
    const frame = this.currentFrame();
    const start = this.segment()?.[0] ?? 0;
    this.segment.set(frame >= start ? [start, frame + 1] : [frame, frame + 1]);
  }

  protected onExtractCountInput(value: string) {
    this.settings.update({ extractCount: Math.floor(Number(value)) });
  }

  protected onExtractClick() {
    const segment = this.segment();
    if (!segment || this.extractProblem() || !this.canExtract()) return;
    this.extractSegment.emit({
      segment,
      nFrames: this.extractCount(),
      selection: this.selection(),
    });
  }
}
