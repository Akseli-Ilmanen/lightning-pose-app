import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  ElementRef,
  inject,
  input,
  OnChanges,
  output,
  signal,
  SimpleChanges,
  untracked,
  viewChild,
} from '@angular/core';
import { LabelerViewOptionsService } from '../labeler-view-options.service';
import { FrameView, fv, mvf, MVFrame } from '../frame.model';
import { LKeypoint, lkp } from '../types';
import { DecimalPipe } from '@angular/common';
import { ProjectInfoService } from '../../project-info.service';
import { HorizontalScrollDirective } from '../../infra/horizontal-scroll.directive';
import { Point } from '@angular/cdk/drag-drop';
import { SessionService } from '../../session.service';
import { CameraCalibrationService } from '../../camera-calibration.service';
import { MVLabelFile } from '../../label-file.model';
import { GetMVAutoLabelsResponse } from '../mv-autolabel';
import { BundleAdjustDialogComponent } from '../../bundle-adjust-dialog/bundle-adjust-dialog.component';
import { PathDisplayComponent } from '../../components/path-display/path-display.component';
import { ToastService } from '../../toast.service';
import { firstValueFrom, Subject } from 'rxjs';
import { FormsModule } from '@angular/forms';
import { ImageLabelWidgetComponent } from '../image-label-widget/image-label-widget.component';
import {
  CONTEXT_SLOT_LABELS,
  contextImgPaths,
  contextShowingText,
} from '../temporal-context';
import { LoadCalibrationDialogComponent } from '../../file-manager/load-calibration-dialog.component';
import { SkeletonViewOptionsComponent } from '../../components/skeleton-view-options/skeleton-view-options.component';
import {
  isSkeletonDefinitionInvalid,
  skeletonPairs,
} from '../../components/skeleton-overlay/skeleton-validity';
import { SplitAreaComponent, SplitComponent } from 'angular-split';
import {
  DropdownComponent,
  DropdownContentComponent,
  DropdownTriggerComponent,
  DropdownTriggerDirective,
} from '../../components/dropdown/dropdown.component';
import { RpcService } from '../../rpc.service';
import {
  AuditFocus,
  EpipolarGuides,
  MidlineGuides,
  ShapeGuides,
  ShapePair,
} from '../label-audit/label-audit.model';
import { LabelAuditService } from '../label-audit/label-audit.service';
import { guideColor } from '../label-audit/guide-colors';

@Component({
  selector: 'app-labeler-center-panel',
  imports: [
    DecimalPipe,
    HorizontalScrollDirective,
    BundleAdjustDialogComponent,
    FormsModule,
    ImageLabelWidgetComponent,
    PathDisplayComponent,
    SkeletonViewOptionsComponent,
    LoadCalibrationDialogComponent,
    SplitComponent,
    SplitAreaComponent,
    DropdownComponent,
    DropdownContentComponent,
    DropdownTriggerComponent,
    DropdownTriggerDirective,
  ],
  providers: [LabelerViewOptionsService],
  templateUrl: './labeler-center-panel.component.html',
  styleUrl: './labeler-center-panel.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: {
    '(document:keydown)': 'onDocumentKeydown($event)',
  },
})
export class LabelerCenterPanelComponent implements OnChanges {
  private projectInfoService = inject(ProjectInfoService);
  private toastService = inject(ToastService);
  protected viewOptions = inject(LabelerViewOptionsService);
  protected skeletonInvalid = computed(() => {
    const info = this.projectInfoService.projectContext()?.projectInfo;
    if (!info) return false;
    return isSkeletonDefinitionInvalid(info.skeleton, info.keypoint_names);
  });
  protected noSkeletonPairs = computed(
    () =>
      skeletonPairs(this.projectInfoService.projectContext()?.projectInfo?.skeleton)
        .length === 0,
  );
  protected isMultiview = computed(() => {
    return (
      (this.projectInfoService.projectContext()?.projectInfo?.views?.length ??
        0) > 0
    );
  });

  labelFile = input<MVLabelFile | null>(null);
  frame = input<MVFrame | null>(null);
  hasNextFrame = input(false);
  hasPreviousFrame = input(false);
  numLabeledFramesGetter = input.required<() => number>();
  /** Passed to the image widgets; bumped by the Labeler's Refresh. */
  imageReload = input(0);

  /** The label audit finding being reviewed, or null when not reviewing (#207). */
  auditFocus = input<AuditFocus | null>(null);
  /** Keypoints to draw, or null for all (review shows only the flagged one). */
  visibleKeypoints = input<string[] | null>(null);
  auditOnlyFlagged = input(true);
  auditOnlyFlaggedChange = output<boolean>();
  auditExit = output<void>();

  /** The keypoint can be selected and edited (reviewing with "Only flagged": only the flagged ones). */
  protected isEditableKeypoint(name: string): boolean {
    const only = this.auditFocus() ? this.visibleKeypoints() : null;
    return !only || only.includes(name);
  }

  /** Keypoints drawn in the main image: the review's ones plus the selected one. */
  protected shownKeypoints = computed((): string[] | null => {
    const only = this.visibleKeypoints();
    const selected = this.selectedKeypoint();
    if (!only) return null;
    return selected && !only.includes(selected) ? [...only, selected] : only;
  });

  /**
   * Whether the epipolar lines are on: in the label audit review by its "Lines" switch,
   * while labeling by the "Epipolar lines" switch of the keypoint tools. Needs a calibration.
   */
  protected epipolarOn = computed(
    () =>
      this.hasCameraCalibrationFiles() &&
      (this.auditFocus()
        ? this.viewOptions.showEpipolarLines()
        : this.viewOptions.showEpipolarLinesWhileLabeling()),
  );

  /** Epipolar lines of the selected keypoint's labels in the other views. */
  private allEpipolarGuides = signal<EpipolarGuides | null>(null);
  /** The guides with only the best `maxEpipolarLines` lines (the backend ranks them). */
  protected epipolarGuides = computed((): EpipolarGuides | null => {
    const guides = this.allEpipolarGuides();
    if (!guides || !this.epipolarOn()) return null;
    const max = this.viewOptions.maxEpipolarLines();
    return { ...guides, lines: guides.lines.filter((l) => l.rank < max) };
  });
  /** Other views with a line in this image, for "3 of 5". */
  protected epipolarLineCount = computed(
    () => this.allEpipolarGuides()?.lines.filter((l) => l.segments.length).length ?? 0,
  );
  private epipolarRequest = 0;
  /** The finding (index and frame) whose view and keypoint were last selected for review. */
  private appliedFocus: string | null = null;

  /**
   * How far the selected keypoint's label is from each other view's line, in pixels of
   * the shown view. Follows every edit, so the user sees at once whether the views agree now.
   */
  protected lineDistances = computed(() => {
    this.cacheBuster(); // a keypoint moved
    const guides = this.allEpipolarGuides();
    const focus = this.auditFocus();
    const frame = this.frame();
    const keypoint = this.selectedKeypoint();
    if (!guides || !this.epipolarOn() || !frame || !keypoint) return [];
    const own = this.labelsOf(frame, keypoint).find((l) => l.view === this.selectedView());
    if (!own) return [];
    // Only the views audit has a tolerance in pixels.
    const kind = focus?.finding.kind;
    const tolerance =
      !focus || kind === 'shape' || kind === 'midline' || kind === 'missing'
        ? null
        : focus.finding.tolerancePx;
    return guides.lines
      .filter((l) => l.segments.length)
      .map((l) => {
        const px = distanceToPolylines(own, l.segments);
        return { view: l.fromView, px, ok: tolerance === null ? null : px <= tolerance };
      })
      .filter((d) => Number.isFinite(d.px));
  });

  /**
   * The rigid groups measured with the labels on screen, per finding and frame (`shapeKey`).
   * Kept for the whole review, so going back to a finding shows its last measurement and
   * not the numbers of the check.
   */
  private liveShape = signal<ReadonlyMap<string, ShapeGuides>>(new Map());
  private shapeRequest = 0;
  private shapeTimer: ReturnType<typeof setTimeout> | null = null;
  /** Edit counter value when the reviewed finding opened: no refit before the first edit. */
  private shapeOpened: { key: string; buster: number } | null = null;
  private auditService = inject(LabelAuditService, { optional: true });

  private shapeKey = computed(() => {
    const focus = this.auditFocus();
    const frame = this.frame();
    if (!focus || focus.finding.kind !== 'shape' || !frame) return null;
    return frame.key === focus.finding.frameKey ? `${focus.index}|${frame.key}` : null;
  });
  private currentShape = computed(() => {
    const key = this.shapeKey();
    return key ? (this.liveShape().get(key) ?? null) : null;
  });
  /** True while the distances are being measured again after an edit. */
  protected shapeRefitting = signal(false);

  /** The pairs of the reviewed rigid group: of the last check, or measured again after an edit. */
  protected shapePairs = computed<ShapePair[]>(() => {
    const finding = this.auditFocus()?.finding;
    // Not while another frame than the finding's is on screen: the distances are not its.
    if (!finding || finding.kind !== 'shape' || !this.shapeKey()) return [];
    return this.currentShape()?.groupPairs ?? finding.groupPairs ?? [];
  });

  /**
   * Reviewing a rigid-group finding: a line per measured pair between its two labels in the
   * shown view. The ends follow the labels at once; the color follows the next measurement.
   */
  protected shapeLines = computed(() => {
    this.cacheBuster(); // a keypoint moved
    const frame = this.frame();
    const pairs = this.shapePairs();
    if (!frame || !pairs.length) return [];
    const view = this.selectedView();
    const selected = this.selectedKeypoint();
    const at = (kp: string) => this.labelsOf(frame, kp).find((l) => l.view === view);
    return pairs.flatMap((p) => {
      const a = at(p.a);
      const b = at(p.b);
      if (p.distance === null || !a || !b) return [];
      return [
        {
          name: `${p.a}|${p.b}`,
          x1: a.x,
          y1: a.y,
          x2: b.x,
          y2: b.y,
          off: !p.off ? null : p.distance > p.usual ? ('long' as const) : ('short' as const),
          main: p.a === selected || p.b === selected,
        },
      ];
    });
  });

  /** The measured distances of a keypoint to the rest of its group, and which are off. */
  private pairsOf(keypoint: string) {
    const own = this.shapePairs()
      .filter((p) => (p.a === keypoint || p.b === keypoint) && p.distance !== null)
      .map((p) => ({
        other: p.a === keypoint ? p.b : p.a,
        long: p.distance! > p.usual,
        percent: (100 * p.distance!) / p.usual,
        off: p.off,
      }));
    return { keypoint, measured: own.length, off: own.filter((p) => p.off) };
  }

  /** The selected keypoint's distances; follows every edit. */
  protected shapeReadout = computed(() => {
    const keypoint = this.selectedKeypoint();
    return keypoint && this.shapePairs().length ? this.pairsOf(keypoint) : null;
  });

  /** The flagged keypoint's distances, for the review bar; follows every edit. */
  protected shapeFlagged = computed(() => {
    const finding = this.auditFocus()?.finding;
    return finding?.kind === 'shape' && this.shapePairs().length
      ? this.pairsOf(finding.keypoint)
      : null;
  });

  /** For the review bar: the selected keypoint's distances, else the flagged keypoint's. */
  protected shapeStatus = computed(() => this.shapeReadout() ?? this.shapeFlagged());

  protected readonly shapeColor = (long: boolean) => (long ? '#ef4444' : '#38bdf8');
  protected readonly abs = Math.abs;

  /** The reviewed midline finding measured with the labels on screen, for the finding of `key`. */
  private liveMidline = signal<{ key: string; guides: MidlineGuides | null } | null>(null);
  private midlineRequest = 0;
  private midlineKey = computed(() => {
    const focus = this.auditFocus();
    const frame = this.frame();
    if (!focus || focus.finding.kind !== 'midline' || !frame) return null;
    return frame.key === focus.finding.frameKey ? `${focus.index}|${frame.key}` : null;
  });
  private currentMidline = computed(() => {
    const live = this.liveMidline();
    return live && live.key === this.midlineKey() ? live : null;
  });

  /** The midline and the other views' suggestion in the shown view, in image pixels. */
  protected midlineGuide = computed(() => {
    const guide = this.currentMidline()?.guides?.views.find((v) => v.view === this.selectedView());
    if (!guide) return null;
    return { points: guide.line.map((p) => `${p.x},${p.y}`).join(' '), suggestion: guide.suggestion };
  });

  /**
   * For the review bar: the angle off the midline now (of the last check until the first
   * measurement arrives), and the left-right distance against usual.
   */
  protected midlineStatus = computed(() => {
    const finding = this.auditFocus()?.finding;
    if (!finding || finding.kind !== 'midline' || !this.midlineKey()) return null;
    const live = this.currentMidline();
    const info = live ? (live.guides?.info ?? null) : (finding.midline ?? null);
    const [point, left, right] = finding.group ?? [];
    const yaw = info?.yawDeg ?? null;
    const usual = finding.midline?.widthUsual ?? null;
    const width = info?.width ?? null;
    return {
      point,
      left,
      right,
      yaw,
      towards: yaw === null ? '' : yaw > 0 ? right : left,
      ok: yaw === null ? null : Math.abs(yaw) <= finding.tolerancePx,
      limit: finding.tolerancePx,
      off: info?.off ?? null,
      widthPercent: usual && width !== null ? (100 * width) / usual : null,
    };
  });

  /** Three significant digits: distances are in calibration units of unknown size. */
  protected sig(x: number | null): string {
    return x === null ? '' : Number(x.toPrecision(3)).toString();
  }
  private rpc = inject(RpcService);
  protected readonly guideColor = guideColor;

  protected guideViews = computed(() => {
    const frame = this.frame();
    return frame ? frame.views.map((v) => v.viewName) : [];
  });

  /** Lines in the legend: views with a line, in frame view order. */
  protected guideLegend = computed(() => {
    const guides = this.epipolarGuides();
    if (!guides) return [];
    const views = this.guideViews();
    return guides.lines
      .filter((l) => l.segments.length)
      .map((l) => ({ view: l.fromView, color: guideColor(views.indexOf(l.fromView)) }));
  });

  protected onDocumentKeydown(event: KeyboardEvent) {
    if (!this.auditFocus()) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.target as HTMLElement | null;
    if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
    // Stepping is done with Back / Next / Save and next, so an edit is not left unsaved.
    if (event.key === 'Escape') {
      this.auditExit.emit();
    }
  }

  /** Labels of `keypoint` in every view of the frame (placed ones only). */
  private labelsOf(frame: MVFrame, keypoint: string) {
    return frame.views
      .map((v) => {
        const kp = v.keypoints.find((k) => k.keypointName === keypoint);
        return kp && !lkp(kp).isNaN() ? { view: v.viewName, x: kp.x, y: kp.y } : null;
      })
      .filter((x): x is { view: string; x: number; y: number } => x !== null);
  }

  private async fetchEpipolarGuides(
    frame: MVFrame,
    view: string,
    keypoint: string,
    size: { w: number; h: number } | null,
  ) {
    const request = ++this.epipolarRequest;
    const sessionKey = mvf(frame).autolabelSessionKey;
    const projectKey = this.projectInfoService.projectContext()?.key;
    const all = this.labelsOf(frame, keypoint);
    const labels = all.filter((l) => l.view !== view);
    const own = all.find((l) => l.view === view);
    if (!sessionKey || !projectKey || !labels.length) {
      this.allEpipolarGuides.set(null);
      return;
    }
    try {
      const guides = (await this.rpc.call('getEpipolarGuides', {
        projectKey,
        sessionKey,
        targetView: view,
        labels,
        imageWidth: size?.w ?? null,
        imageHeight: size?.h ?? null,
        targetPoint: own ? { x: own.x, y: own.y } : null,
      })) as EpipolarGuides;
      if (request === this.epipolarRequest) this.allEpipolarGuides.set(guides);
    } catch (e) {
      // Guides are a help, not essential: drop them and log.
      console.error('Epipolar guides failed', e);
      if (request === this.epipolarRequest) this.allEpipolarGuides.set(null);
    }
  }

  primaryZoomableElement =
    viewChild<ImageLabelWidgetComponent>('primaryZoomable');

  saved = output<{
    labelFile: MVLabelFile;
    frame: MVFrame;
    shouldAdvanceFrame: boolean;
    deletion?: boolean;
  }>();
  /** True while a save is being written, false when it finishes or fails. */
  savingChange = output<boolean>();
  nextFrame = output<void>();
  previousFrame = output<void>();
  closeFrame = output<void>();

  constructor() {
    // Reviewing a finding: show its view with its keypoint selected, once its frame is open.
    effect(() => {
      const focus = this.auditFocus();
      const frame = this.frame();
      if (!focus) this.appliedFocus = null;
      if (!focus || !frame || frame.key !== focus.finding.frameKey) return;
      // Once per finding: a save replaces the frame object and must not undo a view the user chose.
      const key = focus.index + '|' + frame.key;
      if (key === this.appliedFocus) return;
      this.appliedFocus = key;
      untracked(() => {
        this._selectedView.set(focus.finding.view);
        this.selectedKeypoint.set(focus.finding.keypoint);
      });
    });

    // Reviewing with "Only flagged": the selection stays on the flagged keypoint (or its
    // swap partner), whatever else would move it: placing a label, changing view, a new frame.
    effect(() => {
      const only = this.visibleKeypoints();
      const focus = this.auditFocus();
      const frame = this.frame();
      const selected = this.selectedKeypoint();
      if (!only || !focus || !frame || frame.key !== focus.finding.frameKey) return;
      if (selected && only.includes(selected)) return;
      untracked(() => this.selectedKeypoint.set(focus.finding.keypoint));
    });

    // The group's distances follow the edits: measured again after every edit, with the
    // unsaved labels.
    effect(() => {
      const buster = this.cacheBuster(); // a keypoint moved
      const key = this.shapeKey();
      const frame = this.frame();
      const focus = this.auditFocus();
      untracked(() => {
        if (this.shapeTimer) clearTimeout(this.shapeTimer);
        const service = this.auditService;
        // The review stopped: its measurements go (finding numbers start again at 0).
        if (!focus && this.liveShape().size) this.liveShape.set(new Map());
        if (!key || !frame || !focus || !service) {
          this.shapeOpened = null;
          this.shapeRefitting.set(false);
          return;
        }
        if (this.shapeOpened?.key !== key) {
          this.shapeOpened = { key, buster };
          this.shapeRefitting.set(false);
          return;
        }
        if (this.shapeOpened.buster === buster) return;
        const request = ++this.shapeRequest;
        this.shapeRefitting.set(true);
        this.shapeTimer = setTimeout(async () => {
          try {
            const guides = await service.shapeGuides(frame, focus.finding.keypoint);
            // By key, so a late answer never lands on another finding.
            if (guides) this.liveShape.update((all) => new Map(all).set(key, guides));
          } catch (e) {
            // The distances of the last check stay: a help, not essential.
            console.error('Shape refit failed', e);
          } finally {
            if (request === this.shapeRequest) this.shapeRefitting.set(false);
          }
        }, 0); // at once, as the epipolar guides; newer edits supersede older requests
      });
    });

    // The midline follows the edits: measured when the finding opens and after every edit,
    // with the unsaved labels.
    effect(() => {
      this.cacheBuster(); // a keypoint moved
      const key = this.midlineKey();
      const frame = this.frame();
      const focus = this.auditFocus();
      untracked(() => {
        const request = ++this.midlineRequest;
        const service = this.auditService;
        const sessionKey = frame ? mvf(frame).autolabelSessionKey : null;
        if (!key || !frame || !focus || !service || !sessionKey) return;
        service
          .midlineGuides(frame, sessionKey, focus.finding.group ?? [])
          .then((guides) => {
            if (request === this.midlineRequest) this.liveMidline.set({ key, guides });
          })
          .catch((e) => {
            // The numbers of the last check stay: a help, not essential.
            console.error('Midline guides failed', e);
          });
      });
    });

    // Epipolar guides follow the selected keypoint, view and its labels.
    effect(() => {
      this.cacheBuster(); // a keypoint moved
      const frame = this.frame();
      const view = this.selectedView();
      const keypoint = this.selectedKeypoint();
      const show = this.epipolarOn();
      const size = this.primaryZoomableElement()?.imageSize() ?? null;
      untracked(() => {
        if (!show || !frame || !keypoint) {
          this.epipolarRequest++;
          this.allEpipolarGuides.set(null);
          return;
        }
        void this.fetchEpipolarGuides(frame, view, keypoint, size);
      });
    });

    // The controls are disabled without context, so stop playback and show
    // the current frame (with keypoints) rather than leave it running.
    effect(() => {
      if (!this.hasTemporalContext()) {
        untracked(() => this.viewOptions.resetTemporalContext());
      }
    });

    effect(() => {
      this.selectedView();

      // Reset scale when view changes.
      // Use untracked() to avoid creating a dependency on primaryZoomableElement
      untracked(() => {
        if (this.primaryZoomableElement()?.zoomableContent()) {
          this.primaryZoomableElement()!
            .zoomableContent()!
            .fitContentToViewport();
        }
      });
    });
  }
  ngOnChanges(changes: SimpleChanges) {
    if (changes['frame']) {
      this.abortController.abort();
      this.abortController = new AbortController();
      this.calibrationStatus.set('none');
      this.checkIfHasCameraCalibrationFiles();
    }

    if (changes['frame']) {
      // A new frame opens on itself, with keypoints shown.
      this.viewOptions.resetTemporalContext();

      // Reset scale.
      if (this.primaryZoomableElement()?.zoomableContent()) {
        this.primaryZoomableElement()!
          .zoomableContent()!
          .fitContentToViewport();
      }
    }

    // A frame reached by Back opens on its last view. Unlabeled frames, and
    // a frame reached by Next or Save and next, start from the first view.
    if (changes['frame'] && this.frame()) {
      const openOn = this.openFrameOn;
      this.openFrameOn = null;
      if (openOn === 'last') {
        this._selectedView.set(this.frame()!.views.at(-1)?.viewName ?? null);
        this.selectKeypointForView();
      } else if (openOn === 'first' || mvf(this.frame()!).isFromUnlabeledSet) {
        this._selectedView.set(null);
        this.selectKeypointForView();
      }
    }
  }

  /**
   * Which view the next frame opens on: 'first' after Next or Save and next
   * leaves the last view, 'last' after Back leaves the first view.
   */
  private openFrameOn: 'first' | 'last' | null = null;

  /**
   * Selects the keypoint for a newly shown view: the same keypoint when
   * "Keep selected keypoint" is on and the view has it, otherwise the first.
   */
  private selectKeypointForView() {
    const keypoints = this.selectedFrameView()?.keypoints ?? [];
    const current = this.selectedKeypoint();
    if (
      this.viewOptions.keepSelectedKeypoint() &&
      keypoints.some((kp) => kp.keypointName === current)
    ) {
      return;
    }
    this.selectedKeypoint.set(keypoints[0]?.keypointName ?? null);
  }

  protected sessionKey = computed(() => {
    const frame = this.frame();
    if (!frame) return null;
    return mvf(frame).autolabelSessionKey;
  });

  private checkIfHasCameraCalibrationFiles() {
    const abortSignal = this.abortController.signal;
    const sessionKey = this.sessionKey();
    if (!sessionKey) return;
    this.cameraCalibrationService.getCalibrationStatus(sessionKey).then((status) => {
      if (abortSignal.aborted) return;
      this.calibrationStatus.set(status);
    });
  }

  // selectedView is always nonnull. Defaults to first view in frame.
  // Underlying selection state can be null if user hasn't selected any view explicitly yet.
  protected _selectedView = signal<string | null>(null);
  protected selectedView = computed((): string => {
    if (this._selectedView()) {
      return this._selectedView()!;
    } else {
      return this.frame()?.views[0]?.viewName ?? 'unknown';
    }
  });

  protected selectedFrameView = computed(() =>
    this.getFrameView(this.selectedView()),
  );
  protected contextSlotLabels = CONTEXT_SLOT_LABELS;
  /** Why temporal context can't be shown for the selected image, or null when it can. */
  protected temporalContextUnavailableReason = computed((): string | null => {
    const frameView = this.selectedFrameView();
    if (!frameView) return null;
    // LP builds no context when the file name has no frame number.
    if (contextImgPaths(frameView.imgPath) === null) {
      return 'The image file name has no frame number.';
    }
    const frameMessage = this.primaryZoomableElement()?.currentFrameMessage();
    if (frameMessage) return frameMessage;
    if (this.primaryZoomableElement()?.contextFramesMissing()) {
      return 'Context frames were not found.';
    }
    return null;
  });
  /** "Showing: …" text, naming the frame actually shown when a slot shows a substitute. */
  protected contextShowingText = computed(() => {
    const slot = this.viewOptions.temporalContextIndex();
    const views = this.primaryZoomableElement()?.contextSlotViews();
    return contextShowingText(slot, views?.[slot]);
  });
  protected hasTemporalContext = computed(
    () => !!this.selectedFrameView() && !this.temporalContextUnavailableReason(),
  );
  protected getFrameView(view: string): FrameView | null {
    const frame = this.frame();
    if (!frame) {
      return null;
    }
    return frame.views.find((x) => x.viewName === view) ?? null;
  }
  protected selectedKeypoint = signal<string | null>(null);

  handleViewClickFromFilmstrip(viewName: string) {
    this._selectedView.set(viewName);
    this.selectKeypointForView();
  }

  // Set to a random value if you need to bust cache of computed signals.
  protected cacheBuster = signal(0);
  protected loadCalibrationOpen = signal(false);
  /** an overload of `handleKeypointUpdated` that uses viewName: string
   * instead of a reference to frameview. Used for autolabeling */
  private handleKeypointUpdated2(
    keypointName: string,
    viewName: string,
    position: Point,
  ) {
    const frameView = this.getFrameView(viewName)!;
    this.handleKeypointUpdated(keypointName, position, frameView);
  }
  handleKeypointUpdated(
    keypointName: string,
    position: Point,
    frameView: FrameView,
  ) {
    const keypoint = frameView.keypoints.find(
      (kp) => kp.keypointName === keypointName,
    )!;
    const newKp = {
      x: position.x,
      y: position.y,
      keypointName: keypointName,
    };
    // Update domain model.
    frameView.keypoints = frameView.keypoints.map((item) =>
      item === keypoint ? { ...keypoint, ...newKp } : item,
    );

    // With "Keep selected keypoint" on, the placed keypoint stays selected.
    if (lkp(keypoint).isNaN() && !this.viewOptions.keepSelectedKeypoint()) {
      this.selectNextUnlabeledKeypoint(keypointName);
    }
    this.cacheBuster.set(Math.random());
  }

  /**
   * Selects the next keypoint that does not have a label in the currently selected frame view.
   */
  private selectNextUnlabeledKeypoint(originKp: string | null) {
    const frameView = this.selectedFrameView();

    if (frameView) {
      const startIdx =
        originKp === null
          ? 0
          : frameView.keypoints.findIndex((kp) => kp.keypointName === originKp);
      for (let i = startIdx + 1; i < frameView.keypoints.length; i++) {
        const kp = frameView.keypoints[i];
        if (lkp(kp).isNaN()) {
          this.selectedKeypoint.set(kp.keypointName);
          return;
        }
      }
    }
  }

  protected handleKeypointClearClick(kp: LKeypoint) {
    if (this.selectedFrameView()) {
      this.handleKeypointUpdated(
        kp.keypointName,
        { x: NaN, y: NaN },
        this.selectedFrameView()!,
      );
    }
  }

  // export to template
  protected readonly mvf = mvf;

  private sessionService = inject(SessionService);
  private cameraCalibrationService = inject(CameraCalibrationService);
  private abortController = new AbortController();
  protected isSaving = signal(false);

  private setSaving(saving: boolean) {
    this.isSaving.set(saving);
    this.savingChange.emit(saving);
  }
  private isMVAutoLabeling = signal(false);
  protected calibrationStatus = signal<'session' | 'project' | 'none'>('none');
  protected hasCameraCalibrationFiles = computed(
    () => this.calibrationStatus() !== 'none',
  );
  protected disableInteractions = computed(() => {
    this.cacheBuster();

    return !this.frame() || this.isSaving() || this.isMVAutoLabeling();
  });

  protected isSaveDisabled = computed(() => {
    this.cacheBuster();
    return (
      !this.frame() ||
      this.disableInteractions() ||
      !mvf(this.frame()!).hasChanges
    );
  });

  protected hasNextView = computed(() => !!this.nextFrameView());

  private nextFrameView = computed(() => this.frameViewAtOffset(1));
  private previousFrameView = computed(() => this.frameViewAtOffset(-1));

  private frameViewAtOffset(offset: number): FrameView | null {
    const frame = this.frame();
    const frameView = this.selectedFrameView();
    if (!frame || !frameView) return null;
    // Reviewing the label audit: Back / Next / Save and next go from one likely mistake
    // to the next, not through the views of the frame (the filmstrip still switches view).
    if (this.auditFocus()) return null;
    return frame.views[frame.views.indexOf(frameView) + offset] ?? null;
  }

  /** Next is active when there is another view of this frame or another frame. */
  protected isNextDisabled = computed(
    () =>
      (!this.hasNextView() && !this.hasNextFrame()) ||
      this.disableInteractions(),
  );

  /** Back is active when there is an earlier view of this frame or an earlier frame. */
  protected isBackDisabled = computed(
    () =>
      (!this.previousFrameView() && !this.hasPreviousFrame()) ||
      this.disableInteractions(),
  );

  /** Clear view is active when the selected view has a placed keypoint. */
  protected canClearView = computed(() => {
    this.cacheBuster();
    const frameView = this.selectedFrameView();
    return (
      !!frameView &&
      !this.disableInteractions() &&
      fv(frameView).hasPlacedKeypoints
    );
  });

  protected isSaveAndNextDisabled = computed(() => {
    return (
      this.isSaveDisabled() || (!this.hasNextView() && !this.hasNextFrame())
    );
  });

  protected canRevert = computed(() => {
    this.cacheBuster();
    const frame = this.frame();
    if (!frame || this.disableInteractions()) return false;
    return mvf(frame).hasRevertibleEdits;
  });

  // Frame buttons have a tooltip only while inactive, saying why.
  private busyReason = computed(() => {
    if (this.isSaving()) return 'Saving…';
    if (this.isMVAutoLabeling()) return 'Labeling the other views…';
    return '';
  });

  protected saveDisabledReason = computed(() => {
    if (!this.isSaveDisabled()) return '';
    return this.busyReason() || 'No changes to save.';
  });

  protected saveAndNextDisabledReason = computed(() => {
    if (!this.isSaveAndNextDisabled()) return '';
    return (
      this.saveDisabledReason() ||
      (this.auditFocus() ? 'This is the last flagged label.' : 'Already on the last view of the last frame.')
    );
  });

  protected backDisabledReason = computed(() => {
    if (!this.isBackDisabled()) return '';
    return (
      this.busyReason() ||
      (this.auditFocus() ? 'This is the first flagged label.' : 'Already on the first view of the first frame.')
    );
  });

  protected nextDisabledReason = computed(() => {
    if (!this.isNextDisabled()) return '';
    return (
      this.busyReason() ||
      (this.auditFocus() ? 'This is the last flagged label.' : 'Already on the last view of the last frame.')
    );
  });

  protected revertDisabledReason = computed(() => {
    if (this.canRevert()) return '';
    return this.busyReason() || 'No changes to revert.';
  });

  protected removeFrameTooltip(): string {
    const frame = this.frame();
    if (frame && mvf(frame).isFromUnlabeledSet) {
      return 'Remove frame from labeling queue';
    }
    return 'Remove frame from label file';
  }

  /** Shows the next view of this frame, or the next frame from the last view. Does not save. */
  protected handleNext() {
    if (this.isNextDisabled()) return;
    const nextView = this.nextFrameView();
    if (nextView) {
      this._selectedView.set(nextView.viewName);
      this.selectKeypointForView();
      return;
    }
    this.openFrameOn = 'first';
    this.nextFrame.emit();
  }

  /** Shows the previous view of this frame, or the last view of the previous frame from the first view. Does not save. */
  protected handleBack() {
    if (this.isBackDisabled()) return;
    const previousView = this.previousFrameView();
    if (previousView) {
      this._selectedView.set(previousView.viewName);
      this.selectKeypointForView();
      return;
    }
    this.openFrameOn = 'last';
    this.previousFrame.emit();
  }

  /** Unplaces every keypoint in the selected view. Not saved until Save; Revert undoes it. */
  protected handleClearView() {
    const frameView = this.selectedFrameView();
    if (!frameView || !this.canClearView()) return;
    fv(frameView).clearKeypoints();
    this.selectKeypointForView();
    this.cacheBuster.set(Math.random());
  }

  protected handleRevert() {
    const frame = this.frame();
    if (!frame || this.disableInteractions() || !mvf(frame).hasRevertibleEdits) {
      return;
    }
    mvf(frame).revertEdits();
    this.cacheBuster.set(Math.random());
  }

  protected onCalibrationLoadDone(saved: boolean) {
    this.loadCalibrationOpen.set(false);
    if (saved) this.checkIfHasCameraCalibrationFiles();
  }

  newLabelFile = output();

  protected deleteConfirmationDialog = viewChild<ElementRef<HTMLDialogElement>>(
    'deleteConfirmationDialog',
  );
  protected deletionShouldContinue = new Subject<boolean>();

  protected async handleSaveClick(
    labelFile: MVLabelFile,
    frame: MVFrame,
    shouldContinue: boolean,
    deletion?: boolean,
  ) {
    if (deletion) {
      this.deleteConfirmationDialog()!.nativeElement.showModal();
      if (!(await firstValueFrom(this.deletionShouldContinue.asObservable())))
        return;
    }

    this.setSaving(true);
    const nextFrameView = this.nextFrameView();

    this.sessionService
      .saveMVFrame(labelFile, frame, deletion)
      .then(() => {
        this.setSaving(false);
        if (shouldContinue && !nextFrameView) {
          this.openFrameOn = 'first';
        }
        this.saved.emit({
          labelFile,
          frame,
          // If save and next button clicked, and there is no next view, go to the next frame.
          shouldAdvanceFrame: shouldContinue && !nextFrameView,
          deletion,
        });
        this.toastService.showToast({ content: 'Saved successfully' });
        // If save and next button clicked and there is a next view, select it.
        if (shouldContinue && nextFrameView) {
          this._selectedView.set(nextFrameView.viewName);
        }
        this.selectKeypointForView();
      })
      .catch((error) => {
        this.setSaving(false);
        throw error;
      });
  }

  protected handleMVAutoLabelClick(labelFile: MVLabelFile, frame: MVFrame) {
    this.isMVAutoLabeling.set(true);
    const abortSignal = this.abortController.signal;
    const sessionKey = mvf(frame).autolabelSessionKey;
    if (!sessionKey) {
      return;
    }
    this.sessionService
      .mvAutoLabel(frame, sessionKey)
      .then((response: GetMVAutoLabelsResponse) => {
        this.isMVAutoLabeling.set(false);
        if (abortSignal.aborted) return;
        // Patch keypoints with projections from response.
        response.keypoints.forEach((kp) => {
          if (!kp.projections) return;
          kp.projections.forEach((proj) => {
            const projectedPoint = proj.projectedPoint;
            // Skip because there were not enough labeled views to autolabel.
            if (!projectedPoint) return;
            // If this is present, the original point was already labeled.
            if (proj.originalPoint) return;
            this.handleKeypointUpdated2(kp.keypointName, proj.view, {
              x: projectedPoint.x,
              y: projectedPoint.y,
            });
          });
        });
        this.toastService.showToast({
          content: 'Success: keypoints have been labeled wherever possible.',
        });
      })
      .catch((error) => {
        this.isMVAutoLabeling.set(false);
        throw error;
      });
  }

  protected handleDeleteConfirmationDialogContinuation(
    shouldContinue: boolean,
  ) {
    this.deleteConfirmationDialog()!.nativeElement.close();
    this.deletionShouldContinue.next(shouldContinue);
  }
}

/** Smallest distance from a point to a set of polylines (Infinity when there is none). */
export function distanceToPolylines(
  p: { x: number; y: number },
  polylines: { x: number; y: number }[][],
): number {
  let best = Infinity;
  for (const line of polylines) {
    for (let i = 0; i + 1 < line.length; i++) {
      const a = line[i];
      const b = line[i + 1];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const len2 = dx * dx + dy * dy;
      const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2));
      best = Math.min(best, Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy)));
    }
  }
  return best;
}
