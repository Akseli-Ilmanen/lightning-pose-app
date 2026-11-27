import { computed, inject, Injectable, signal } from '@angular/core';
import { RpcService } from '../../rpc.service';
import { ProjectInfoService } from '../../project-info.service';
import { MVLabelFile } from '../../label-file.model';
import { MVFrame } from '../frame.model';
import {
  AuditFinding,
  AuditFocus,
  AuditMode,
  auditModeOf,
  AuditScope,
  GetLabelAuditResponse,
  MidlineChoice,
  MidlineGuides,
  MissingWhere,
  ShapeGuides,
} from './label-audit.model';

export const DEFAULT_AUDIT_TOLERANCE_PX = 5;
const CHOICES_KEY = 'label-audit.keypoints';
export const MAX_RIGID_GROUPS = 4;
export const DEFAULT_SHAPE_STRICTNESS = 3;
export const DEFAULT_MIDLINE_TOLERANCE_DEG = 5;

/**
 * The midline choice with its empty fields guessed from the keypoint names: a left and a
 * right shoulder, and a keypoint called "back".
 */
export function midlineChoiceFor(choice: MidlineChoice, names: string[]): MidlineChoice {
  const left = names.find((n) => /left.*shoulder|shoulder.*left/i.test(n)) ?? '';
  const right = names.find((n) => /right.*shoulder|shoulder.*right/i.test(n)) ?? '';
  const point = names.find((n) => /^back$/i.test(n)) ?? '';
  return {
    ...choice,
    left: choice.left || (left && right ? left : ''),
    right: choice.right || (left && right ? right : ''),
    point: choice.point || point,
  };
}

/** True when the three keypoints are chosen, differ and are in the label file. */
export function midlineIsSet(choice: MidlineChoice, names: string[]): boolean {
  const kps = [choice.left, choice.right, choice.point];
  return new Set(kps).size === 3 && kps.every((k) => names.includes(k));
}

/**
 * One finding per frame where `keypoint` has no label: in at least one view ("any"), in
 * every view ("all"), or in some views but not all ("some"). Frame order. Views in
 * `skipViews` are not looked at: "every view" then means every view that is looked at.
 */
export function missingFindings(
  frames: MVFrame[],
  keypoint: string,
  where: MissingWhere,
  skipViews: string[] = [],
): AuditFinding[] {
  if (!keypoint) return [];
  const out: AuditFinding[] = [];
  for (const frame of frames) {
    const looked = frame.views.filter((v) => !skipViews.includes(v.viewName));
    const missing = looked
      .filter((v) => {
        const k = v.keypoints.find((kp) => kp.keypointName === keypoint);
        return !k || !Number.isFinite(k.x) || !Number.isFinite(k.y);
      })
      .map((v) => v.viewName);
    const all = missing.length === looked.length;
    if (!missing.length || (where === 'all' && !all) || (where === 'some' && all)) continue;
    const everyView = missing.length === frame.views.length && frame.views.length > 1;
    out.push({
      kind: 'missing',
      view: missing[0],
      keypoint,
      partner: null,
      imgPaths: Object.fromEntries(frame.views.map((v) => [v.viewName, v.imgPath])),
      errorPx: missing.length,
      tolerancePx: looked.length,
      score: missing.length / looked.length,
      reason: everyView
        ? `${keypoint} is missing in every view`
        : `${keypoint} is missing in ${missing.join(', ')}`,
      missingIn: everyView ? 'every view' : missing.join(', '),
      frameKey: frame.key,
    });
  }
  return out;
}

/** Which keypoints the audit reports, and which rigid group (1..) each belongs to. */
export interface AuditKeypointChoices {
  /** Keypoints left out of the findings; every other keypoint is checked. */
  unchecked: string[];
  /** Rigid group number per keypoint; absent or 0 = none. */
  groups: Record<string, number>;
}

/** Keypoints of each rigid group, groups in number order, names in `names` order. */
export function rigidGroupsOf(groups: Record<string, number>, names: string[]): string[][] {
  const byGroup = new Map<number, string[]>();
  for (const name of names) {
    const g = groups[name] ?? 0;
    if (g > 0) byGroup.set(g, [...(byGroup.get(g) ?? []), name]);
  }
  return [...byGroup.entries()].sort((a, b) => a[0] - b[0]).map(([, kps]) => kps);
}

/** Keypoint names of the label file, from its first frame. */
export function keypointNamesOf(frames: MVFrame[]): string[] {
  return frames[0]?.views[0]?.keypoints.map((k) => k.keypointName) ?? [];
}

/**
 * Label audit state for one Labeler page (#207): the findings of the last run
 * and where the review is. Provided by LabelerPageComponent.
 */
@Injectable()
export class LabelAuditService {
  private rpc = inject(RpcService);
  private projectInfoService = inject(ProjectInfoService);

  readonly tolerancePx = signal(DEFAULT_AUDIT_TOLERANCE_PX);
  /** Keypoints to report and their rigid groups. Remembered per project. */
  readonly choices = signal<AuditKeypointChoices>(this.storedChoices());
  /** What the next check runs on: the saved labels, or the frames still to label. */
  readonly scope = signal<AuditScope>('labeled');
  /** What the findings shown were checked on. */
  readonly auditedScope = signal<AuditScope>('labeled');
  /** Queue check: keys of the queue frames without any finding (they can be approved as they are). */
  readonly cleanFrameKeys = signal<string[]>([]);
  readonly isRunning = signal(false);
  readonly error = signal<string | null>(null);
  /** Label file key the findings belong to; null before the first run. */
  readonly auditedLabelFileKey = signal<string | null>(null);
  /** Findings of both audits, as the last check returned them. */
  readonly allFindings = signal<AuditFinding[]>([]);
  /** The audit shown: its settings, findings and review. */
  readonly mode = signal<AuditMode>('missing');
  /** Shape check: a distance this many spreads from its usual value counts as off. */
  readonly shapeStrictness = signal(DEFAULT_SHAPE_STRICTNESS);
  /**
   * Distances check: names of the animals as they occur in the session names, comma
   * separated. Each gets its own distances. Remembered per project.
   */
  readonly animals = signal(this.storedAnimals());
  readonly animalNames = computed(() =>
    this.animals()
      .split(',')
      .map((a) => a.trim())
      .filter((a) => a.length > 0),
  );
  /** Midline check: the three keypoints and the allowed angle. Remembered per project. */
  readonly midline = signal<MidlineChoice>(this.storedMidline());
  private countOf(mode: AuditMode): number {
    return this.allFindings().filter((f) => auditModeOf(f.kind) === mode).length;
  }
  /** Missing: the keypoint whose missing labels are listed, and in which views. Remembered per project. */
  readonly missingKeypoint = signal(this.stored('missing-keypoint'));
  readonly missingWhere = signal<MissingWhere>(
    (['any', 'all', 'some'] as const).find((w) => w === this.stored('missing-where')) ?? 'any',
  );
  /** Missing: cameras that are not looked at. Remembered per project. */
  readonly missingSkipViews = signal<string[]>(
    this.stored('missing-skip-views')
      .split('\n')
      .filter((v) => v.length > 0),
  );
  readonly missingCount = computed(() => this.countOf('missing'));
  /** The findings of the server and the frames of the last check, to list another keypoint's gaps. */
  private serverFindings: AuditFinding[] = [];
  private checkedFrames: MVFrame[] = [];
  readonly viewsCount = computed(() => this.countOf('views'));
  readonly shapeCount = computed(() => this.countOf('shape'));
  readonly midlineCount = computed(() => this.countOf('midline'));
  /**
   * Findings of the shown audit; the review steps through these. Midline findings are one
   * per frame, so they come worst first; the others stay in frame order.
   */
  readonly findings = computed(() => {
    const mode = this.mode();
    const own = this.allFindings().filter((f) => auditModeOf(f.kind) === mode);
    return mode === 'midline' ? own.sort((a, b) => b.score - a.score) : own;
  });
  readonly summary = signal<Omit<GetLabelAuditResponse, 'items'> | null>(null);
  /** Findings whose frame is not in the loaded label file (should not happen). */
  readonly unmatched = signal(0);

  /** Index of the finding being reviewed; null when not reviewing. */
  readonly reviewIndex = signal<number | null>(null);
  /** While reviewing, only the flagged keypoint (and its swap partner) is drawn. */
  readonly onlyFlaggedKeypoints = signal(true);
  /** Findings the user has looked at in this review. */
  readonly visited = signal<ReadonlySet<number>>(new Set());

  readonly isReviewing = computed(() => this.reviewIndex() !== null);

  readonly focus = computed((): AuditFocus | null => {
    const i = this.reviewIndex();
    const all = this.findings();
    if (i === null || !all[i]) return null;
    return { finding: all[i], index: i, total: all.length };
  });

  /** Keypoints to draw while reviewing, or null for all. */
  readonly visibleKeypoints = computed((): string[] | null => {
    const f = this.focus()?.finding;
    if (!f || !this.onlyFlaggedKeypoints()) return null;
    // A rigid group is judged as a whole: any of its labels may be the wrong one.
    // The midline keypoint is judged against the left and right keypoints, which may be the wrong ones.
    if ((f.kind === 'shape' || f.kind === 'midline') && f.group?.length) return f.group;
    // A missing keypoint is placed by looking at the others.
    if (f.kind === 'missing') return null;
    return f.partner ? [f.keypoint, f.partner] : [f.keypoint];
  });

  async run(labelFile: MVLabelFile, frames: MVFrame[]): Promise<void> {
    const projectKey = this.projectInfoService.projectContext()?.key;
    if (!projectKey) throw new Error('Project key missing from project context');
    this.isRunning.set(true);
    this.error.set(null);
    this.stopReview();
    this.lastRun = { labelFile, names: keypointNamesOf(frames) };
    const scope = this.scope();
    const names = keypointNamesOf(frames);
    const { unchecked, groups } = this.choices();
    const checked = names.filter((n) => !unchecked.includes(n));
    const midline = midlineChoiceFor(this.midline(), names);
    try {
      const response = (await this.rpc.call('getLabelAudit', {
        projectKey,
        views: labelFile.views.map((v) => ({
          viewName: v.viewName,
          csvPath: v.csvPath,
        })),
        defaultTolerancePx: this.tolerancePx(),
        rigidGroups: rigidGroupsOf(groups, names),
        keypoints: checked.length === names.length ? null : checked,
        shapeStrictness: this.shapeStrictness(),
        animals: this.animalNames(),
        midline: midlineIsSet(midline, names) ? midline : null,
        scope: scope === 'queue' ? 'unlabeled' : 'labeled',
      })) as GetLabelAuditResponse;
      const { items, ...summary } = response;
      const findings = matchFindingsToFrames(items, frames);
      this.serverFindings = findings;
      this.checkedFrames = frames;
      this.allFindings.set([
        ...findings,
        ...missingFindings(
          frames,
          this.missingKeypoint(),
          this.missingWhere(),
          this.missingSkipViews(),
        ),
      ]);
      this.unmatched.set(items.length - findings.length);
      this.summary.set(summary);
      this.auditedScope.set(scope);
      const frameOfImg = new Map<string, string>();
      for (const frame of frames) {
        for (const view of frame.views) frameOfImg.set(view.imgPath, frame.key);
      }
      this.cleanFrameKeys.set(
        (response.cleanFrames ?? [])
          .map((paths) => frameOfImg.get(Object.values(paths)[0]))
          .filter((key): key is string => !!key),
      );
      this.auditedLabelFileKey.set(labelFile.key);
    } catch (e) {
      this.serverFindings = [];
      this.checkedFrames = [];
      this.allFindings.set([]);
      this.cleanFrameKeys.set([]);
      this.summary.set(null);
      this.error.set(errorText(e));
    } finally {
      this.isRunning.set(false);
    }
  }

  /** The label file and keypoints of the last check, for refitting a frame during the review. */
  private lastRun: { labelFile: MVLabelFile; names: string[] } | null = null;

  /**
   * The rigid group of `keypoint` refitted to the frame's labels as they are on screen,
   * unsaved edits included. Null when there is nothing to fit.
   */
  async shapeGuides(frame: MVFrame, keypoint: string): Promise<ShapeGuides | null> {
    const projectKey = this.projectInfoService.projectContext()?.key;
    const last = this.lastRun;
    if (!projectKey || !last) return null;
    const labels: Record<string, Record<string, [number, number] | null>> = {};
    const imgPaths: Record<string, string> = {};
    for (const view of frame.views) {
      imgPaths[view.viewName] = view.imgPath;
      labels[view.viewName] = {};
      for (const k of view.keypoints) {
        labels[view.viewName][k.keypointName] =
          Number.isFinite(k.x) && Number.isFinite(k.y) ? [k.x, k.y] : null;
      }
    }
    const guides = (await this.rpc.call('getShapeGuides', {
      projectKey,
      views: last.labelFile.views.map((v) => ({ viewName: v.viewName, csvPath: v.csvPath })),
      defaultTolerancePx: this.tolerancePx(),
      rigidGroups: rigidGroupsOf(this.choices().groups, last.names),
      shapeStrictness: this.shapeStrictness(),
      animals: this.animalNames(),
      scope: this.auditedScope() === 'queue' ? 'unlabeled' : 'labeled',
      imgPaths,
      labels,
      keypoint,
    })) as ShapeGuides;
    return guides.group.length ? guides : null;
  }

  /**
   * The midline of the frame with its labels as they are on screen, unsaved edits included.
   * `group` is a midline finding's [point, left, right]. Null when the left and right
   * keypoints cannot be triangulated.
   */
  async midlineGuides(
    frame: MVFrame,
    sessionKey: string,
    group: string[],
  ): Promise<MidlineGuides | null> {
    const projectKey = this.projectInfoService.projectContext()?.key;
    const [point, left, right] = group;
    if (!projectKey || !point || !left || !right) return null;
    const labels: Record<string, Record<string, [number, number] | null>> = {};
    for (const view of frame.views) {
      labels[view.viewName] = {};
      for (const k of view.keypoints) {
        if (!group.includes(k.keypointName)) continue;
        labels[view.viewName][k.keypointName] =
          Number.isFinite(k.x) && Number.isFinite(k.y) ? [k.x, k.y] : null;
      }
    }
    const guides = (await this.rpc.call('getMidlineGuides', {
      projectKey,
      sessionKey,
      left,
      right,
      point,
      labels,
    })) as MidlineGuides;
    return guides.views.length ? guides : null;
  }

  setMidline(patch: Partial<MidlineChoice>) {
    const choice = { ...this.midline(), ...patch };
    this.midline.set(choice);
    try {
      localStorage.setItem(this.midlineStorageKey(), JSON.stringify(choice));
    } catch {
      // Storage is a convenience only.
    }
  }

  private midlineStorageKey(): string {
    return `label-audit.midline.${this.projectInfoService.projectContext()?.key ?? ''}`;
  }

  private storedMidline(): MidlineChoice {
    try {
      const raw = JSON.parse(localStorage.getItem(this.midlineStorageKey()) ?? 'null');
      if (raw && typeof raw === 'object') {
        const tolerance = Number(raw.toleranceDeg);
        return {
          left: String(raw.left ?? ''),
          right: String(raw.right ?? ''),
          point: String(raw.point ?? ''),
          toleranceDeg: tolerance > 0 ? tolerance : DEFAULT_MIDLINE_TOLERANCE_DEG,
        };
      }
    } catch {
      // fall through to the default
    }
    return { left: '', right: '', point: '', toleranceDeg: DEFAULT_MIDLINE_TOLERANCE_DEG };
  }

  /**
   * Missing: another keypoint, or another rule for the views. The list follows at once, from
   * the frames as they were at the last check.
   */
  setMissing(patch: { keypoint?: string; where?: MissingWhere; skipViews?: string[] }) {
    if (patch.keypoint !== undefined) this.missingKeypoint.set(patch.keypoint);
    if (patch.where !== undefined) this.missingWhere.set(patch.where);
    if (patch.skipViews !== undefined) this.missingSkipViews.set(patch.skipViews);
    this.store('missing-skip-views', this.missingSkipViews().join('\n'));
    this.store('missing-keypoint', this.missingKeypoint());
    this.store('missing-where', this.missingWhere());
    if (this.auditedLabelFileKey() === null) return;
    this.stopReview();
    this.allFindings.set([
      ...this.serverFindings,
      ...missingFindings(
        this.checkedFrames,
        this.missingKeypoint(),
        this.missingWhere(),
        this.missingSkipViews(),
      ),
    ]);
  }

  private stored(name: string): string {
    try {
      return (
        localStorage.getItem(
          `label-audit.${name}.${this.projectInfoService.projectContext()?.key ?? ''}`,
        ) ?? ''
      );
    } catch {
      return '';
    }
  }

  private store(name: string, value: string) {
    try {
      localStorage.setItem(
        `label-audit.${name}.${this.projectInfoService.projectContext()?.key ?? ''}`,
        value,
      );
    } catch {
      // Storage is a convenience only.
    }
  }

  /** Another scope: the findings of the last check no longer apply. */
  setScope(scope: AuditScope) {
    if (scope === this.scope()) return;
    this.clear();
    this.scope.set(scope);
  }

  /** Frames that were approved are no longer waiting. */
  dropClean(keys: string[]) {
    this.cleanFrameKeys.update((all) => all.filter((k) => !keys.includes(k)));
  }

  setMode(mode: AuditMode) {
    if (mode === this.mode()) return;
    this.stopReview();
    this.mode.set(mode);
  }

  isChecked(name: string): boolean {
    return !this.choices().unchecked.includes(name);
  }

  setChecked(names: string[], on: boolean) {
    const rest = this.choices().unchecked.filter((n) => !names.includes(n));
    this.setChoices({ ...this.choices(), unchecked: on ? rest : [...rest, ...names] });
  }

  groupOf(name: string): number {
    return this.choices().groups[name] ?? 0;
  }

  setGroup(name: string, group: number) {
    const groups = { ...this.choices().groups };
    if (group > 0) groups[name] = group;
    else delete groups[name];
    this.setChoices({ ...this.choices(), groups });
  }

  private setChoices(choices: AuditKeypointChoices) {
    this.choices.set(choices);
    try {
      localStorage.setItem(this.choicesStorageKey(), JSON.stringify(choices));
    } catch {
      // Storage is a convenience only.
    }
  }

  setAnimals(text: string) {
    this.animals.set(text);
    try {
      localStorage.setItem(this.animalsStorageKey(), text);
    } catch {
      // Storage is a convenience only.
    }
  }

  private animalsStorageKey(): string {
    return `label-audit.animals.${this.projectInfoService.projectContext()?.key ?? ''}`;
  }

  private storedAnimals(): string {
    try {
      return localStorage.getItem(this.animalsStorageKey()) ?? '';
    } catch {
      return '';
    }
  }

  private choicesStorageKey(): string {
    return `${CHOICES_KEY}.${this.projectInfoService.projectContext()?.key ?? ''}`;
  }

  private storedChoices(): AuditKeypointChoices {
    try {
      const raw = JSON.parse(localStorage.getItem(this.choicesStorageKey()) ?? 'null');
      if (raw && Array.isArray(raw.unchecked) && raw.groups && typeof raw.groups === 'object') {
        return { unchecked: raw.unchecked.map(String), groups: raw.groups };
      }
    } catch {
      // fall through to the default
    }
    return { unchecked: [], groups: {} };
  }

  /** Forget the findings, for example when another label file opens. */
  clear() {
    this.stopReview();
    this.serverFindings = [];
    this.checkedFrames = [];
    this.allFindings.set([]);
    this.summary.set(null);
    this.error.set(null);
    this.unmatched.set(0);
    this.cleanFrameKeys.set([]);
    this.auditedLabelFileKey.set(null);
  }

  startReview(index = 0) {
    if (!this.findings().length) return;
    this.visited.set(new Set());
    this.goTo(index);
  }

  goTo(index: number) {
    const n = this.findings().length;
    if (!n) return;
    const i = Math.max(0, Math.min(n - 1, index));
    this.reviewIndex.set(i);
    this.visited.update((s) => new Set(s).add(i));
  }

  next() {
    const i = this.reviewIndex();
    if (i !== null && i < this.findings().length - 1) this.goTo(i + 1);
  }

  previous() {
    const i = this.reviewIndex();
    if (i !== null && i > 0) this.goTo(i - 1);
  }

  stopReview() {
    this.reviewIndex.set(null);
  }
}

/** Attach the Labeler frame key to each item; items without a loaded frame are dropped. */
export function matchFindingsToFrames(
  items: GetLabelAuditResponse['items'],
  frames: MVFrame[],
): AuditFinding[] {
  const frameOfImg = new Map<string, string>();
  for (const frame of frames) {
    for (const view of frame.views) frameOfImg.set(view.imgPath, frame.key);
  }
  const out: AuditFinding[] = [];
  for (const item of items) {
    const frameKey = frameOfImg.get(item.imgPaths[item.view]);
    if (frameKey) out.push({ ...item, frameKey });
  }
  return out;
}

function errorText(e: unknown): string {
  const detail = (e as { error?: { detail?: unknown } })?.error?.detail;
  if (typeof detail === 'string') return detail;
  if (e instanceof Error) return e.message;
  return String(e);
}
