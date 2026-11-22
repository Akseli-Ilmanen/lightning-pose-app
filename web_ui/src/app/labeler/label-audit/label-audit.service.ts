import { computed, inject, Injectable, signal } from '@angular/core';
import { RpcService } from '../../rpc.service';
import { ProjectInfoService } from '../../project-info.service';
import { MVLabelFile } from '../../label-file.model';
import { MVFrame } from '../frame.model';
import {
  AuditFinding,
  AuditFocus,
  AuditMode,
  GetLabelAuditResponse,
  ShapeGuides,
} from './label-audit.model';

export const DEFAULT_AUDIT_TOLERANCE_PX = 5;
const CHOICES_KEY = 'label-audit.keypoints';
export const MAX_RIGID_GROUPS = 4;
export const DEFAULT_SHAPE_STRICTNESS = 3;

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
  /**
   * Views check: only compare labels that exist in two or more views. Off: keypoints
   * labeled in some views and missing in another are listed too, to place them on the
   * epipolar line.
   */
  readonly onlyTwoOrMoreViews = signal(true);
  /** Keypoints to report and their rigid groups. Remembered per project. */
  readonly choices = signal<AuditKeypointChoices>(this.storedChoices());
  readonly isRunning = signal(false);
  readonly error = signal<string | null>(null);
  /** Label file key the findings belong to; null before the first run. */
  readonly auditedLabelFileKey = signal<string | null>(null);
  /** Findings of both audits, as the last check returned them. */
  readonly allFindings = signal<AuditFinding[]>([]);
  /** The audit shown: its settings, findings and review. */
  readonly mode = signal<AuditMode>('views');
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
  readonly viewsCount = computed(() => this.allFindings().filter((f) => f.kind !== 'shape').length);
  /** Views findings that are mistakes for certain (not the "one view" ones). */
  readonly disagreeCount = computed(
    () => this.allFindings().filter((f) => f.kind === 'multiview' || f.kind === 'swap').length,
  );
  readonly shapeCount = computed(() => this.allFindings().filter((f) => f.kind === 'shape').length);
  /** Findings of the shown audit; the review steps through these. */
  readonly findings = computed(() =>
    this.allFindings().filter((f) => (f.kind === 'shape') === (this.mode() === 'shape')),
  );
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
    if (f.kind === 'shape' && f.group?.length) return f.group;
    return f.partner ? [f.keypoint, f.partner] : [f.keypoint];
  });

  async run(labelFile: MVLabelFile, frames: MVFrame[]): Promise<void> {
    const projectKey = this.projectInfoService.projectContext()?.key;
    if (!projectKey) throw new Error('Project key missing from project context');
    this.isRunning.set(true);
    this.error.set(null);
    this.stopReview();
    this.lastRun = { labelFile, names: keypointNamesOf(frames) };
    const names = keypointNamesOf(frames);
    const { unchecked, groups } = this.choices();
    const checked = names.filter((n) => !unchecked.includes(n));
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
        onlyTwoOrMoreViews: this.onlyTwoOrMoreViews(),
      })) as GetLabelAuditResponse;
      const { items, ...summary } = response;
      const findings = matchFindingsToFrames(items, frames);
      this.allFindings.set(findings);
      this.unmatched.set(items.length - findings.length);
      this.summary.set(summary);
      this.auditedLabelFileKey.set(labelFile.key);
    } catch (e) {
      this.allFindings.set([]);
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
      imgPaths,
      labels,
      keypoint,
    })) as ShapeGuides;
    return guides.group.length ? guides : null;
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
    this.allFindings.set([]);
    this.summary.set(null);
    this.error.set(null);
    this.unmatched.set(0);
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
