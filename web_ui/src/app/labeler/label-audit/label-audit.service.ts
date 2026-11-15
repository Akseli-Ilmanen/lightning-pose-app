import { computed, inject, Injectable, signal } from '@angular/core';
import { RpcService } from '../../rpc.service';
import { ProjectInfoService } from '../../project-info.service';
import { MVLabelFile } from '../../label-file.model';
import { MVFrame } from '../frame.model';
import {
  AuditFinding,
  AuditFocus,
  GetLabelAuditResponse,
} from './label-audit.model';

export const DEFAULT_AUDIT_TOLERANCE_PX = 5;

/**
 * Label audit state for one Labeler page (#207): the findings of the last run
 * and where the review is. Provided by LabelerPageComponent.
 */
@Injectable()
export class LabelAuditService {
  private rpc = inject(RpcService);
  private projectInfoService = inject(ProjectInfoService);

  readonly tolerancePx = signal(DEFAULT_AUDIT_TOLERANCE_PX);
  readonly isRunning = signal(false);
  readonly error = signal<string | null>(null);
  /** Label file key the findings belong to; null before the first run. */
  readonly auditedLabelFileKey = signal<string | null>(null);
  readonly findings = signal<AuditFinding[]>([]);
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
    return f.partner ? [f.keypoint, f.partner] : [f.keypoint];
  });

  async run(labelFile: MVLabelFile, frames: MVFrame[]): Promise<void> {
    const projectKey = this.projectInfoService.projectContext()?.key;
    if (!projectKey) throw new Error('Project key missing from project context');
    this.isRunning.set(true);
    this.error.set(null);
    this.stopReview();
    try {
      const response = (await this.rpc.call('getLabelAudit', {
        projectKey,
        views: labelFile.views.map((v) => ({
          viewName: v.viewName,
          csvPath: v.csvPath,
        })),
        defaultTolerancePx: this.tolerancePx(),
      })) as GetLabelAuditResponse;
      const { items, ...summary } = response;
      const findings = matchFindingsToFrames(items, frames);
      this.findings.set(findings);
      this.unmatched.set(items.length - findings.length);
      this.summary.set(summary);
      this.auditedLabelFileKey.set(labelFile.key);
    } catch (e) {
      this.findings.set([]);
      this.summary.set(null);
      this.error.set(errorText(e));
    } finally {
      this.isRunning.set(false);
    }
  }

  /** Forget the findings, for example when another label file opens. */
  clear() {
    this.stopReview();
    this.findings.set([]);
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
