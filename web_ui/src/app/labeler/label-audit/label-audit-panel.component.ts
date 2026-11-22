import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
} from '@angular/core';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MVLabelFile } from '../../label-file.model';
import { MVFrame } from '../frame.model';
import {
  keypointNamesOf,
  LabelAuditService,
  MAX_RIGID_GROUPS,
  rigidGroupsOf,
} from './label-audit.service';
import { AuditFinding, ShapeTemplate } from './label-audit.model';
import { ShapeTemplateViewComponent } from './shape-template-view.component';

/**
 * Left-pane panel of the Labeler: runs the label audit on the open label file
 * and lists the likely mistakes. Clicking one opens it for review.
 */
@Component({
  selector: 'app-label-audit-panel',
  imports: [FormsModule, DecimalPipe, PercentPipe, ShapeTemplateViewComponent],
  templateUrl: './label-audit-panel.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LabelAuditPanelComponent {
  protected audit = inject(LabelAuditService);

  labelFile = input<MVLabelFile | null>(null);
  /** Labeled frames of the open label file. */
  frames = input<MVFrame[]>([]);

  protected keypointNames = computed(() => keypointNamesOf(this.frames()));
  protected groupNumbers = Array.from({ length: MAX_RIGID_GROUPS }, (_, i) => i + 1);
  protected checkedCount = computed(
    () => this.keypointNames().filter((n) => !this.audit.choices().unchecked.includes(n)).length,
  );
  protected rigidGroupCount = computed(
    () => rigidGroupsOf(this.audit.choices().groups, this.keypointNames()).length,
  );

  protected canRun = computed(
    () =>
      this.checkedCount() > 0 &&
      !!this.labelFile() &&
      (this.labelFile()?.views.length ?? 0) >= 2 &&
      this.frames().length > 0 &&
      !this.audit.isRunning(),
  );

  protected runDisabledReason = computed(() => {
    const lf = this.labelFile();
    if (!lf) return 'Open a label file first.';
    if (lf.views.length < 2) return 'The audit needs a multiview label file.';
    if (!this.frames().length) return 'This label file has no labeled frames.';
    if (!this.checkedCount()) return 'Tick at least one keypoint.';
    if (this.audit.isRunning()) return 'Checking…';
    return '';
  });

  /** Findings are stale when another label file is open. */
  protected showFindings = computed(
    () =>
      this.audit.auditedLabelFileKey() !== null &&
      this.audit.auditedLabelFileKey() === this.labelFile()?.key,
  );

  protected run() {
    const lf = this.labelFile();
    if (!lf || !this.canRun()) return;
    void this.audit.run(lf, this.frames());
  }

  protected frameName(f: AuditFinding): string {
    return f.imgPaths[f.view]?.split('/').pop() ?? f.frameKey;
  }

  protected kindLabel(f: AuditFinding): string {
    if (f.kind === 'shape') return 'distance';
    if (f.kind === 'oneview') return 'one view';
    return f.kind === 'swap' ? 'swap' : 'views';
  }

  /** The learned pairs that belong to a drawn group. */
  protected pairsOf(t: ShapeTemplate) {
    const members = new Set(t.keypoints);
    return (this.audit.summary()?.shapePairs ?? []).filter(
      (p) => (p.animal ?? null) === t.animal && members.has(p.a) && members.has(p.b),
    );
  }

  /** Sessions sorted by the size of their rigid groups (animals differ in size). */
  protected sessionSizes = computed(() =>
    Object.entries(this.audit.summary()?.shapeSessionSize ?? {})
      .map(([session, size]) => ({ session, size }))
      .sort((a, b) => a.size - b.size),
  );

  /** Three significant digits: distances are in calibration units of unknown size. */
  protected sig(x: number): string {
    return Number(x.toPrecision(3)).toString();
  }

  protected onStrictness(value: number | string) {
    const v = Number(value);
    if (Number.isFinite(v) && v >= 1) this.audit.shapeStrictness.set(v);
  }

  protected onTolerance(value: number | string) {
    const v = Number(value);
    if (Number.isFinite(v) && v > 0) this.audit.tolerancePx.set(v);
  }
}
