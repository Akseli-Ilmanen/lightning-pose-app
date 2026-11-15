import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
} from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MVLabelFile } from '../../label-file.model';
import { MVFrame } from '../frame.model';
import { LabelAuditService } from './label-audit.service';
import { AuditFinding } from './label-audit.model';

/**
 * Left-pane panel of the Labeler: runs the label audit on the open label file
 * and lists the likely mistakes. Clicking one opens it for review.
 */
@Component({
  selector: 'app-label-audit-panel',
  imports: [FormsModule, DecimalPipe],
  templateUrl: './label-audit-panel.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LabelAuditPanelComponent {
  protected audit = inject(LabelAuditService);

  labelFile = input<MVLabelFile | null>(null);
  /** Labeled frames of the open label file. */
  frames = input<MVFrame[]>([]);

  protected canRun = computed(
    () =>
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
    return f.kind === 'swap' ? 'swap' : 'views';
  }

  protected onTolerance(value: number | string) {
    const v = Number(value);
    if (Number.isFinite(v) && v > 0) this.audit.tolerancePx.set(v);
  }
}
