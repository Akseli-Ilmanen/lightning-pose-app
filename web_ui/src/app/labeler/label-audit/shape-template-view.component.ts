import { ChangeDetectionStrategy, Component, computed, input, signal } from '@angular/core';
import { ShapePair, ShapeTemplate } from './label-audit.model';

/**
 * One rigid group of the label audit drawn in 3D: its keypoints, and the usual
 * distance on every edge. Drag to rotate, wheel to zoom (double click resets); click a
 * keypoint to see only its distances.
 */
@Component({
  selector: 'app-shape-template-view',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <svg
      viewBox="-100 -100 200 200"
      class="w-full aspect-square text-base-content select-none touch-none cursor-grab overflow-hidden"
      data-testid="shape-template-view"
      role="img"
      [attr.aria-label]="'Rigid group' + (template().animal ? ' of ' + template().animal : '')"
      (pointerdown)="startDrag($event)"
      (pointermove)="drag($event)"
      (pointerup)="dragging = false"
      (pointercancel)="dragging = false"
      (wheel)="zoomBy($event)"
      (dblclick)="zoom.set(1)"
    >
      @for (e of edges(); track e.key) {
        <line
          [attr.x1]="e.x1"
          [attr.y1]="e.y1"
          [attr.x2]="e.x2"
          [attr.y2]="e.y2"
          stroke="currentColor"
          [attr.stroke-width]="e.shown ? 0.8 : 0.4"
          [attr.opacity]="e.shown ? 0.55 : 0.12"
        />
      }
      @for (e of edges(); track e.key) {
        @if (e.shown) {
          <text
            [attr.x]="(e.x1 + e.x2) / 2"
            [attr.y]="(e.y1 + e.y2) / 2"
            text-anchor="middle"
            dominant-baseline="middle"
            font-size="6"
            fill="currentColor"
            stroke="var(--color-base-100, white)"
            stroke-width="2"
            paint-order="stroke"
            [attr.data-edge]="e.key"
          >
            {{ e.label }}
          </text>
        }
      }
      @for (p of points(); track p.name) {
        <g class="cursor-pointer" (click)="toggle(p.name)" [attr.data-keypoint]="p.name">
          <circle
            [attr.cx]="p.x"
            [attr.cy]="p.y"
            [attr.r]="p.r"
            [attr.fill]="selected() === p.name ? 'var(--color-accent, #f59e0b)' : 'currentColor'"
          />
          <text
            [attr.x]="p.x"
            [attr.y]="p.y - p.r - 2"
            text-anchor="middle"
            font-size="7"
            font-weight="600"
            fill="currentColor"
            stroke="var(--color-base-100, white)"
            stroke-width="2.5"
            paint-order="stroke"
          >
            {{ p.name }}
          </text>
        </g>
      }
    </svg>
  `,
})
export class ShapeTemplateViewComponent {
  template = input.required<ShapeTemplate>();
  /** The learned pairs of this group and animal. */
  pairs = input<ShapePair[]>([]);

  private yaw = signal(0.6);
  private pitch = signal(-0.35);
  /** Keypoint whose distances are shown; null shows all. */
  protected selected = signal<string | null>(null);
  protected dragging = false;
  /** 1 fits the group in the panel. */
  protected zoom = signal(1);

  protected points = computed(() => {
    const t = this.template();
    const n = t.points.length;
    if (!n) return [];
    const c = [0, 1, 2].map((k) => t.points.reduce((s, p) => s + p[k], 0) / n);
    const centred = t.points.map((p) => p.map((v, k) => v - c[k]));
    const radius = Math.max(...centred.map((p) => Math.hypot(p[0], p[1], p[2]))) || 1;
    const scale = 86 * this.zoom();
    const [cy, sy, cx, sx] = [
      Math.cos(this.yaw()),
      Math.sin(this.yaw()),
      Math.cos(this.pitch()),
      Math.sin(this.pitch()),
    ];
    return centred.map((p, i) => {
      const x = cy * p[0] + sy * p[2];
      const z1 = -sy * p[0] + cy * p[2];
      const y = cx * p[1] - sx * z1;
      const z = sx * p[1] + cx * z1;
      return {
        name: t.keypoints[i],
        x: (x / radius) * scale,
        y: (-y / radius) * scale,
        // Nearer points are drawn larger.
        r: 2.6 + (z / radius) * 0.9,
      };
    });
  });

  protected edges = computed(() => {
    const at = new Map(this.points().map((p) => [p.name, p]));
    const selected = this.selected();
    return this.pairs().flatMap((pair) => {
      const a = at.get(pair.a);
      const b = at.get(pair.b);
      if (!a || !b) return [];
      return [
        {
          key: `${pair.a}|${pair.b}`,
          x1: a.x,
          y1: a.y,
          x2: b.x,
          y2: b.y,
          label: Number(pair.usual.toPrecision(3)).toString(),
          shown: selected === null || pair.a === selected || pair.b === selected,
        },
      ];
    });
  });

  protected toggle(name: string) {
    this.selected.update((s) => (s === name ? null : name));
  }

  protected zoomBy(event: WheelEvent) {
    event.preventDefault();
    this.zoom.update((z) => Math.max(0.5, Math.min(8, z * Math.exp(-event.deltaY * 0.0015))));
  }

  protected startDrag(event: PointerEvent) {
    this.dragging = true;
    (event.currentTarget as Element).setPointerCapture?.(event.pointerId);
  }

  protected drag(event: PointerEvent) {
    if (!this.dragging) return;
    this.yaw.update((v) => v + event.movementX * 0.01);
    this.pitch.update((v) => Math.max(-1.5, Math.min(1.5, v + event.movementY * 0.01)));
  }
}
