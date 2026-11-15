import { HeadingOverlay, TraceSource } from './pose-signals';

export interface HeadingArrow {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** Arrowhead points. */
  head: string;
}

/**
 * Arrow from the midpoint of the left and right keypoints along movement's
 * forward vector: (right→left) × up, with up = -z for top_down and +z for
 * bottom_up. Null when either keypoint is missing or below the threshold.
 */
export function headingArrow(
  source: TraceSource,
  overlay: HeadingOverlay,
  frame: number,
  threshold: number,
): HeadingArrow | null {
  const at = (kp: string, coord: 'x' | 'y' | 'likelihood') =>
    source.column(kp, coord)?.[frame] ?? NaN;
  const lx = at(overlay.left, 'x');
  const ly = at(overlay.left, 'y');
  const rx = at(overlay.right, 'x');
  const ry = at(overlay.right, 'y');
  if (![lx, ly, rx, ry].every(Number.isFinite)) return null;
  for (const kp of [overlay.left, overlay.right]) {
    const likelihood = at(kp, 'likelihood');
    if (Number.isFinite(likelihood) && likelihood < threshold) return null;
  }
  const vx = lx - rx;
  const vy = ly - ry;
  const span = Math.hypot(vx, vy);
  if (span === 0) return null;
  const sign = overlay.cameraView === 'top_down' ? 1 : -1;
  const fx = (-sign * vy) / span;
  const fy = (sign * vx) / span;
  const length = Math.min(Math.max(span * 2, 30), 200);
  const x1 = (lx + rx) / 2;
  const y1 = (ly + ry) / 2;
  const x2 = x1 + fx * length;
  const y2 = y1 + fy * length;
  const h = Math.min(12, length / 3);
  const head = [
    [x2, y2],
    [x2 - fx * h - fy * h * 0.5, y2 - fy * h + fx * h * 0.5],
    [x2 - fx * h + fy * h * 0.5, y2 - fy * h - fx * h * 0.5],
  ]
    .map(([x, y]) => `${x},${y}`)
    .join(' ');
  return { x1, y1, x2, y2, head };
}
