import { headingArrow } from './heading-arrow';
import { HeadingOverlay, TraceSource } from './pose-signals';

/** One frame: left ear at (lx, ly), right ear at (rx, ry). */
function source(lx: number, ly: number, rx: number, ry: number, likelihood = 1): TraceSource {
  const data: Record<string, number[]> = {
    'ear_l|x': [lx],
    'ear_l|y': [ly],
    'ear_l|likelihood': [likelihood],
    'ear_r|x': [rx],
    'ear_r|y': [ry],
    'ear_r|likelihood': [likelihood],
  };
  return {
    key: 'm/cam',
    modelKey: 'm',
    viewName: 'cam',
    path: '',
    keypoints: ['ear_l', 'ear_r'],
    column: (kp, coord) => data[`${kp}|${coord}`] ?? null,
  };
}

const overlay: HeadingOverlay = {
  modelKey: 'm',
  left: 'ear_l',
  right: 'ear_r',
  cameraView: 'top_down',
};

describe('headingArrow', () => {
  it('starts between the ears and points forward (top-down, image y down)', () => {
    // Animal facing up the image: left ear on the left, right ear on the right.
    const a = headingArrow(source(90, 100, 110, 100), overlay, 0, 0.5)!;
    expect(a.x1).toBe(100);
    expect(a.y1).toBe(100);
    expect(a.x2).toBeCloseTo(100);
    expect(a.y2).toBeLessThan(100);
  });

  it('flips for a bottom-up camera', () => {
    const a = headingArrow(
      source(90, 100, 110, 100),
      { ...overlay, cameraView: 'bottom_up' },
      0,
      0.5,
    )!;
    expect(a.y2).toBeGreaterThan(100);
  });

  it('is hidden when a keypoint is below the likelihood threshold', () => {
    expect(headingArrow(source(90, 100, 110, 100, 0.2), overlay, 0, 0.5)).toBeNull();
  });

  it('is hidden when a keypoint is missing', () => {
    expect(headingArrow(source(NaN, 100, 110, 100), overlay, 0, 0.5)).toBeNull();
  });
});
