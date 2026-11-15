/**
 * Colors of epipolar lines, one per source view (by the view's index in the
 * frame), so a view keeps its color while the user moves between views.
 * Bright on dark video frames; none of them is the yellow of the selected ring.
 */
const GUIDE_COLORS = ['#22d3ee', '#f472b6', '#a3e635', '#fb923c', '#c084fc', '#f87171', '#34d399', '#60a5fa'];

export function guideColor(viewIndex: number): string {
  return GUIDE_COLORS[((viewIndex % GUIDE_COLORS.length) + GUIDE_COLORS.length) % GUIDE_COLORS.length];
}
