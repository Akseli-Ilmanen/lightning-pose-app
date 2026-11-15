/** Wire types of getLabelAudit and getEpipolarGuides (routes/labeler/label_audit.py). */

export type LabelAuditKind = 'multiview' | 'swap';

export interface LabelAuditItem {
  kind: LabelAuditKind;
  /** The view whose label is probably wrong. */
  view: string;
  keypoint: string;
  /** The other keypoint of a left/right swap. */
  partner: string | null;
  /** Image path of the frame in every view (label CSV index). */
  imgPaths: Record<string, string>;
  errorPx: number;
  tolerancePx: number;
  score: number;
  reason: string;
}

export interface GetLabelAuditResponse {
  items: LabelAuditItem[];
  framesChecked: number;
  pointsChecked: number;
  sessionsWithoutCalibration: string[];
}

export interface GuidePoint {
  x: number;
  y: number;
}

export interface EpipolarLine {
  fromView: string;
  segments: GuidePoint[][];
  /** 0 = most useful: lines are ranked so the first ones cross most steeply. */
  rank: number;
  /** Smallest angle to the lines ranked before it, in degrees. */
  crossingAngleDeg: number | null;
}

export interface EpipolarGuides {
  lines: EpipolarLine[];
  /** The keypoint triangulated from the other views, projected into this view. */
  predictedPoint: GuidePoint | null;
  predictedFromViews: string[];
}

/** An audit item matched to a frame the Labeler has loaded. */
export interface AuditFinding extends LabelAuditItem {
  frameKey: string;
}

/** What the center panel shows while reviewing. */
export interface AuditFocus {
  finding: AuditFinding;
  index: number;
  total: number;
}
