/** Wire types of getLabelAudit and getEpipolarGuides (routes/labeler/label_audit.py). */

export type LabelAuditKind = 'multiview' | 'swap' | 'shape';
/** The two audits: "views" (the cameras disagree, swaps) and "shape" (off a rigid group). */
export type AuditMode = 'views' | 'shape';

export interface LabelAuditItem {
  kind: LabelAuditKind;
  /** The view whose label is probably wrong. */
  view: string;
  keypoint: string;
  /** The other keypoint of a left/right swap, or of a single off distance (shape). */
  partner: string | null;
  /** Image path of the frame in every view (label CSV index). */
  imgPaths: Record<string, string>;
  /** px for views findings; shape: the worst pair's difference from usual and its limit (calibration units). */
  errorPx: number;
  tolerancePx: number;
  score: number;
  reason: string;
  /** shape: the rigid group in this frame, reviewed as a whole, and all its pairs. */
  group?: string[];
  groupPairs?: ShapePair[];
  /** shape: how many of the keypoint's measurable distances are off. */
  offPairs?: number;
  checkedPairs?: number;
}

/** A 3D arrangement of a rigid group that has its learned distances. */
export interface ShapeTemplate {
  animal: string | null;
  keypoints: string[];
  /** One [x, y, z] per keypoint, in the units of the calibration. */
  points: number[][];
}

/** The 3D distance between two keypoints of a rigid group, in the units of the calibration. */
export interface ShapePair {
  a: string;
  b: string;
  /** Median over all frames, its robust spread, and the allowed difference from usual. */
  usual: number;
  spread: number;
  limit: number;
  frames: number;
  /** Whose distances these are, when animals are named. */
  animal?: string | null;
  /** In this frame; null when it cannot be measured (a label missing, or its views disagree). */
  distance: number | null;
  off: boolean;
}

export interface GetLabelAuditResponse {
  items: LabelAuditItem[];
  framesChecked: number;
  pointsChecked: number;
  sessionsWithoutCalibration: string[];
  /** Rigid groups that could be checked. */
  shapeGroups?: string[][];
  /** Why a rigid group, or part of it, was not checked. */
  shapeProblems?: string[];
  /** The learned distances of all rigid groups. */
  shapePairs?: ShapePair[];
  /** Size of the rigid groups per session, relative to usual; distances are divided by it. */
  shapeSessionSize?: Record<string, number>;
  /** With animals named: sessions per animal, and sessions that match no name. */
  shapeAnimalSessions?: Record<string, number>;
  shapeUnmatchedSessions?: string[];
  /** One 3D arrangement per rigid group and animal, to draw it. */
  shapeTemplates?: ShapeTemplate[];
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

/** A rigid group measured with one frame's labels as they are on screen (getShapeGuides). */
export interface ShapeGuides {
  group: string[];
  groupPairs: ShapePair[];
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
