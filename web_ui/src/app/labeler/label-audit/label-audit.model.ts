/** Wire types of getLabelAudit and getEpipolarGuides (routes/labeler/label_audit.py). */

export type LabelAuditKind = 'multiview' | 'swap' | 'shape' | 'midline' | 'missing';
/**
 * The audits: "views" (the cameras disagree, swaps), "shape" (off a rigid group),
 * "midline" (a keypoint off the midplane of a left and a right keypoint) and "missing"
 * (every frame where a chosen keypoint has no label; found in the browser, not by the server).
 */
export type AuditMode = 'views' | 'shape' | 'midline' | 'missing';
/** Missing: in which frames the keypoint counts as missing. */
export type MissingWhere = 'any' | 'all' | 'some';

/**
 * What is checked: the saved labels, or the frames still in the unlabeled queue with the
 * model predictions they were queued with.
 */
export type AuditScope = 'labeled' | 'queue';

/** The audit a finding belongs to. */
export function auditModeOf(kind: LabelAuditKind): AuditMode {
  return kind === 'shape' || kind === 'midline' || kind === 'missing' ? kind : 'views';
}

/** A keypoint on the midline between a left and a right keypoint, and the allowed angle. */
export interface MidlineChoice {
  left: string;
  right: string;
  point: string;
  toleranceDeg: number;
}

/** One frame's midline keypoint against the midplane of left and right (3D). */
export interface MidlineInfo {
  /** Angle between the axis midpoint -> point and the midplane; positive = towards `right`. */
  yawDeg: number | null;
  /** Distance from the midplane, and distance to left minus distance to right (calibration units). */
  off: number | null;
  asymmetry: number | null;
  /** Distance left-right, and its median over the animal's (or all) frames. */
  width: number | null;
  widthUsual?: number | null;
}

/** The midline of one frame with its labels as they are on screen (getMidlineGuides). */
export interface MidlineGuides {
  views: {
    view: string;
    /** The midline in this view, in image pixels. */
    line: GuidePoint[];
    /** Where the other views put the point on the midplane. */
    suggestion: GuidePoint | null;
  }[];
  info: MidlineInfo | null;
}

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
  /** midline: errorPx and tolerancePx are degrees; group = [point, left, right]. */
  midline?: MidlineInfo | null;
  /** missing: where it is missing, for the list ("every view", or the views). */
  missingIn?: string;
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
  /** Midline: frames measured, frames with the point but no midplane, and why it did not run. */
  midlineChecked?: number;
  midlineNoGuide?: number;
  midlineProblems?: string[];
  /** Queue scope: the queue frames (image path per view) that were checked and have no finding. */
  cleanFrames?: Record<string, string>[];
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
