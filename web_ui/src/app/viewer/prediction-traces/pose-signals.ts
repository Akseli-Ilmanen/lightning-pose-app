/** Types for the listPoseSignals / computePoseSignal RPCs (routes/pose_signals.py). */

export interface PoseSignalParam {
  name: string;
  label: string;
  kind: 'keypoints' | 'keypoint' | 'number' | 'choice';
  default?: unknown;
  choices?: string[] | null;
  help?: string;
}

export interface PoseSignalInfo {
  id: string;
  label: string;
  /** e.g. movement.kinematics.compute_speed */
  function: string;
  docUrl: string;
  unit: string;
  params: PoseSignalParam[];
}

export interface ListPoseSignalsResponse {
  movementAvailable: boolean;
  movementVersion?: string | null;
  signals: PoseSignalInfo[];
}

export interface ComputePoseSignalRequest {
  projectKey: string;
  predictionPath: string;
  signal: string;
  params: Record<string, unknown>;
  fps: number | null;
  confidenceThreshold: number | null;
}

export interface ComputePoseSignalResponse {
  signal: string;
  unit: string;
  nFrames: number;
  series: {
    label: string;
    values: (number | null)[];
    keypoint?: string | null;
    space?: string | null;
  }[];
}

/** Predictions of one model for one view, as the Viewer has loaded them. */
export interface TraceSource {
  /** `${modelKey}/${viewName}` */
  key: string;
  modelKey: string;
  viewName: string;
  /** Prediction CSV path relative to the model directory. */
  path: string;
  keypoints: string[];
  /** Per-frame column, or null when the file has no such column. */
  column(keypoint: string, coord: 'x' | 'y' | 'likelihood'): ArrayLike<number> | null;
}

/** A k-means extract request for a marked segment. */
export interface SegmentExtractRequest {
  segment: [number, number];
  nFrames: number;
  /** kmeans: diverse frames. uniform: evenly spaced across the segment. */
  selection: 'kmeans' | 'uniform';
}

/** Signals computed in the browser from the loaded predictions. */
export const BUILTIN_SIGNALS = [
  { id: 'position', label: 'Position', unit: 'px' },
  { id: 'confidence', label: 'Confidence (heatmap)', unit: '' },
] as const;

/** Keypoints that define the head-direction arrow drawn on the videos. */
export interface HeadingOverlay {
  modelKey: string;
  left: string;
  right: string;
  cameraView: 'top_down' | 'bottom_up';
}
