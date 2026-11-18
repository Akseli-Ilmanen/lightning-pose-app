"""Label audit: find labels that are probably wrong, and epipolar guides for fixing them (#207).

``getLabelAudit`` checks every labeled frame of a multiview label file against the
camera calibration. No model is needed:

- **multiview**: the keypoint is labeled in two or more views, but the views do not
  describe the same 3D point. With three or more views, each view's label is
  compared with the point triangulated from the *other* views, so the view that
  is off is the one flagged. With two views the check cannot tell which one is off.
- **swap**: a left/right pair (``L1A``/``R1A``, ``leftEye``/``rightEye``, ``leye``/``reye``)
  agrees with the other views much better after swapping the two labels in one view.
- **shape**: the keypoint breaks the distances of a rigid group of keypoints the user
  named (a head, an implant). The usual 3D distance of every pair of the group is
  learned from the labels (``label_audit_shape``); the keypoint in the most off pairs
  is reported, with the distances of the whole group in that frame.

``getShapeGuides`` measures one frame's rigid group with the labels as they are on screen
(unsaved edits included), so the distances follow the user's corrections.

``getEpipolarGuides`` returns, for one keypoint in one view, the epipolar lines of the
other views' labels and the point they agree on.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from aniposelib.cameras import CameraGroup
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from litpose_app import deps
from litpose_app.datatypes import Project
from litpose_app.deps import ProjectInfoGetter
from litpose_app.routes.labeler import find_calibration_file
from litpose_app.routes.labeler.label_audit_shape import (
    MIN_GROUP_SIZE,
    blame,
    learn_distances,
    learn_scaled_distances,
    session_sizes,
    template_points,
    pair_scores,
)
from litpose_app.session_view import split_session_view
from litpose_app.utils.fix_empty_first_row import fix_empty_first_row

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE_PX = 5.0
# A swap is reported when swapping brings both labels within tolerance and at
# least halves their combined error.
SWAP_GAIN = 0.5
# Rigid groups: a pair of keypoints is off when its 3D distance is this many spreads
# from its usual distance.
SHAPE_SPREAD_FACTOR = 3.0

# ---------------------------------------------------------------- models
class AuditLabelFileView(BaseModel):
    """One view's label CSV, as the Labeler lists it."""

    viewName: str
    csvPath: Path


class GetLabelAuditRequest(BaseModel):
    projectKey: str
    views: list[AuditLabelFileView]
    defaultTolerancePx: float = DEFAULT_TOLERANCE_PX
    # Per keypoint tolerance in px; keypoints not listed use defaultTolerancePx.
    tolerancePx: dict[str, float] = {}
    # Groups of keypoints that keep their distances to each other (3 or more each).
    rigidGroups: list[list[str]] = []
    # Report only these keypoints (None: all). The others still take part in the
    # checks, e.g. as members of a rigid group.
    keypoints: list[str] | None = None
    # Rigid groups: a pair is off when its distance is this many spreads from usual
    # (lower = stricter).
    shapeStrictness: float = SHAPE_SPREAD_FACTOR
    # Rigid groups: names of the animals, as they occur in the session names. Each animal
    # gets its own distances. Empty: one set of distances, every session at its own size.
    animals: list[str] = []


class ShapePair(BaseModel):
    """The 3D distance between two keypoints of a rigid group (calibration units)."""

    a: str
    b: str
    usual: float  # median over all frames
    spread: float  # robust spread over all frames
    limit: float  # off when |distance - usual| exceeds this
    frames: int = 0  # frames the pair was learned from
    animal: str | None = None  # whose distances these are, when animals are named
    distance: float | None = None  # in this frame; None when it cannot be measured
    off: bool = False


class ShapeTemplate(BaseModel):
    """A 3D arrangement of a rigid group that has its learned distances (to draw it)."""

    animal: str | None = None
    keypoints: list[str]
    points: list[list[float]]  # one [x, y, z] per keypoint, calibration units


class LabelAuditItem(BaseModel):
    """One label that is probably wrong."""

    kind: str  # "multiview" | "swap" | "shape"
    view: str  # the view whose label is probably wrong (shape: not known, the first view)
    keypoint: str
    partner: str | None = None  # the other keypoint of a swap, or of a single off pair
    # Image path of this frame in every view (CSV index), so the client can find the frame.
    imgPaths: dict[str, str]
    # multiview: distance to the other views' point; swap: error before swapping (px).
    # shape: the worst pair's difference from usual and its limit (calibration units).
    errorPx: float
    tolerancePx: float
    score: float  # errorPx / tolerancePx; sorts items within a frame
    reason: str
    # shape: the rigid group in this frame, to review it as a whole, and all its pairs.
    group: list[str] = []
    groupPairs: list[ShapePair] = []
    # shape: how many of the keypoint's measurable pairs are off.
    offPairs: int = 0
    checkedPairs: int = 0


class GetShapeGuidesRequest(GetLabelAuditRequest):
    """The audit's settings, plus one frame as it is on screen."""

    # Image path of the frame in every view (CSV index).
    imgPaths: dict[str, str]
    # The frame's labels now, per view and keypoint: [x, y], or null where not placed.
    labels: dict[str, dict[str, list[float] | None]]
    # A keypoint of the rigid group to refit.
    keypoint: str


class GetShapeGuidesResponse(BaseModel):
    """As the group fields of a shape finding; empty when the group is not known."""

    group: list[str] = []
    groupPairs: list[ShapePair] = []


class GetLabelAuditResponse(BaseModel):
    items: list[LabelAuditItem]
    framesChecked: int
    pointsChecked: int  # keypoint x frame with labels in 2+ views
    sessionsWithoutCalibration: list[str]
    # Rigid groups that could be checked (keypoints with a learned distance).
    shapeGroups: list[list[str]] = []
    # Why a requested rigid group, or part of it, was not checked.
    shapeProblems: list[str] = []
    # The learned pairs of all rigid groups (no per-frame fields).
    shapePairs: list[ShapePair] = []
    # Size of the rigid groups per session, relative to usual (animals differ in size);
    # distances are divided by it before they are compared.
    shapeSessionSize: dict[str, float] = {}
    # With animals named: sessions per animal, and the sessions that match no name
    # (those are compared among themselves).
    shapeAnimalSessions: dict[str, int] = {}
    shapeUnmatchedSessions: list[str] = []
    # One arrangement per rigid group and animal.
    shapeTemplates: list[ShapeTemplate] = []


class GuideLabel(BaseModel):
    view: str
    x: float
    y: float


class GetEpipolarGuidesRequest(BaseModel):
    projectKey: str
    sessionKey: str
    targetView: str
    # Labels of the keypoint in the other views (the target view's own label is ignored).
    labels: list[GuideLabel]
    # Size of the target image as shown, to clip the lines. Defaults to the calibration size.
    imageWidth: float | None = None
    imageHeight: float | None = None
    # The keypoint's label in the target view, if any: where line angles are measured
    # when fewer than two other views can triangulate it.
    targetPoint: GuidePoint | None = None


class GuidePoint(BaseModel):
    x: float
    y: float


class EpipolarLine(BaseModel):
    fromView: str
    segments: list[list[GuidePoint]]
    # 0 = most useful. Lines that cross the others at a wide angle pin the point down;
    # near-parallel lines (similar cameras) add little. Show the first N.
    rank: int = 0
    # Smallest angle (deg) between this line and the lines ranked before it, at the point.
    crossingAngleDeg: float | None = None


class GetEpipolarGuidesResponse(BaseModel):
    lines: list[EpipolarLine]
    # Triangulated from the other views (2+ needed), projected into the target view.
    predictedPoint: GuidePoint | None = None
    predictedFromViews: list[str] = []


# ---------------------------------------------------------------- left/right pairs
_LR_WORDS = [("left", "right"), ("Left", "Right"), ("LEFT", "RIGHT")]


def _mirror_candidates(name: str) -> set[str]:
    out = set()
    for a, b in _LR_WORDS:
        if a in name:
            out.add(name.replace(a, b))
        if b in name:
            out.add(name.replace(b, a))
    # single letter L/R or l/r at the start or end, or next to a separator
    for a, b in [("L", "R"), ("l", "r")]:
        for x, y in [(a, b), (b, a)]:
            if name.startswith(x):
                out.add(y + name[1:])
            if name.endswith(x):
                out.add(name[:-1] + y)
            out.update(
                name[: m.start(1)] + y + name[m.end(1) :]
                for m in re.finditer(rf"[_\-.]({x})(?=[_\-.]|$)", name)
            )
    out.discard(name)
    return out


def left_right_pairs(keypoints: list[str]) -> list[tuple[str, str]]:
    """Pairs of keypoints whose names differ only by left/right."""
    names = set(keypoints)
    pairs = set()
    for k in keypoints:
        for c in _mirror_candidates(k):
            if c in names:
                pairs.add(tuple(sorted((k, c))))
    return sorted(pairs)


# ---------------------------------------------------------------- audit
def _read_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    return fix_empty_first_row(df)


def _frame_id(img_path: str, view_names: list[str]) -> tuple[str, str] | None:
    """(session, image file name) for labeled-data/<session>_<view>/<img>."""
    parts = img_path.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None
    sv = split_session_view(parts[-2], view_names)
    if sv is None:
        return None
    return sv[0], parts[-1]


class _Rig:
    """A camera group with the labels undistorted once, for fast batched triangulation.

    Same linear (DLT) triangulation as ``aniposelib.cameras.triangulate_simple``, but
    for all points at once instead of one by one.
    """

    def __init__(self, cg: CameraGroup):
        self.cg = cg
        self.P = np.stack([c.get_extrinsics_mat()[:3] for c in cg.cameras])  # (C, 3, 4)

    def undistort(self, pts: np.ndarray) -> np.ndarray:
        """(C, N, 2) pixels -> normalized image coordinates (NaN stays NaN)."""
        out = np.full_like(pts, np.nan)
        for c, cam in enumerate(self.cg.cameras):
            ok = ~np.isnan(pts[c]).any(axis=1)
            if ok.any():
                out[c, ok] = np.asarray(cam.undistort_points(pts[c, ok].astype(np.float64))).reshape(-1, 2)
        return out

    def triangulate(self, und: np.ndarray, use: np.ndarray) -> np.ndarray:
        """und: (C, N, 2) normalized points; use: (C, N) bool. Returns (N, 3); NaN with < 2 views."""
        C, N, _ = und.shape
        x, y = np.nan_to_num(und[..., 0]), np.nan_to_num(und[..., 1])
        w = use.astype(float)
        rows1 = (x[..., None] * self.P[:, None, 2] - self.P[:, None, 0]) * w[..., None]  # (C, N, 4)
        rows2 = (y[..., None] * self.P[:, None, 2] - self.P[:, None, 1]) * w[..., None]
        A = np.concatenate([rows1, rows2], axis=0).transpose(1, 0, 2)  # (N, 2C, 4)
        out = np.full((N, 3), np.nan)
        ok = use.sum(0) >= 2
        if ok.any():
            vh = np.linalg.svd(A[ok])[2][:, -1]
            with np.errstate(divide="ignore", invalid="ignore"):
                out[ok] = vh[:, :3] / vh[:, 3:4]
        return out

    def project(self, c: int, p3: np.ndarray) -> np.ndarray:
        out = np.full((len(p3), 2), np.nan)
        ok = np.isfinite(p3).all(axis=1)
        if ok.any():
            out[ok] = np.asarray(self.cg.cameras[c].project(p3[ok])).reshape(-1, 2)
        return out


def _loo_errors(rig: _Rig, pts: np.ndarray, with_residual: bool = False):
    """Distance of each view's label to the point triangulated from the other views.

    pts: (C, N, 2) with NaN where not labeled. Returns err (C, N); NaN where the view has
    no label or fewer than two other views do. With ``with_residual`` also returns how
    well the *other* views agree among themselves (C, N): their largest reprojection
    error. The view whose removal leaves the others agreeing best is the one that is off.
    """
    C, N, _ = pts.shape
    err = np.full((C, N), np.nan)
    resid = np.full((C, N), np.nan)
    labeled = ~np.isnan(pts).any(axis=2)
    und = rig.undistort(pts)
    for c in range(C):
        use = labeled.copy()
        use[c] = False
        ok = labeled[c] & (use.sum(0) >= 2)
        if not ok.any():
            continue
        p3 = rig.triangulate(und[:, ok], use[:, ok])
        err[c, ok] = np.hypot(*(rig.project(c, p3) - pts[c, ok]).T)
        if with_residual:
            e = np.stack([np.hypot(*(rig.project(o, p3) - pts[o, ok]).T) for o in range(C)])
            resid[c, ok] = np.nanmax(np.where(use[:, ok], e, np.nan), axis=0)
    return (err, resid) if with_residual else err


def _pair_errors(rig: _Rig, pts: np.ndarray) -> np.ndarray:
    """For points labeled in exactly two views: the larger reprojection error. (N,)"""
    labeled = ~np.isnan(pts).any(axis=2)
    ok = labeled.sum(0) == 2
    out = np.full(pts.shape[1], np.nan)
    if ok.any():
        p3 = rig.triangulate(rig.undistort(pts[:, ok]), labeled[:, ok])
        e = np.stack([np.hypot(*(rig.project(c, p3) - pts[c, ok]).T) for c in range(len(pts))])
        out[ok] = np.nanmax(np.where(labeled[:, ok], e, np.nan), axis=0)
    return out


def run_label_audit(
    project: Project,
    views: list[AuditLabelFileView],
    default_tol: float = DEFAULT_TOLERANCE_PX,
    tol_by_kp: dict[str, float] | None = None,
    rigid_groups: list[list[str]] | None = None,
    only_keypoints: list[str] | None = None,
    shape_strictness: float = SHAPE_SPREAD_FACTOR,
    overrides: dict[str, dict[str, dict[str, list[float] | None]]] | None = None,
    want: dict | None = None,
    learned: list | None = None,
    animals: list[str] | None = None,
) -> GetLabelAuditResponse:
    """Run the audit. ``overrides`` (view -> image path -> keypoint -> [x, y] or None) replaces
    labels in memory only. ``want`` ({"keypoint", "img": set of image paths}) gets "result":
    the group data of that frame for the rigid group of that keypoint."""
    tol_by_kp = tol_by_kp or {}
    data_dir = project.paths.data_dir.resolve()
    view_names = [v.viewName for v in views]
    project_views = list(project.config.view_names or view_names) if project.config else view_names

    labels: dict[str, pd.DataFrame] = {}
    for v in views:
        p = v.csvPath if v.csvPath.is_absolute() else data_dir / v.csvPath
        if not p.resolve().is_relative_to(data_dir):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{v.csvPath} is not inside the project's data directory.",
            )
        labels[v.viewName] = _read_labels(p)

    for v, rows in (overrides or {}).items():
        df = labels.get(v)
        if df is None:
            continue
        scorer = df.columns[0][0]
        for img, kps in rows.items():
            if img not in df.index:
                continue
            for kp, xy in kps.items():
                if (scorer, kp, "x") in df.columns:
                    df.loc[img, (scorer, kp, "x")] = np.nan if xy is None else float(xy[0])
                    df.loc[img, (scorer, kp, "y")] = np.nan if xy is None else float(xy[1])

    # Frames: (session, image name) -> {view: CSV index}
    frames: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for v, df in labels.items():
        for idx in df.index:
            fid = _frame_id(str(idx), project_views)
            if fid is not None:
                frames[fid][v] = str(idx)

    keypoints = list(dict.fromkeys(labels[view_names[0]].columns.get_level_values(1)))
    pairs = left_right_pairs(keypoints)
    items: list[LabelAuditItem] = []
    no_calib: list[str] = []
    points_checked = 0
    # For the shape check: per session, its rig, frames, labels and consistent 3D points.
    shape_data: list[tuple[_Rig, list[str], list[tuple[str, str]], np.ndarray, np.ndarray]] = []
    flagged: set[tuple[tuple[str, str], str]] = set()  # (frame, keypoint) with a finding

    by_session: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for fid in frames:
        by_session[fid[0]].append(fid)

    for session, fids in sorted(by_session.items()):
        calib = find_calibration_file(session, project)
        if calib is None:
            no_calib.append(session)
            continue
        cg_all = CameraGroup.load(str(calib))
        cam_names = [c.get_name() for c in cg_all.cameras]
        vs = [v for v in view_names if v in cam_names]
        if len(vs) < 2:
            no_calib.append(session)
            continue
        rig = _Rig(cg_all.subset_cameras_names(vs))
        fids = sorted(fids)

        # (C, F, K, 2)
        arr = np.full((len(vs), len(fids), len(keypoints), 2), np.nan)
        for c, v in enumerate(vs):
            df = labels[v]
            rows = [frames[f].get(v) for f in fids]
            have = [i for i, r in enumerate(rows) if r is not None]
            if not have:
                continue
            sub = df.loc[[rows[i] for i in have]]
            for k, kp in enumerate(keypoints):
                try:
                    xy = sub.xs(kp, axis=1, level=1).droplevel(0, axis=1)[["x", "y"]].to_numpy(float)
                except KeyError:
                    continue
                arr[c, have, k] = xy
        F, K = len(fids), len(keypoints)
        pts = arr.reshape(len(vs), F * K, 2)
        n_labeled = (~np.isnan(pts).any(axis=2)).sum(0)
        points_checked += int((n_labeled >= 2).sum())

        loo, resid = _loo_errors(rig, pts, with_residual=True)
        loo, resid = loo.reshape(len(vs), F, K), resid.reshape(len(vs), F, K)
        two = _pair_errors(rig, pts).reshape(F, K)

        def tol_of(kp: str) -> float:
            return float(tol_by_kp.get(kp, default_tol))

        if rigid_groups:
            # 3D point of every keypoint whose labeled views agree within tolerance.
            labeled = ~np.isnan(pts).any(axis=2)
            p3 = rig.triangulate(rig.undistort(pts), labeled)
            e = np.stack([np.hypot(*(rig.project(c, p3) - pts[c]).T) for c in range(len(vs))])
            worst = np.nanmax(np.where(labeled, e, -np.inf), axis=0).reshape(F, K)
            p3 = p3.reshape(F, K, 3)
            tol_k = np.array([tol_of(kp) for kp in keypoints])
            p3[~(worst <= tol_k[None])] = np.nan
            shape_data.append((rig, vs, fids, arr, p3))

        swapped: set[tuple[int, int, int]] = set()  # (frame, view, keypoint) explained by a swap
        # ---- swaps: only where one of the pair is off in some view; all candidates in one batch
        cands = []  # (a, b, ka, kb, c, f)
        for a, b in pairs:
            ka, kb = keypoints.index(a), keypoints.index(b)
            for c in range(len(vs)):
                off = (loo[c, :, ka] > tol_of(a)) | (loo[c, :, kb] > tol_of(b))
                for f in np.nonzero(off)[0]:
                    if not (np.isnan(arr[c, f, ka]).any() or np.isnan(arr[c, f, kb]).any()):
                        cands.append((a, b, ka, kb, c, int(f)))
        if cands:
            sw = np.full((len(vs), 2 * len(cands), 2), np.nan)
            for i, (a, b, ka, kb, c, f) in enumerate(cands):
                sw[:, 2 * i], sw[:, 2 * i + 1] = arr[:, f, ka], arr[:, f, kb]
                sw[c, 2 * i], sw[c, 2 * i + 1] = arr[c, f, kb], arr[c, f, ka]
            sw_err = _loo_errors(rig, sw)
            for i, (a, b, ka, kb, c, f) in enumerate(cands):
                ta, tb = tol_of(a), tol_of(b)
                ea, eb = sw_err[c, 2 * i], sw_err[c, 2 * i + 1]
                before = loo[c, f, ka] + loo[c, f, kb]
                if np.isnan([ea, eb, before]).any():
                    continue
                if ea <= ta and eb <= tb and ea + eb < SWAP_GAIN * before:
                    swapped.update({(f, c, ka), (f, c, kb)})
                    flagged.update({(fids[f], a), (fids[f], b)})
                    items.append(
                        LabelAuditItem(
                            kind="swap",
                            view=vs[c],
                            keypoint=a,
                            partner=b,
                            imgPaths=frames[fids[f]],
                            errorPx=float(max(loo[c, f, ka], loo[c, f, kb])),
                            tolerancePx=min(ta, tb),
                            score=float(max(loo[c, f, ka] / ta, loo[c, f, kb] / tb)),
                            reason=f"{a} and {b} look swapped in {vs[c]}",
                        )
                    )

        # ---- multiview: 3+ views -> blame the view whose removal leaves the others agreeing best
        for f in range(F):
            for k, kp in enumerate(keypoints):
                t = tol_of(kp)
                n = int((~np.isnan(arr[:, f, k]).any(axis=1)).sum())
                if n >= 3:
                    e, r = loo[:, f, k], resid[:, f, k]
                    if np.isnan(e).all() or not np.nanmax(e) > t:
                        continue
                    c = int(np.nanargmin(np.where(np.isnan(e), np.nan, r)))
                    if not e[c] > t or (f, c, k) in swapped:
                        continue
                    others = [vs[i] for i in range(len(vs)) if i != c and not np.isnan(arr[i, f, k]).any()]
                    reason = f"{kp} in {vs[c]} is {e[c]:.0f}px from where {', '.join(others)} put it"
                    err = float(e[c])
                elif n == 2:
                    err = float(two[f, k])
                    if not err > t:
                        continue
                    lab = [i for i in range(len(vs)) if not np.isnan(arr[i, f, k]).any()]
                    if any((f, i, k) in swapped for i in lab):
                        continue
                    c = lab[0]
                    reason = f"{kp}: {vs[lab[0]]} and {vs[lab[1]]} disagree by {err:.0f}px (can't tell which is off)"
                else:
                    continue
                flagged.add((fids[f], kp))
                items.append(
                    LabelAuditItem(
                        kind="multiview",
                        view=vs[c],
                        keypoint=kp,
                        imgPaths=frames[fids[f]],
                        errorPx=err,
                        tolerancePx=t,
                        score=err / t,
                        reason=reason,
                    )
                )

    shape_groups: list[list[str]] = []
    shape_problems: list[str] = []
    shape_pairs: list[ShapePair] = []
    shape_sizes: dict[str, float] = {}
    shape_animals: dict = dict(sessions={}, unmatched=[])
    shape_templates: list[ShapeTemplate] = []
    for group in rigid_groups or []:
        _shape_findings(
            group, keypoints, shape_data, frames, flagged,
            items, shape_groups, shape_problems, shape_strictness, shape_pairs, want, learned, shape_sizes,
            animals, shape_animals, shape_templates,
        )

    if only_keypoints is not None:
        only = set(only_keypoints)
        items = [it for it in items if it.keypoint in only or it.partner in only]

    # Frame order (session, image), then worst first within a frame.
    order = {fid: i for i, fid in enumerate(sorted(frames))}

    def key(it: LabelAuditItem):
        fid = _frame_id(next(iter(it.imgPaths.values())), project_views)
        return (order.get(fid, 0), -it.score)

    items.sort(key=key)
    return GetLabelAuditResponse(
        items=items,
        framesChecked=len(frames),
        pointsChecked=points_checked,
        sessionsWithoutCalibration=no_calib,
        shapeGroups=shape_groups,
        shapeProblems=shape_problems,
        shapePairs=shape_pairs,
        shapeSessionSize=shape_sizes,
        shapeAnimalSessions=shape_animals["sessions"],
        shapeUnmatchedSessions=shape_animals["unmatched"],
        shapeTemplates=shape_templates,
    )


def _animal_of(session: str, animals: list[str]) -> str | None:
    """The animal whose name occurs in the session name (the longest name wins), or None."""
    low = session.lower()
    hits = [a for a in animals if a.strip() and a.strip().lower() in low]
    return max(hits, key=len).strip() if hits else None


def _g(x: float) -> str:
    return f"{x:.3g}"


def _group_pairs(names, usual, spread, counts, p, factor, animal=None) -> tuple[list[ShapePair], np.ndarray]:
    """One frame of a rigid group (p: (G, 3)): all learned pairs, and the score matrix."""
    dist, score = pair_scores(usual, spread, p, factor)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if not np.isfinite(usual[i, j]):
                continue
            pairs.append(
                ShapePair(
                    a=names[i],
                    b=names[j],
                    usual=float(usual[i, j]),
                    spread=float(spread[i, j]),
                    limit=float(factor * spread[i, j]),
                    frames=int(counts[i, j]),
                    animal=animal,
                    distance=float(dist[i, j]) if np.isfinite(dist[i, j]) else None,
                    off=bool(score[i, j] > 1),
                )
            )
    return pairs, score


def _shape_findings(
    group, keypoints, shape_data, frames, flagged, items, shape_groups, shape_problems,
    factor=SHAPE_SPREAD_FACTOR,
    shape_pairs=None,
    want=None,
    learned=None,
    size_by_session=None,
    animals=None,
    animal_info=None,
    templates=None,
):
    """Add the labels of one rigid group that break its distances (one per frame: the worst)."""
    name = ", ".join(group)
    unknown = [kp for kp in group if kp not in keypoints]
    names = list(dict.fromkeys(kp for kp in group if kp in keypoints))
    if unknown:
        shape_problems.append(f"Not in the label file: {', '.join(unknown)}")
    if len(names) < MIN_GROUP_SIZE:
        shape_problems.append(f"A rigid group needs {MIN_GROUP_SIZE} or more keypoints: {name}")
        return
    if not shape_data:
        return
    cols = [keypoints.index(kp) for kp in names]
    # One entry of shape_data per session. All sessions together, each at its own size:
    sessions = [d[2][0][0] if d[2] else "" for d in shape_data]
    pts = [d[4][:, cols] for d in shape_data]
    usual, spread, counts, sizes = learn_scaled_distances(pts)
    # Per session: (usual, spread, counts, size, animal) to compare its frames with.
    tables = {s: (usual, spread, counts, size, None) for s, size in zip(sessions, sizes)}
    by_animal: dict[str | None, list[int]] = {}
    if animals:
        for i, session in enumerate(sessions):
            by_animal.setdefault(_animal_of(session, animals), []).append(i)
        for animal, idx in by_animal.items():
            own = np.concatenate([pts[i] for i in idx])
            u, sp, n = learn_distances(own)
            # Pairs this animal has too few frames for: those of all animals, at its size.
            size = session_sizes(usual, [own])[0]
            missing = ~np.isfinite(u)
            u[missing], sp[missing] = (usual * size)[missing], (spread * size)[missing]
            for i in idx:
                tables[sessions[i]] = (u, sp, n, 1.0, animal or "other sessions")
        if animal_info is not None:
            for animal, idx in by_animal.items():
                if animal is None:
                    animal_info["unmatched"] = sorted(sessions[i] for i in idx)
                else:
                    animal_info["sessions"][animal] = len(idx)
    usable = np.isfinite(usual).any(axis=1)
    if usable.sum() < MIN_GROUP_SIZE:
        shape_problems.append(f"Too few frames with these keypoints labeled together in 2+ views: {name}")
        return
    if not usable.all():
        left_out = [kp for kp, u in zip(names, usable) if not u]
        shape_problems.append(f"Too rarely labeled together with the rest of the group: {', '.join(left_out)}")
    members = [kp for kp, u in zip(names, usable) if u]
    shape_groups.append(members)
    if learned is not None:
        learned.append(dict(names=names, members=members, tables=tables))
    if shape_pairs is not None:
        nowhere = np.full((len(names), 3), np.nan)
        shown = set()
        for session in sessions:
            u, sp, n, _, animal = tables[session]
            if animal not in shown:
                shown.add(animal)
                shape_pairs.extend(_group_pairs(names, u, sp, n, nowhere, factor, animal)[0])
    if templates is not None:
        drawn = set()
        for session in sessions:
            u, _, _, _, animal = tables[session]
            if animal in drawn:
                continue
            drawn.add(animal)
            own = np.concatenate(
                [p / tables[s][3] for p, s in zip(pts, sessions) if tables[s][4] == animal]
            )
            xyz = template_points(u, own)
            placed = [i for i in range(len(names)) if np.isfinite(xyz[i]).all()]
            if len(placed) >= 3:
                templates.append(
                    ShapeTemplate(
                        animal=animal,
                        keypoints=[names[i] for i in placed],
                        points=[[float(v) for v in xyz[i]] for i in placed],
                    )
                )
    if size_by_session is not None and not animals:
        size_by_session.update({s: size for s, size in zip(sessions, sizes)})

    for (rig, vs, fids, arr, p3), session in zip(shape_data, sessions):
        usual, spread, counts, size, animal = tables[session]
        for f, fid in enumerate(fids):
            pairs, score = _group_pairs(names, usual, spread, counts, p3[f, cols] / size, factor, animal)
            if want is not None and want["keypoint"] in names and want["img"] & set(frames[fid].values()):
                want["result"] = dict(group=members, groupPairs=pairs)
            skip = {g for g in range(len(names)) if (fid, names[g]) in flagged}
            found = blame(score, skip)
            if found is None:
                continue
            g, partner = found
            row = np.nan_to_num(score[g], nan=0.0)
            o = int(np.argmax(row))
            worst = next(p for p in pairs if {p.a, p.b} == {names[g], names[o]})
            n_off, n_checked = int((row > 1).sum()), int(np.isfinite(score[g]).sum())
            way = "too long" if worst.distance > worst.usual else "too short"
            reason = f"{names[g]} to {names[o]} {way} ({100 * worst.distance / worst.usual:.0f}% of usual); "
            if partner is None:
                reason += f"{n_off} of {n_checked} distances of {names[g]} are off"
            else:
                reason += "one of the two is off"
            items.append(
                LabelAuditItem(
                    kind="shape",
                    view=vs[0],
                    keypoint=names[g],
                    partner=names[partner] if partner is not None else None,
                    imgPaths=frames[fid],
                    errorPx=abs(worst.distance - worst.usual),
                    tolerancePx=worst.limit,
                    score=float(row[o]),
                    reason=reason,
                    group=members,
                    groupPairs=pairs,
                    offPairs=n_off,
                    checkedPairs=n_checked,
                )
            )


@router.post("/app/v0/rpc/getLabelAudit")
async def get_label_audit(
    request: GetLabelAuditRequest,
    project_info_getter: ProjectInfoGetter = Depends(deps.project_info_getter),
) -> GetLabelAuditResponse:
    """Find labels that are probably wrong in a multiview label file."""
    project: Project = project_info_getter(request.projectKey)
    if len(request.views) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The label audit needs a multiview label file (two or more views).",
        )
    return await run_in_threadpool(
        run_label_audit,
        project,
        request.views,
        request.defaultTolerancePx,
        request.tolerancePx,
        request.rigidGroups,
        request.keypoints,
        request.shapeStrictness,
        animals=request.animals,
    )


# Learned distances of the last request settings, so an update after an edit only handles one frame.
_SHAPE_CACHE: dict[tuple, list[dict]] = {}


def _learned_shapes(project: Project, request: GetShapeGuidesRequest) -> list[dict]:
    """Pair distances of the rigid groups, learned from the saved labels (cached until they change)."""
    data_dir = project.paths.data_dir.resolve()
    files = []
    for v in request.views:
        p = v.csvPath if v.csvPath.is_absolute() else data_dir / v.csvPath
        files.append((str(p), p.stat().st_mtime_ns if p.exists() else 0))
    key = (
        request.projectKey,
        tuple(files),
        request.defaultTolerancePx,
        tuple(sorted(request.tolerancePx.items())),
        tuple(tuple(g) for g in request.rigidGroups),
        request.shapeStrictness,
        tuple(request.animals),
    )
    if key not in _SHAPE_CACHE:
        learned: list[dict] = []
        run_label_audit(
            project, request.views, request.defaultTolerancePx, request.tolerancePx,
            request.rigidGroups, None, request.shapeStrictness, None, None, learned, request.animals,
        )
        _SHAPE_CACHE.clear()
        _SHAPE_CACHE[key] = learned
    return _SHAPE_CACHE[key]


def shape_guides(project: Project, request: GetShapeGuidesRequest) -> GetShapeGuidesResponse:
    entry = next((e for e in _learned_shapes(project, request) if request.keypoint in e["names"]), None)
    if entry is None:
        return GetShapeGuidesResponse()
    view_names = [v.viewName for v in request.views]
    project_views = list(project.config.view_names or view_names) if project.config else view_names
    fid = next((f for f in (_frame_id(p, project_views) for p in request.imgPaths.values()) if f), None)
    calib = find_calibration_file(fid[0], project) if fid else None
    if calib is None:
        return GetShapeGuidesResponse()
    cg_all = CameraGroup.load(str(calib))
    cam_names = [c.get_name() for c in cg_all.cameras]
    vs = [v for v in view_names if v in cam_names]
    if len(vs) < 2:
        return GetShapeGuidesResponse()
    rig = _Rig(cg_all.subset_cameras_names(vs))
    names = entry["names"]
    arr = np.full((len(vs), len(names), 2), np.nan)
    for c, v in enumerate(vs):
        for k, kp in enumerate(names):
            xy = request.labels.get(v, {}).get(kp)
            if xy is not None and len(xy) == 2:
                arr[c, k] = xy
    # 3D point of every keypoint whose labeled views agree within tolerance (as in the audit).
    labeled = ~np.isnan(arr).any(axis=2)
    p3 = rig.triangulate(rig.undistort(arr), labeled)
    e = np.stack([np.hypot(*(rig.project(c, p3) - arr[c]).T) for c in range(len(vs))])
    reproj = np.nanmax(np.where(labeled, e, -np.inf), axis=0)
    tol_k = np.array([float(request.tolerancePx.get(kp, request.defaultTolerancePx)) for kp in names])
    p3[~(reproj <= tol_k)] = np.nan
    table = entry["tables"].get(fid[0])
    if table is None:
        return GetShapeGuidesResponse()
    usual, spread, counts, size, animal = table
    pairs, _ = _group_pairs(names, usual, spread, counts, p3 / size, request.shapeStrictness, animal)
    return GetShapeGuidesResponse(group=entry["members"], groupPairs=pairs)


@router.post("/app/v0/rpc/getShapeGuides")
async def get_shape_guides(
    request: GetShapeGuidesRequest,
    project_info_getter: ProjectInfoGetter = Depends(deps.project_info_getter),
) -> GetShapeGuidesResponse:
    """Measure one frame's rigid group with its labels as they are on screen (nothing is saved)."""
    project: Project = project_info_getter(request.projectKey)
    return await run_in_threadpool(shape_guides, project, request)


# ---------------------------------------------------------------- epipolar guides
def _camera_center_and_ray(cam, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    """World-frame center of ``cam`` and direction of the ray through pixel (x, y)."""
    ext = cam.get_extrinsics_mat()
    R, t = ext[:3, :3], ext[:3, 3]
    center = -R.T @ t
    xn = np.asarray(cam.undistort_points(np.array([[x, y]], dtype=np.float64))).reshape(-1)[:2]
    d = R.T @ np.array([xn[0], xn[1], 1.0])
    return center, d / np.linalg.norm(d)


def epipolar_segments(
    cg: CameraGroup, source: str, target: str, x: float, y: float, w: float, h: float, n: int = 400
) -> list[list[tuple[float, float]]]:
    """Polyline(s) in ``target`` of the ray through (x, y) in ``source``, clipped to the image."""
    names = [c.get_name() for c in cg.cameras]
    src, tgt = cg.cameras[names.index(source)], cg.cameras[names.index(target)]
    center, d = _camera_center_and_ray(src, x, y)
    centers = [-(c.get_extrinsics_mat()[:3, :3].T @ c.get_extrinsics_mat()[:3, 3]) for c in cg.cameras]
    scale = float(np.median([np.linalg.norm(center - c) for c in centers if np.linalg.norm(center - c) > 0]))
    ext = tgt.get_extrinsics_mat()
    m = 0.25 * max(w, h)  # a margin, so a line that leaves the image is still drawn to its edge

    def sample(depths):
        p3 = center[None] + depths[:, None] * d[None]
        z = (p3 @ ext[:3, :3].T + ext[:3, 3])[:, 2]
        proj = np.asarray(tgt.project(p3)).reshape(-1, 2)
        inside = (
            (z > 0) & np.isfinite(proj).all(1)
            & (proj[:, 0] > -m) & (proj[:, 0] < w + m) & (proj[:, 1] > -m) & (proj[:, 1] < h + m)
        )
        return proj, inside

    # Coarse pass over a wide depth range finds where the ray crosses the image,
    # then a dense pass over that stretch draws it (lens distortion can bend it).
    coarse = np.geomspace(scale * 1e-4, scale * 1e4, 4000)
    _, inside = sample(coarse)
    if not inside.any():
        return []
    i0, i1 = max(np.argmax(inside) - 1, 0), min(len(inside) - np.argmax(inside[::-1]), len(inside) - 1)
    proj, inside = sample(np.geomspace(coarse[i0], coarse[i1], n))
    segs, cur = [], []
    for ok, p in zip(inside, proj):
        if ok:
            cur.append((float(p[0]), float(p[1])))
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return [s for s in segs if len(s) >= 2]


def _direction_near(line: EpipolarLine, ref: tuple[float, float]) -> np.ndarray | None:
    """Unit direction of the line at its point nearest ``ref``."""
    best = None
    for seg in line.segments:
        q = np.array([[p.x, p.y] for p in seg])
        if len(q) < 2:
            continue
        i = int(np.argmin(np.hypot(*(q - ref).T)))
        a, b = q[max(i - 1, 0)], q[min(i + 1, len(q) - 1)]
        d = np.hypot(*(q[i] - ref))
        if best is None or d < best[0]:
            best = (d, b - a)
    if best is None or not np.linalg.norm(best[1]) > 0:
        return None
    return best[1] / np.linalg.norm(best[1])


def rank_lines(lines: list[EpipolarLine], ref: tuple[float, float]) -> None:
    """Order lines by how much each adds: widest-crossing pair first, then greedily the
    line whose smallest angle to the lines already chosen is largest. Sets ``rank`` and
    ``crossingAngleDeg`` in place; lines not in the image come last."""
    dirs = {i: _direction_near(l, ref) for i, l in enumerate(lines)}
    usable = [i for i, d in dirs.items() if d is not None]

    def angle(i: int, j: int) -> float:  # acute angle between two lines, degrees
        return float(np.degrees(np.arccos(min(1.0, abs(float(dirs[i] @ dirs[j]))))))

    order: list[int] = []
    if len(usable) == 1:
        order = usable[:]
    elif len(usable) >= 2:
        i, j = max(((i, j) for i in usable for j in usable if i < j), key=lambda p: angle(*p))
        order = [i, j]
        lines[i].crossingAngleDeg = lines[j].crossingAngleDeg = angle(i, j)
        rest = [k for k in usable if k not in order]
        while rest:
            k = max(rest, key=lambda k: min(angle(k, o) for o in order))
            lines[k].crossingAngleDeg = min(angle(k, o) for o in order)
            order.append(k)
            rest.remove(k)
    order += [i for i in range(len(lines)) if i not in order]
    for r, i in enumerate(order):
        lines[i].rank = r
    lines.sort(key=lambda l: l.rank)


def epipolar_guides(project: Project, req: GetEpipolarGuidesRequest) -> GetEpipolarGuidesResponse:
    calib = find_calibration_file(req.sessionKey, project)
    if calib is None:
        return GetEpipolarGuidesResponse(lines=[])
    cg = CameraGroup.load(str(calib))
    names = [c.get_name() for c in cg.cameras]
    if req.targetView not in names:
        return GetEpipolarGuidesResponse(lines=[])
    tgt = cg.cameras[names.index(req.targetView)]
    size = tgt.get_size()
    w = req.imageWidth or (float(size[0]) if size is not None else 1000.0)
    h = req.imageHeight or (float(size[1]) if size is not None else 1000.0)
    others = [l for l in req.labels if l.view != req.targetView and l.view in names and np.isfinite([l.x, l.y]).all()]

    lines = [
        EpipolarLine(
            fromView=l.view,
            segments=[[GuidePoint(x=a, y=b) for a, b in s] for s in epipolar_segments(cg, l.view, req.targetView, l.x, l.y, w, h)],
        )
        for l in others
    ]
    predicted = None
    if len(others) >= 2:
        # NumPy triangulation, not CameraGroup.triangulate: its numba code runs on the
        # "workqueue" threading layer (warm_up_anipose), which aborts the server when two
        # requests triangulate at once in FastAPI's thread pool.
        rig = _Rig(cg.subset_cameras_names([l.view for l in others]))
        pts = np.array([[[l.x, l.y]] for l in others], dtype=np.float64)  # (C, 1, 2)
        p3 = rig.triangulate(rig.undistort(pts), np.ones((len(others), 1), dtype=bool))
        p = np.asarray(tgt.project(p3)).reshape(-1, 2)[0]
        if np.isfinite(p).all():
            predicted = GuidePoint(x=float(p[0]), y=float(p[1]))
    ref = predicted or req.targetPoint or GuidePoint(x=w / 2, y=h / 2)
    rank_lines(lines, (ref.x, ref.y))
    return GetEpipolarGuidesResponse(
        lines=lines, predictedPoint=predicted, predictedFromViews=[l.view for l in others] if predicted else []
    )


@router.post("/app/v0/rpc/getEpipolarGuides")
def get_epipolar_guides(
    request: GetEpipolarGuidesRequest,
    project_info_getter: ProjectInfoGetter = Depends(deps.project_info_getter),
) -> GetEpipolarGuidesResponse:
    """Epipolar lines and the other views' point for one keypoint in one view."""
    project: Project = project_info_getter(request.projectKey)
    return epipolar_guides(project, request)
