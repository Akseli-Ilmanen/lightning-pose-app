"""Label audit: find labels that are probably wrong, and epipolar guides for fixing them (#207).

``getLabelAudit`` checks every labeled frame of a multiview label file against the
camera calibration. No model is needed:

- **multiview**: the keypoint is labeled in two or more views, but the views do not
  describe the same 3D point. With three or more views, each view's label is
  compared with the point triangulated from the *other* views, so the view that
  is off is the one flagged. With two views the check cannot tell which one is off.
- **swap**: a left/right pair (``L1A``/``R1A``, ``leftEye``/``rightEye``, ``leye``/``reye``)
  agrees with the other views much better after swapping the two labels in one view.

``getEpipolarGuides`` returns, for one keypoint and one target view, the epipolar
line of each other view's label (as a polyline, so lens distortion bends it), and
the point triangulated from the other views projected into the target view.
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
from litpose_app.session_view import split_session_view
from litpose_app.utils.fix_empty_first_row import fix_empty_first_row

router = APIRouter()
logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE_PX = 5.0
# A swap is reported when swapping brings both labels within tolerance and at
# least halves their combined error.
SWAP_GAIN = 0.5


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


class LabelAuditItem(BaseModel):
    """One label that is probably wrong."""

    kind: str  # "multiview" | "swap"
    view: str  # the view whose label is probably wrong
    keypoint: str
    partner: str | None = None  # the other keypoint of a swap
    # Image path of this frame in every view (CSV index), so the client can find the frame.
    imgPaths: dict[str, str]
    errorPx: float  # multiview: distance to the other views' point; swap: error before swapping
    tolerancePx: float
    score: float  # errorPx / tolerancePx; sorts items within a frame
    reason: str


class GetLabelAuditResponse(BaseModel):
    items: list[LabelAuditItem]
    framesChecked: int
    pointsChecked: int  # keypoint x frame with labels in 2+ views
    sessionsWithoutCalibration: list[str]


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
) -> GetLabelAuditResponse:
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
        run_label_audit, project, request.views, request.defaultTolerancePx, request.tolerancePx
    )


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
