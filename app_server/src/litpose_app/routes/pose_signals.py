"""RPC endpoints for time series derived from a prediction file (speed, head direction, ...).

The Viewer plots these under the video so a user can spot tracking errors (spikes,
jitter) and pick frames to extract from there. Position and confidence come straight
from the prediction CSV the Viewer already loads; the signals here are computed with
`movement <https://movement.neuroinformatics.dev/>`_, an optional dependency
(``pip install lightning-pose-app[signals]``).

The choice of signal is limited to the entries of ``SIGNALS`` below: each is a call
into movement's public API with named, typed parameters. No user code runs.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import deps
from ..deps import ProjectInfoGetter

logger = logging.getLogger(__name__)
router = APIRouter()

MOVEMENT_DOCS = "https://movement.neuroinformatics.dev/latest/api/"


class SignalParam(BaseModel):
    """One argument the UI asks for before computing a signal."""

    name: str
    label: str
    # keypoints: several keypoint names. keypoint: one keypoint name.
    kind: Literal["keypoints", "keypoint", "number", "choice"]
    default: Any = None
    choices: list[str] | None = None
    help: str = ""


class SignalInfo(BaseModel):
    """A derived signal the Viewer can plot."""

    id: str
    label: str
    function: str
    """The movement function behind it, e.g. ``movement.kinematics.compute_speed``."""
    docUrl: str
    unit: str
    params: list[SignalParam]


class ListPoseSignalsResponse(BaseModel):
    movementAvailable: bool
    movementVersion: str | None = None
    signals: list[SignalInfo]


class ComputePoseSignalRequest(BaseModel):
    projectKey: str
    predictionPath: str
    """Prediction CSV path relative to the project's model directory (as the UI lists it)."""
    signal: str
    params: dict[str, Any] = {}
    fps: float | None = None
    """Frames per second. When set, rates are per second; otherwise per frame."""
    confidenceThreshold: float | None = None
    """Drop positions below this likelihood before computing (movement.filtering.filter_by_confidence)."""


class SignalSeries(BaseModel):
    label: str
    keypoint: str | None = None
    """The keypoint this series belongs to, so the UI can filter by keypoint."""
    space: str | None = None
    """"x" or "y" for per-axis signals (velocity), else None."""
    values: list[float | None]
    """One value per frame. None where the signal is undefined (NaN)."""


class ComputePoseSignalResponse(BaseModel):
    signal: str
    unit: str
    nFrames: int
    series: list[SignalSeries]


# ---------------------------------------------------------------------------
# Signal registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Signal:
    info: SignalInfo
    compute: Callable[[Any, dict[str, Any]], list[SignalSeries]]


def _keypoints(position) -> list[str]:
    return [str(k) for k in position.coords["keypoint"].values]


def _keypoint(position, params: dict[str, Any], name: str) -> str:
    value = params.get(name)
    names = [str(k) for k in position.coords["keypoint"].values]
    if value not in names:
        raise HTTPException(status_code=422, detail=f"Choose a keypoint for '{name}'.")
    return str(value)


def _series(label, values, keypoint=None, space=None) -> SignalSeries:
    return SignalSeries(label=label, values=_json_floats(values), keypoint=keypoint, space=space)


def _per_keypoint(da, keypoints: list[str]) -> list[SignalSeries]:
    da = da.isel(individual=0) if "individual" in da.dims else da
    return [_series(kp, da.sel(keypoint=kp).values, keypoint=kp) for kp in keypoints]


def _speed(position, params):
    from movement.kinematics import compute_speed

    return _per_keypoint(compute_speed(position), _keypoints(position))


def _velocity(position, params):
    from movement.kinematics import compute_velocity

    velocity = compute_velocity(position)
    velocity = velocity.isel(individual=0) if "individual" in velocity.dims else velocity
    out = []
    for kp in _keypoints(position):
        for axis in ("x", "y"):
            out.append(
                _series(f"{kp} {axis}", velocity.sel(keypoint=kp, space=axis).values, kp, axis)
            )
    return out


def _acceleration(position, params):
    from movement.kinematics import compute_acceleration
    from movement.utils.vector import compute_norm

    magnitude = compute_norm(compute_acceleration(position))
    return _per_keypoint(magnitude, _keypoints(position))


def _head_direction(position, params):
    from movement.kinematics import compute_forward_vector_angle

    left = _keypoint(position, params, "left_keypoint")
    right = _keypoint(position, params, "right_keypoint")
    if left == right:
        raise HTTPException(status_code=422, detail="Left and right keypoints must differ.")
    camera_view = params.get("camera_view") or "top_down"
    angle = compute_forward_vector_angle(
        position, left, right, camera_view=camera_view, in_degrees=True
    )
    angle = angle.isel(individual=0) if "individual" in angle.dims else angle
    return [_series(f"heading ({left}→{right})", angle.values)]


def _distance(position, params):
    from movement.kinematics import compute_pairwise_distances

    a = _keypoint(position, params, "keypoint_a")
    b = _keypoint(position, params, "keypoint_b")
    if a == b:
        raise HTTPException(status_code=422, detail="Choose two different keypoints.")
    distance = compute_pairwise_distances(position, dim="keypoint", pairs={a: b})
    distance = distance.isel(individual=0) if "individual" in distance.dims else distance
    return [_series(f"{a}–{b}", np.asarray(distance.values).reshape(-1))]


def _info(id, label, function, unit, params) -> SignalInfo:
    module, name = function.rsplit(".", 1)
    return SignalInfo(
        id=id,
        label=label,
        function=function,
        docUrl=f"{MOVEMENT_DOCS}{module}.{name}.html",
        unit=unit,
        params=params,
    )


SIGNALS: dict[str, _Signal] = {
    s.info.id: s
    for s in [
        _Signal(
            _info("speed", "Speed", "movement.kinematics.compute_speed", "px/s", []),
            _speed,
        ),
        _Signal(
            _info(
                "velocity", "Velocity (x, y)", "movement.kinematics.compute_velocity", "px/s",
                [],
            ),
            _velocity,
        ),
        _Signal(
            _info(
                "acceleration", "Acceleration (magnitude)",
                "movement.kinematics.compute_acceleration", "px/s²", [],
            ),
            _acceleration,
        ),
        _Signal(
            _info(
                "head_direction",
                "Head direction",
                "movement.kinematics.compute_forward_vector_angle",
                "deg",
                [
                    SignalParam(
                        name="left_keypoint", label="Left keypoint", kind="keypoint",
                        help="e.g. left ear or left eye.",
                    ),
                    SignalParam(
                        name="right_keypoint", label="Right keypoint", kind="keypoint",
                        help="e.g. right ear or right eye.",
                    ),
                    SignalParam(
                        name="camera_view", label="Camera view", kind="choice",
                        default="top_down", choices=["top_down", "bottom_up"],
                    ),
                ],
            ),
            _head_direction,
        ),
        _Signal(
            _info(
                "distance",
                "Distance between keypoints",
                "movement.kinematics.compute_pairwise_distances",
                "px",
                [
                    SignalParam(name="keypoint_a", label="Keypoint A", kind="keypoint"),
                    SignalParam(name="keypoint_b", label="Keypoint B", kind="keypoint"),
                ],
            ),
            _distance,
        ),
    ]
}


def movement_version() -> str | None:
    """The installed movement version, or None when movement is not installed."""
    try:
        import movement
    except ImportError:
        return None
    return getattr(movement, "__version__", "unknown")


@router.post("/app/v0/rpc/listPoseSignals")
def list_pose_signals() -> ListPoseSignalsResponse:
    """Signals the Viewer can compute, and whether movement is installed to compute them."""
    version = movement_version()
    return ListPoseSignalsResponse(
        movementAvailable=version is not None,
        movementVersion=version,
        signals=[s.info for s in SIGNALS.values()],
    )


# ---------------------------------------------------------------------------
# Loading predictions
# ---------------------------------------------------------------------------


def read_lp_predictions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Keypoint names, positions (frames, 2, keypoints) and likelihoods (frames, keypoints).

    Reads x, y and likelihood only. EKS outputs carry extra columns (ensemble
    variances) that a plain LP/DLC reader would trip over.
    """
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    coords = df.columns.get_level_values(2)
    bodyparts = list(dict.fromkeys(df.columns.get_level_values(1)[coords == "x"]))
    if not bodyparts:
        raise HTTPException(status_code=422, detail=f"No x/y columns in {path.name}.")
    position = np.full((len(df), 2, len(bodyparts)), np.nan)
    confidence = np.full((len(df), len(bodyparts)), np.nan)
    for j, kp in enumerate(bodyparts):
        sub = df.xs(kp, axis=1, level=1)
        sub.columns = sub.columns.get_level_values(-1)
        position[:, 0, j] = pd.to_numeric(sub["x"], errors="coerce").to_numpy()
        position[:, 1, j] = pd.to_numeric(sub["y"], errors="coerce").to_numpy()
        if "likelihood" in sub.columns:
            confidence[:, j] = pd.to_numeric(sub["likelihood"], errors="coerce").to_numpy()
    return bodyparts, position, confidence


def _resolve_prediction_path(model_dir: Path, relative: str) -> Path:
    path = (model_dir / relative).resolve()
    try:
        path.relative_to(model_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Prediction file is outside the model directory.")
    if path.suffix.lower() != ".csv" or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Prediction file not found: {relative}")
    return path


def compute_pose_signal(path: Path, request: ComputePoseSignalRequest) -> ComputePoseSignalResponse:
    """Load predictions into a movement dataset and run the requested signal on it."""
    signal = SIGNALS.get(request.signal)
    if signal is None:
        raise HTTPException(status_code=422, detail=f"Unknown signal: {request.signal}")
    if movement_version() is None:
        raise HTTPException(
            status_code=501,
            detail="This signal needs movement: pip install 'lightning-pose-app[signals]'",
        )
    from movement.io.load_poses import from_numpy

    keypoints, position, confidence = read_lp_predictions(path)
    fps = request.fps if request.fps and request.fps > 0 else None
    ds = from_numpy(
        position_array=position[..., None],
        confidence_array=confidence[..., None],
        individual_names=["individual_0"],
        keypoint_names=keypoints,
        fps=fps,
        source_software="LightningPose",
    )
    data = ds.position
    if request.confidenceThreshold is not None:
        from movement.filtering import filter_by_confidence

        data = filter_by_confidence(data, ds.confidence, threshold=request.confidenceThreshold)

    series = signal.compute(data, request.params)
    unit = signal.info.unit if fps else signal.info.unit.replace("/s", "/frame")
    return ComputePoseSignalResponse(
        signal=request.signal,
        unit=unit,
        nFrames=position.shape[0],
        series=series,
    )


def _json_floats(values: np.ndarray) -> list[float | None]:
    return [None if not math.isfinite(x) else float(x) for x in np.asarray(values, float).ravel()]


@router.post("/app/v0/rpc/computePoseSignal")
async def compute_pose_signal_route(
    request: ComputePoseSignalRequest,
    project_info_getter: ProjectInfoGetter = Depends(deps.project_info_getter),
) -> ComputePoseSignalResponse:
    """Compute one derived signal from a prediction file, one value per frame."""
    project = project_info_getter(request.projectKey)
    path = _resolve_prediction_path(project.paths.model_dir, request.predictionPath)
    return await run_in_threadpool(compute_pose_signal, path, request)
