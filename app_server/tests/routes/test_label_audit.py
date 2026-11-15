"""getLabelAudit / getEpipolarGuides: wrong labels are found from the calibration (#207)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from aniposelib.cameras import Camera, CameraGroup
from fastapi.testclient import TestClient

from litpose_app.routes.labeler.label_audit import (
    EpipolarLine,
    GuidePoint,
    left_right_pairs,
    rank_lines,
)

VIEWS = ["camA", "camB", "camC"]
KPS = ["nose", "leftEar", "rightEar"]
SESSION = "sess1"


def _rig() -> CameraGroup:
    """Three cameras around the origin, 2 m away, looking at it."""
    cams = []
    for i, (name, angle) in enumerate(zip(VIEWS, [-0.6, 0.0, 0.6])):
        R = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
        rvec = np.array([0.0, -angle, 0.0])
        cams.append(
            Camera(
                name=name,
                size=(640, 480),
                matrix=[[800, 0, 320], [0, 800, 240], [0, 0, 1]],
                dist=[-0.1, 0.01, 0, 0, 0],
                rvec=rvec,
                tvec=np.array([0.0, 0.0, 2.0]),
            )
        )
    return CameraGroup(cams)


def _write_project(data_dir: Path, cg: CameraGroup, n_frames: int = 6) -> dict[str, np.ndarray]:
    """Project random 3D keypoints into every view and write LP label CSVs."""
    (data_dir / "calibrations").mkdir(parents=True, exist_ok=True)
    cg.dump(str(data_dir / "calibrations" / f"{SESSION}.toml"))
    rng = np.random.default_rng(0)
    p3 = rng.uniform(-0.15, 0.15, size=(n_frames, len(KPS), 3))
    labels = {}
    for c, v in enumerate(VIEWS):
        xy = np.asarray(cg.cameras[c].project(p3.reshape(-1, 3))).reshape(n_frames, len(KPS), 2)
        labels[v] = xy
    return labels


def _save(data_dir: Path, labels: dict[str, np.ndarray]) -> None:
    for v, xy in labels.items():
        cols = pd.MultiIndex.from_product([["me"], KPS, ["x", "y"]])
        idx = [f"labeled-data/{SESSION}_{v}/img{f:04d}.png" for f in range(len(xy))]
        pd.DataFrame(xy.reshape(len(xy), -1), index=idx, columns=cols).to_csv(data_dir / f"CollectedData_{v}.csv")


def _audit(client: TestClient, data_dir: Path, **kw):
    return client.post(
        "/app/v0/rpc/getLabelAudit",
        json={
            "projectKey": "p",
            "views": [{"viewName": v, "csvPath": str(data_dir / f"CollectedData_{v}.csv")} for v in VIEWS],
            **kw,
        },
    )


@pytest.fixture
def project(register_project):
    data_dir = register_project("p", views=VIEWS, keypoints=KPS)
    cg = _rig()
    return data_dir, cg, _write_project(data_dir, cg)


def test_clean_labels_have_no_findings(client: TestClient, project):
    data_dir, _, labels = project
    _save(data_dir, labels)

    r = _audit(client, data_dir)

    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["framesChecked"] == 6
    assert body["pointsChecked"] == 6 * len(KPS)


def test_a_shifted_label_is_found_in_the_view_that_is_off(client: TestClient, project):
    data_dir, _, labels = project
    labels["camB"][2, 0] += [15, -10]  # nose, frame 2, camB
    _save(data_dir, labels)

    items = _audit(client, data_dir).json()["items"]

    assert len(items) == 1
    it = items[0]
    assert (it["kind"], it["view"], it["keypoint"]) == ("multiview", "camB", "nose")
    assert it["imgPaths"]["camA"] == f"labeled-data/{SESSION}_camA/img0002.png"
    assert it["errorPx"] > 10


def test_a_left_right_swap_in_one_view_is_reported_as_a_swap(client: TestClient, project):
    data_dir, _, labels = project
    labels["camC"][4, [1, 2]] = labels["camC"][4, [2, 1]]
    _save(data_dir, labels)

    items = _audit(client, data_dir).json()["items"]

    assert [(i["kind"], i["view"], i["keypoint"], i["partner"]) for i in items] == [
        ("swap", "camC", "leftEar", "rightEar")
    ]


def test_tolerance_per_keypoint(client: TestClient, project):
    data_dir, _, labels = project
    labels["camA"][1, 0] += [4, 0]  # a 4px nudge
    _save(data_dir, labels)

    assert _audit(client, data_dir, defaultTolerancePx=10).json()["items"] == []
    flagged = _audit(client, data_dir, defaultTolerancePx=10, tolerancePx={"nose": 1}).json()["items"]
    assert [(i["view"], i["keypoint"]) for i in flagged] == [("camA", "nose")]


def test_two_views_only_flags_the_disagreement(client: TestClient, project):
    data_dir, _, labels = project
    labels["camC"][3, 0] = np.nan  # nose only in camA and camB
    labels["camA"][3, 0] += [0, 12]
    _save(data_dir, labels)

    items = _audit(client, data_dir).json()["items"]

    assert len(items) == 1
    assert items[0]["keypoint"] == "nose" and "can't tell" in items[0]["reason"]


def test_sessions_without_calibration_are_listed(client: TestClient, project):
    data_dir, _, labels = project
    _save(data_dir, labels)
    (data_dir / "calibrations" / f"{SESSION}.toml").unlink()

    body = _audit(client, data_dir).json()

    assert body["sessionsWithoutCalibration"] == [SESSION]
    assert body["items"] == []


def test_refuses_a_single_view(client: TestClient, project):
    data_dir, _, labels = project
    _save(data_dir, labels)
    r = client.post(
        "/app/v0/rpc/getLabelAudit",
        json={"projectKey": "p", "views": [{"viewName": "camA", "csvPath": str(data_dir / "CollectedData_camA.csv")}]},
    )
    assert r.status_code == 400


def test_epipolar_lines_pass_through_the_true_point(client: TestClient, project):
    data_dir, cg, labels = project
    _save(data_dir, labels)
    f, k = 0, 0
    r = client.post(
        "/app/v0/rpc/getEpipolarGuides",
        json={
            "projectKey": "p",
            "sessionKey": SESSION,
            "targetView": "camB",
            "labels": [
                {"view": v, "x": float(labels[v][f, k, 0]), "y": float(labels[v][f, k, 1])} for v in VIEWS
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    truth = labels["camB"][f, k]
    assert sorted(l["fromView"] for l in body["lines"]) == ["camA", "camC"]
    for line in body["lines"]:
        assert min(_dist_to_polyline(truth, s) for s in line["segments"]) < 0.5
    pred = body["predictedPoint"]
    assert np.hypot(pred["x"] - truth[0], pred["y"] - truth[1]) < 0.5
    assert sorted(body["predictedFromViews"]) == ["camA", "camC"]


def _dist_to_polyline(p: np.ndarray, seg: list[dict]) -> float:
    q = np.array([[s["x"], s["y"]] for s in seg])
    a, b = q[:-1], q[1:]
    t = np.clip(np.sum((p - a) * (b - a), 1) / np.maximum(np.sum((b - a) ** 2, 1), 1e-12), 0, 1)
    return float(np.min(np.hypot(*(a + t[:, None] * (b - a) - p).T)))


def _line(view: str, angle_deg: float) -> EpipolarLine:
    """A straight line through (100, 100) at the given angle."""
    a = np.radians(angle_deg)
    pts = [GuidePoint(x=100 + t * np.cos(a), y=100 + t * np.sin(a)) for t in np.linspace(-50, 50, 11)]
    return EpipolarLine(fromView=view, segments=[pts])


def test_lines_are_ranked_by_how_steeply_they_cross():
    lines = [_line("nearly_a", 5), _line("a", 0), _line("diag", 45), _line("b", 90), EpipolarLine(fromView="off", segments=[])]

    rank_lines(lines, (100, 100))

    # widest pair first (a, b: 90 deg), then the line that best splits them, the
    # near-duplicate of a after that, and the line outside the image last.
    assert [l.fromView for l in lines] == ["a", "b", "diag", "nearly_a", "off"]
    assert [l.rank for l in lines] == [0, 1, 2, 3, 4]
    assert lines[2].crossingAngleDeg == pytest.approx(45)
    assert lines[3].crossingAngleDeg == pytest.approx(5)
    assert lines[4].crossingAngleDeg is None


def test_left_right_pairs():
    assert left_right_pairs(["L1A", "R1A", "L2A"]) == [("L1A", "R1A")]
    assert left_right_pairs(["leye", "reye", "leftShoulder", "rightShoulder", "nose"]) == [
        ("leftShoulder", "rightShoulder"),
        ("leye", "reye"),
    ]
    assert left_right_pairs(["paw_l", "paw_r", "tail"]) == [("paw_l", "paw_r")]
    assert left_right_pairs(["nose", "tail"]) == []
