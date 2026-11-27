"""Label audit, midline check: a back off the midplane of the shoulders is found (#207)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from aniposelib.cameras import Camera, CameraGroup
from fastapi.testclient import TestClient

VIEWS = ["camA", "camB"]
KPS = ["leftShoulder", "rightShoulder", "back", "nose"]
SESSION = "sess1"
MIDLINE = {"left": "leftShoulder", "right": "rightShoulder", "point": "back", "toleranceDeg": 5}
N_FRAMES = 5
OFF_FRAME = 3


def _rig() -> CameraGroup:
    """Two cameras 2 m from the origin, looking at it from either side."""
    cams = [
        Camera(
            name=name,
            size=(640, 480),
            matrix=[[800, 0, 320], [0, 800, 240], [0, 0, 1]],
            dist=[-0.1, 0.01, 0, 0, 0],
            rvec=np.array([0.0, -angle, 0.0]),
            tvec=np.array([0.0, 0.0, 2.0]),
        )
        for name, angle in zip(VIEWS, [-0.5, 0.5])
    ]
    return CameraGroup(cams)


def _trunks() -> np.ndarray:
    """(frames, keypoints, 3): shoulders 0.2 apart, back 0.2 behind their midpoint, in the midplane."""
    trunk = np.array([[-0.1, 0, 0], [0.1, 0, 0], [0, 0.02, 0.2], [0, 0, -0.1]], float)
    return np.stack([trunk + [0.01 * f, 0, 0] for f in range(N_FRAMES)])


def _save(data_dir: Path, cg: CameraGroup, p3: np.ndarray, drop: list[tuple[str, int, str]] = ()) -> dict:
    labels = {}
    for c, v in enumerate(VIEWS):
        xy = np.asarray(cg.cameras[c].project(p3.reshape(-1, 3))).reshape(len(p3), len(KPS), 2)
        for view, f, kp in drop:
            if view == v:
                xy[f, KPS.index(kp)] = np.nan
        cols = pd.MultiIndex.from_product([["me"], KPS, ["x", "y"]])
        idx = [f"labeled-data/{SESSION}_{v}/img{f:04d}.png" for f in range(len(xy))]
        pd.DataFrame(xy.reshape(len(xy), -1), index=idx, columns=cols).to_csv(data_dir / f"CollectedData_{v}.csv")
        labels[v] = xy
    return labels


def _audit(client: TestClient, data_dir: Path, **kw) -> dict:
    r = client.post(
        "/app/v0/rpc/getLabelAudit",
        json={
            "projectKey": "p",
            "views": [{"viewName": v, "csvPath": str(data_dir / f"CollectedData_{v}.csv")} for v in VIEWS],
            **kw,
        },
    )
    assert r.status_code == 200
    return r.json()


def _guides(client: TestClient, labels: dict, frame: int) -> dict:
    r = client.post(
        "/app/v0/rpc/getMidlineGuides",
        json={
            "projectKey": "p",
            "sessionKey": SESSION,
            **{k: MIDLINE[k] for k in ("left", "right", "point")},
            "labels": {
                v: {kp: (None if np.isnan(xy[frame, k]).any() else xy[frame, k].tolist()) for k, kp in enumerate(KPS)}
                for v, xy in labels.items()
            },
        },
    )
    assert r.status_code == 200
    return r.json()


@pytest.fixture
def project(register_project):
    data_dir = register_project("p", views=VIEWS, keypoints=KPS)
    cg = _rig()
    (data_dir / "calibrations").mkdir(parents=True, exist_ok=True)
    cg.dump(str(data_dir / "calibrations" / f"{SESSION}.toml"))
    return data_dir, cg


def test_a_back_in_the_midplane_has_no_midline_finding(client: TestClient, project):
    data_dir, cg = project
    _save(data_dir, cg, _trunks())

    body = _audit(client, data_dir, midline=MIDLINE)

    assert body["items"] == []
    assert body["midlineChecked"] == N_FRAMES


def test_a_back_off_the_midplane_is_found_though_the_views_agree(client: TestClient, project):
    data_dir, cg = project
    p3 = _trunks()
    p3[OFF_FRAME, KPS.index("back")] += [0.05, 0, 0]  # towards the right shoulder: atan(0.05 / 0.2) = 14 deg
    _save(data_dir, cg, p3)

    items = _audit(client, data_dir, midline=MIDLINE)["items"]

    assert len(items) == 1
    it = items[0]
    assert (it["kind"], it["keypoint"]) == ("midline", "back")
    assert it["imgPaths"]["camA"] == f"labeled-data/{SESSION}_camA/img{OFF_FRAME:04d}.png"
    assert it["midline"]["yawDeg"] == pytest.approx(np.degrees(np.arctan2(0.05, np.hypot(0.2, 0.02))), abs=0.2)
    assert it["midline"]["width"] == pytest.approx(0.2, abs=1e-3)
    assert "rightShoulder" in it["reason"]


def test_without_the_setting_the_midline_is_not_checked(client: TestClient, project):
    data_dir, cg = project
    p3 = _trunks()
    p3[OFF_FRAME, KPS.index("back")] += [0.05, 0, 0]
    _save(data_dir, cg, p3)

    assert _audit(client, data_dir)["items"] == []


def test_frames_without_a_shoulder_in_two_views_are_counted_as_no_guide(client: TestClient, project):
    data_dir, cg = project
    _save(data_dir, cg, _trunks(), drop=[("camB", 1, "leftShoulder")])

    body = _audit(client, data_dir, midline=MIDLINE)

    assert (body["midlineChecked"], body["midlineNoGuide"]) == (N_FRAMES - 1, 1)


def test_an_unknown_midline_keypoint_is_reported(client: TestClient, project):
    data_dir, cg = project
    _save(data_dir, cg, _trunks())

    body = _audit(client, data_dir, midline={**MIDLINE, "point": "tail"})

    assert body["items"] == [] and "tail" in body["midlineProblems"][0]


def test_the_suggestion_puts_the_back_into_the_midplane(client: TestClient, project):
    data_dir, cg = project
    p3 = _trunks()
    p3[OFF_FRAME, KPS.index("back")] += [0.05, 0, 0]
    labels = _save(data_dir, cg, p3)

    before = _guides(client, labels, OFF_FRAME)
    assert before["info"]["yawDeg"] == pytest.approx(14, abs=0.5)
    assert all(len(v["line"]) > 2 for v in before["views"])
    # Move camA's back onto its suggestion (camB's ray cut with the midplane): now on the midline.
    s = next(v["suggestion"] for v in before["views"] if v["view"] == "camA")
    labels["camA"][OFF_FRAME, KPS.index("back")] = [s["x"], s["y"]]

    after = _guides(client, labels, OFF_FRAME)

    assert after["info"]["yawDeg"] == pytest.approx(0, abs=0.05)


def test_no_midline_guides_without_both_shoulders(client: TestClient, project):
    data_dir, cg = project
    labels = _save(data_dir, cg, _trunks(), drop=[("camB", 1, "leftShoulder")])

    assert _guides(client, labels, 1) == {"views": [], "info": None}


def _queue(data_dir: Path, cg: CameraGroup, p3: np.ndarray) -> None:
    """Queue frames img9000.. with the projections of p3 as their predictions."""
    import json

    for c, v in enumerate(VIEWS):
        xy = np.asarray(cg.cameras[c].project(p3.reshape(-1, 3))).reshape(len(p3), len(KPS), 2)
        lines = [
            {
                "frame_path": f"labeled-data/{SESSION}_{v}/img{9000 + f}.png",
                "predictions": {
                    "model_name": "m",
                    "date_time": 0,
                    "predictions": [
                        {"keypoint_name": kp, "x": float(x), "y": float(y)} for kp, (x, y) in zip(KPS, xy[f])
                    ],
                },
            }
            for f in range(len(xy))
        ]
        (data_dir / f"CollectedData_{v}.unlabeled.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))


def test_the_queue_scope_checks_the_queued_predictions_only(client: TestClient, project):
    data_dir, cg = project
    saved = _trunks()
    saved[OFF_FRAME, KPS.index("back")] += [0.05, 0, 0]  # a saved label that is off: not reported for the queue
    _save(data_dir, cg, saved)
    queued = _trunks()[:2]
    queued[1, KPS.index("back")] += [0.05, 0, 0]
    _queue(data_dir, cg, queued)

    body = _audit(client, data_dir, midline=MIDLINE, scope="unlabeled")

    assert [it["imgPaths"]["camA"] for it in body["items"]] == [f"labeled-data/{SESSION}_camA/img9001.png"]
    assert (body["framesChecked"], body["midlineChecked"]) == (2, 2)
    assert body["cleanFrames"] == [{v: f"labeled-data/{SESSION}_{v}/img9000.png" for v in VIEWS}]
    labeled = _audit(client, data_dir, midline=MIDLINE)
    assert len(labeled["items"]) == 1 and labeled["framesChecked"] == N_FRAMES and labeled["cleanFrames"] == []
