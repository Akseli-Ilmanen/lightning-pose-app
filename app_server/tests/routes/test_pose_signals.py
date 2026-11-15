"""Derived prediction signals (movement) and k-means extraction within a segment."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

N_FRAMES = 40


def _write_predictions(path: Path, keypoints=("nose", "ear_l", "ear_r")) -> None:
    """LP prediction CSV: nose moves 1 px/frame in x, then jumps 50 px at frame 20."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        ",".join(["scorer"] + ["heatmap_tracker"] * 3 * len(keypoints)),
        ",".join(["bodyparts"] + [kp for kp in keypoints for _ in range(3)]),
        ",".join(["coords"] + ["x", "y", "likelihood"] * len(keypoints)),
    ]
    rows = []
    for t in range(N_FRAMES):
        values = []
        for j, _ in enumerate(keypoints):
            x = float(t) + (50.0 if (j == 0 and t == 20) else 0.0) + 10 * j
            likelihood = 0.1 if t == 5 else 0.9
            values += [x, 0.0 if j != 2 else 10.0, likelihood]
        rows.append(",".join([str(t)] + [str(v) for v in values]))
    path.write_text("\n".join(header + rows) + "\n")


@pytest.fixture
def project(register_project):
    data_dir = register_project("demo", views=["cam"], keypoints=["nose", "ear_l", "ear_r"])
    _write_predictions(data_dir / "models" / "m1" / "video_preds" / "sess_cam.csv")
    return data_dir


def _compute(client, **body):
    payload = {
        "projectKey": "demo",
        "predictionPath": "m1/video_preds/sess_cam.csv",
        "fps": 10,
    }
    payload.update(body)
    return client.post("/app/v0/rpc/computePoseSignal", json=payload)


def test_list_pose_signals_reports_movement_functions(client: TestClient):
    """The signal list names the movement function behind each choice."""
    response = client.post("/app/v0/rpc/listPoseSignals", json={})
    assert response.status_code == 200, response.text
    body = response.json()
    functions = {s["id"]: s["function"] for s in body["signals"]}
    assert functions["speed"] == "movement.kinematics.compute_speed"
    assert functions["head_direction"] == "movement.kinematics.compute_forward_vector_angle"


def test_speed_spikes_where_the_keypoint_jumps(client: TestClient, project):
    """Speed has one value per frame, and the 50 px jump at frame 20 is its peak."""
    pytest.importorskip("movement")
    response = _compute(client, signal="speed")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["nFrames"] == N_FRAMES
    assert body["unit"] == "px/s"
    series = body["series"][0]
    assert [s["keypoint"] for s in body["series"]] == ["nose", "ear_l", "ear_r"]
    values = np.array([np.nan if v is None else v for v in series["values"]])
    assert len(values) == N_FRAMES
    assert int(np.nanargmax(values)) in (19, 20, 21)
    # steady 1 px/frame at 10 fps away from the jump
    assert values[10] == pytest.approx(10.0)


def test_confidence_threshold_blanks_low_likelihood_frames(client: TestClient, project):
    """Frames below the threshold have no position, so the signal is None there."""
    pytest.importorskip("movement")
    response = _compute(
        client, signal="speed", confidenceThreshold=0.5
    )
    assert response.status_code == 200, response.text
    assert response.json()["series"][0]["values"][5] is None


def test_velocity_series_are_tagged_with_keypoint_and_axis(client: TestClient, project):
    """Velocity has one series per keypoint and axis, tagged so the UI can filter them."""
    pytest.importorskip("movement")
    response = _compute(client, signal="velocity")
    assert response.status_code == 200, response.text
    tags = [(s["keypoint"], s["space"]) for s in response.json()["series"]]
    assert ("nose", "x") in tags and ("nose", "y") in tags
    assert len(tags) == 6


def test_head_direction_needs_two_keypoints(client: TestClient, project):
    """Head direction returns one series; the same keypoint twice is rejected."""
    pytest.importorskip("movement")
    ok = _compute(
        client,
        signal="head_direction",
        params={"left_keypoint": "ear_l", "right_keypoint": "ear_r"},
    )
    assert ok.status_code == 200, ok.text
    assert len(ok.json()["series"]) == 1
    bad = _compute(
        client,
        signal="head_direction",
        params={"left_keypoint": "ear_l", "right_keypoint": "ear_l"},
    )
    assert bad.status_code == 422


def test_prediction_path_outside_model_dir_is_rejected(client: TestClient, project):
    """A path that climbs out of the model directory is refused."""
    response = _compute(client, signal="speed", predictionPath="../project.yaml")
    assert response.status_code == 400


def test_unknown_signal_is_rejected(client: TestClient, project):
    response = _compute(client, signal="os.system")
    assert response.status_code == 422


def _write_video(path: Path, n_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
    writer.release()


def test_kmeans_selects_frames_inside_the_segment(tmp_path):
    """Frames picked with a frame range all lie in [start, end)."""
    from litpose_app.config import Config
    from litpose_app.utils.video.frame_selection import frame_selection_kmeans_impl

    video = tmp_path / "v.mp4"
    _write_video(video, 60)
    picked = frame_selection_kmeans_impl(Config(), video, 4, 30, 45)
    assert len(picked) == 4
    assert all(30 <= int(i) < 45 for i in picked)


def test_extract_rejects_segment_shorter_than_frame_count(client: TestClient, register_project):
    """Asking for more frames than the segment holds is a 422 before any work."""
    data_dir = register_project("demo", views=["cam"], keypoints=["nose"])
    response = client.post(
        "/app/v0/rpc/extractFrames",
        json={
            "projectKey": "demo",
            "session": {
                "views": [{"viewName": "cam", "videoPath": str(data_dir / "videos" / "s_cam.mp4")}]
            },
            "labelFile": {
                "views": [{"viewName": "cam", "csvPath": str(data_dir / "CollectedData_cam.csv")}]
            },
            "method": "random",
            "options": {"nFrames": 10, "frameRange": [100, 105]},
        },
    )
    assert response.status_code == 422, response.text
    assert not (data_dir / "labeled-data").exists()


def test_uniform_selection_spreads_frames_evenly_over_the_segment():
    """Uniform picks the centre of n equal bins of [start, end)."""
    from litpose_app.tasks.extract_frames import uniform_frame_indices

    assert list(uniform_frame_indices(100, 200, 4)) == [112, 137, 162, 187]
    with pytest.raises(ValueError):
        uniform_frame_indices(0, 3, 4)
