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


def test_labels_in_one_view_only_are_listed_when_asked_for(client: TestClient, project):
    data_dir, _, labels = project
    labels["camB"][2, 1] = np.nan  # leftEar missing in camB
    labels["camA"][4, 0] = np.nan  # nose only in camC
    labels["camB"][4, 0] = np.nan
    _save(data_dir, labels)

    assert _audit(client, data_dir).json()["items"] == []  # default: only labels in 2+ views

    items = _audit(client, data_dir, onlyTwoOrMoreViews=False).json()["items"]

    assert {i["kind"] for i in items} == {"oneview"}
    # one finding per view without the label, opened in that view
    assert sorted((i["keypoint"], i["view"]) for i in items) == [
        ("leftEar", "camB"), ("nose", "camA"), ("nose", "camB"),
    ]
    assert all(i["imgPaths"][i["view"]].startswith(f"labeled-data/{SESSION}_{i['view']}/") for i in items)


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


# ---------------------------------------------------------------- rigid shape
RIGID = ['tip', 'base', 'left', 'right', 'back']
RIGID_SHAPE = np.array([[0.10, 0, 0], [0.04, 0, 0.01], [0.03, 0.03, 0.02], [0.03, -0.03, 0.02], [-0.05, 0, 0.05]])


def _rigid_project(register_project, n_frames: int = 12):
    '''A rigid group of five keypoints in random poses, labeled in all three views.'''
    data_dir = register_project('p', views=VIEWS, keypoints=RIGID)
    cg = _rig()
    (data_dir / 'calibrations').mkdir(parents=True, exist_ok=True)
    cg.dump(str(data_dir / 'calibrations' / f'{SESSION}.toml'))
    rng = np.random.default_rng(1)
    p3 = np.empty((n_frames, len(RIGID), 3))
    for f in range(n_frames):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q *= np.sign(np.linalg.det(q))
        p3[f] = RIGID_SHAPE @ q.T + rng.uniform(-0.05, 0.05, size=3)
    return data_dir, cg, p3


def _save_3d(data_dir: Path, cg: CameraGroup, p3: np.ndarray, drop: dict | None = None) -> None:
    for c, v in enumerate(VIEWS):
        xy = np.asarray(cg.cameras[c].project(p3.reshape(-1, 3))).reshape(len(p3), len(RIGID), 2)
        for f, k in (drop or {}).get(v, []):
            xy[f, k] = np.nan
        cols = pd.MultiIndex.from_product([['me'], RIGID, ['x', 'y']])
        idx = [f'labeled-data/{SESSION}_{v}/img{f:04d}.png' for f in range(len(xy))]
        pd.DataFrame(xy.reshape(len(xy), -1), index=idx, columns=cols).to_csv(data_dir / f'CollectedData_{v}.csv')


def test_rigid_group_labels_that_fit_have_no_shape_findings(client: TestClient, register_project):
    data_dir, cg, p3 = _rigid_project(register_project)
    _save_3d(data_dir, cg, p3)

    body = _audit(client, data_dir, rigidGroups=[RIGID]).json()

    assert body['items'] == []
    assert body['shapeGroups'] == [RIGID]
    assert body['shapeProblems'] == []


def test_only_the_chosen_keypoints_are_reported(client: TestClient, project):
    data_dir, _, labels = project
    labels["camB"][2, 0] += [15, -10]  # nose
    labels["camA"][3, 1] += [0, 20]  # leftEar
    _save(data_dir, labels)

    assert len(_audit(client, data_dir).json()["items"]) == 2
    items = _audit(client, data_dir, keypoints=["leftEar"]).json()["items"]

    assert [it["keypoint"] for it in items] == ["leftEar"]
    assert _audit(client, data_dir, keypoints=[]).json()["items"] == []


def test_a_label_that_breaks_the_group_distances_is_found_though_the_views_agree(client: TestClient, register_project):
    data_dir, cg, p3 = _rigid_project(register_project)
    p3[5, 2] += [0.03, 0.02, 0.0]  # 'left' in frame 5: a consistent 3D point, but not where the group has it
    _save_3d(data_dir, cg, p3)

    items = _audit(client, data_dir, rigidGroups=[RIGID]).json()["items"]

    assert [(it["kind"], it["keypoint"], it["partner"]) for it in items] == [("shape", "left", None)]
    it = items[0]
    assert "img0005" in it["imgPaths"]["camA"]
    assert it["offPairs"] >= 2 and it["checkedPairs"] == 4
    assert it["group"] == RIGID
    assert len(it["groupPairs"]) == 10
    off = [p for p in it["groupPairs"] if p["off"]]
    assert off and all("left" in (p["a"], p["b"]) for p in off)  # the others still agree with each other
    pair = next(p for p in it["groupPairs"] if {p["a"], p["b"]} == {"tip", "base"})
    assert pair["distance"] == pytest.approx(np.linalg.norm(RIGID_SHAPE[0] - RIGID_SHAPE[1]), rel=1e-3)
    assert pair["usual"] == pytest.approx(pair["distance"], rel=1e-3)


def test_a_single_off_pair_names_both_keypoints(client: TestClient, register_project):
    data_dir, cg, p3 = _rigid_project(register_project)
    _save_3d(data_dir, cg, p3)

    items = _audit(client, data_dir, rigidGroups=[["tip", "base", "back"]]).json()["items"]
    assert items == []

    # 'tip' swings around 'back': its distance to 'back' stays, the one to 'base' does not.
    v = p3[4, 0] - p3[4, 4]
    axis = np.cross(v, [0.3, 0.5, 0.8])
    axis /= np.linalg.norm(axis)
    a = 0.6
    p3[4, 0] = p3[4, 4] + v * np.cos(a) + np.cross(axis, v) * np.sin(a)
    _save_3d(data_dir, cg, p3)
    items = _audit(client, data_dir, rigidGroups=[["tip", "base", "back"]]).json()["items"]

    assert len(items) == 1
    assert {items[0]["keypoint"], items[0]["partner"]} == {"tip", "base"}
    assert "one of the two" in items[0]["reason"]


def test_rigid_group_problems_are_reported(client: TestClient, register_project):
    data_dir, cg, p3 = _rigid_project(register_project)
    _save_3d(data_dir, cg, p3)

    body = _audit(client, data_dir, rigidGroups=[["tip", "nope"]]).json()

    assert body["shapeGroups"] == []
    assert any("nope" in p for p in body["shapeProblems"])
    assert any("3 or more" in p for p in body["shapeProblems"])


def test_the_limit_is_a_number_of_spreads(client: TestClient, register_project):
    data_dir, cg, p3 = _rigid_project(register_project, n_frames=40)
    rng = np.random.default_rng(3)
    p3 += rng.normal(scale=0.004, size=p3.shape)  # a group that is only roughly rigid
    p3[5, 2] += [0.08, 0.05, 0.0]
    _save_3d(data_dir, cg, p3)

    def run(strictness):
        return _audit(client, data_dir, rigidGroups=[RIGID], defaultTolerancePx=0.5, shapeStrictness=strictness).json()

    strict, loose = run(2.0), run(6.0)

    assert len(strict["shapePairs"]) == 10
    for p in strict["shapePairs"]:
        assert p["limit"] == pytest.approx(2.0 * p["spread"])
        assert p["frames"] == 40 and p["distance"] is None
    assert [p["limit"] for p in loose["shapePairs"]] == pytest.approx([3 * p["limit"] for p in strict["shapePairs"]])
    assert any(i["keypoint"] == "left" and "img0005" in i["imgPaths"]["camA"] for i in loose["items"])
    assert len(loose["items"]) <= len(strict["items"])


def test_shape_guides_follow_unsaved_edits(client: TestClient, register_project):
    data_dir, cg, p3 = _rigid_project(register_project)
    truth = p3.copy()
    p3[5, 0] += [0.03, 0.02, 0.0]  # 'tip' is wrong on disk
    _save_3d(data_dir, cg, p3)
    before = (data_dir / "CollectedData_camA.csv").read_text()

    def guides(points):
        labels = {}
        for c, v in enumerate(VIEWS):
            xy = np.asarray(cg.cameras[c].project(points)).reshape(len(RIGID), 2)
            labels[v] = {kp: [float(xy[k, 0]), float(xy[k, 1])] for k, kp in enumerate(RIGID)}
        labels["camB"]["back"] = labels["camC"]["back"] = None  # one view only: cannot be measured
        r = client.post(
            "/app/v0/rpc/getShapeGuides",
            json={
                "projectKey": "p",
                "views": [{"viewName": v, "csvPath": str(data_dir / f"CollectedData_{v}.csv")} for v in VIEWS],
                "rigidGroups": [RIGID],
                "imgPaths": {v: f"labeled-data/{SESSION}_{v}/img0005.png" for v in VIEWS},
                "labels": labels,
                "keypoint": "left",
            },
        )
        assert r.status_code == 200
        return r.json()

    as_saved, corrected = guides(p3[5]), guides(truth[5])

    assert as_saved["group"] == RIGID
    off = [p for p in as_saved["groupPairs"] if p["off"]]
    assert off and all("tip" in (p["a"], p["b"]) for p in off)
    assert not any(p["off"] for p in corrected["groupPairs"])
    back = [p for p in corrected["groupPairs"] if "back" in (p["a"], p["b"])]
    assert len(back) == 4 and all(p["distance"] is None for p in back)
    assert (data_dir / "CollectedData_camA.csv").read_text() == before  # nothing is saved


def test_sessions_of_different_size_share_one_set_of_distances(client: TestClient, register_project):
    """A second session shows a 15% larger animal: no findings, and its size is reported."""
    data_dir, cg, p3 = _rigid_project(register_project, n_frames=24)
    cg.dump(str(data_dir / "calibrations" / "sess2.toml"))
    centre = p3[12:].mean(axis=1, keepdims=True)
    p3[12:] = centre + 1.15 * (p3[12:] - centre)
    for c, v in enumerate(VIEWS):
        xy = np.asarray(cg.cameras[c].project(p3.reshape(-1, 3))).reshape(len(p3), len(RIGID), 2)
        cols = pd.MultiIndex.from_product([["me"], RIGID, ["x", "y"]])
        idx = [f"labeled-data/{SESSION if f < 12 else 'sess2'}_{v}/img{f:04d}.png" for f in range(len(xy))]
        pd.DataFrame(xy.reshape(len(xy), -1), index=idx, columns=cols).to_csv(data_dir / f"CollectedData_{v}.csv")

    body = _audit(client, data_dir, rigidGroups=[RIGID]).json()

    assert [i for i in body["items"] if i["kind"] == "shape"] == []
    sizes = body["shapeSessionSize"]
    assert sizes["sess2"] / sizes[SESSION] == pytest.approx(1.15, rel=1e-2)


def test_named_animals_get_their_own_distances(client: TestClient, register_project):
    """Two animals of different proportions: one set of distances per animal, found by name."""
    data_dir, cg, p3 = _rigid_project(register_project, n_frames=24)
    cg.dump(str(data_dir / "calibrations" / "day2_Bo.toml"))
    cg.dump(str(data_dir / "calibrations" / "day1_Al.toml"))
    p3[12:, 0] += 0.6 * (p3[12:, 0] - p3[12:, 1])  # Bo has a longer 'tip'
    for c, v in enumerate(VIEWS):
        xy = np.asarray(cg.cameras[c].project(p3.reshape(-1, 3))).reshape(len(p3), len(RIGID), 2)
        cols = pd.MultiIndex.from_product([["me"], RIGID, ["x", "y"]])
        idx = [f"labeled-data/{'day1_Al' if f < 12 else 'day2_Bo'}_{v}/img{f:04d}.png" for f in range(len(xy))]
        pd.DataFrame(xy.reshape(len(xy), -1), index=idx, columns=cols).to_csv(data_dir / f"CollectedData_{v}.csv")

    pooled = _audit(client, data_dir, rigidGroups=[RIGID]).json()
    named = _audit(client, data_dir, rigidGroups=[RIGID], animals=["al", "Bo", "Cy"]).json()

    def tip_base(body, animal):
        return next(p for p in body["shapePairs"] if (p["a"], p["b"], p["animal"]) == ("tip", "base", animal))

    assert named["items"] == []
    assert named["shapeAnimalSessions"] == {"al": 1, "Bo": 1}
    assert named["shapeUnmatchedSessions"] == [] and named["shapeSessionSize"] == {}
    assert len(named["shapePairs"]) == 20
    assert tip_base(named, "Bo")["usual"] == pytest.approx(1.6 * tip_base(named, "al")["usual"], rel=1e-2)
    # Without names the two animals share one range for tip-base, which is then much wider.
    assert tip_base(pooled, None)["spread"] > 5 * tip_base(named, "Bo")["spread"]


def test_a_template_with_the_learned_distances_is_returned_to_draw(client: TestClient, register_project):
    data_dir, cg, p3 = _rigid_project(register_project)
    _save_3d(data_dir, cg, p3)

    templates = _audit(client, data_dir, rigidGroups=[RIGID]).json()["shapeTemplates"]

    assert [(t["animal"], t["keypoints"]) for t in templates] == [(None, RIGID)]
    xyz = np.array(templates[0]["points"])
    dist = lambda x: np.linalg.norm(x[:, None] - x[None], axis=2)  # noqa: E731
    assert dist(xyz) == pytest.approx(dist(RIGID_SHAPE), abs=1e-3)
    # Not the mirror image: the same handedness as the true shape.
    triple = lambda x: np.linalg.det(np.stack([x[1] - x[0], x[2] - x[0], x[4] - x[0]]))  # noqa: E731
    assert np.sign(triple(xyz)) == np.sign(triple(RIGID_SHAPE))
