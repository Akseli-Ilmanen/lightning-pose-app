"""Rigid-group check of the label audit (#207): labels that break the distances of a rigid group.

Some keypoints keep their distances to each other (a head, a skull implant, a tool).
The user names such a group. The check needs no model and no template:

1. Every keypoint labeled consistently in two or more views is triangulated.
2. Per pair of keypoints, the usual 3D distance (median over all frames) and its
   spread (robust, from the median absolute deviation) are learned from the labels.
3. In a frame, a pair is off when its distance is more than ``factor`` spreads from
   usual. The keypoint in the most off pairs is the one that is probably wrong:
   a wrong label breaks its distances to all others, while those still agree
   with each other.

Animals differ in size, and the label file does not say which animal a session shows.
So every session gets its own size factor (the median of distance / usual over its
frames and pairs), and distances are compared after dividing by it.

Distances are in the units of the calibration (of a session of size 1).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

# A pair of keypoints needs this many frames with both triangulated to count.
MIN_PAIR_FRAMES = 5
# The spread of a pair is at least this fraction of its usual distance, so that
# very consistent labels do not make the check flag hair-thin differences.
MIN_REL_SPREAD = 0.02
MIN_GROUP_SIZE = 3
# A session needs this many measured distances for its own size factor.
MIN_SIZE_MEASUREMENTS = 6
# A single off pair does not say which of its two keypoints is wrong, and one pair in
# many is off by chance: it is only reported when it is this many times over the limit.
SINGLE_PAIR_FACTOR = 1.5


def learn_distances(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Usual distance, spread and frame count of every pair of a keypoint group.

    pts: (N, G, 3) triangulated keypoints, NaN where unknown.
    Returns (usual (G, G), spread (G, G), frames (G, G)); NaN where a pair was seen
    together in fewer than MIN_PAIR_FRAMES frames.
    """
    G = pts.shape[1]
    usual = np.full((G, G), np.nan)
    spread = np.full((G, G), np.nan)
    frames = np.zeros((G, G), int)
    for i in range(G):
        for j in range(i + 1, G):
            d = np.linalg.norm(pts[:, i] - pts[:, j], axis=1)
            d = d[np.isfinite(d)]
            frames[i, j] = frames[j, i] = len(d)
            if len(d) < MIN_PAIR_FRAMES:
                continue
            med = float(np.median(d))
            mad = 1.4826 * float(np.median(np.abs(d - med)))
            usual[i, j] = usual[j, i] = med
            spread[i, j] = spread[j, i] = max(mad, MIN_REL_SPREAD * med)
    return usual, spread, frames


def session_sizes(usual: np.ndarray, sessions: list[np.ndarray]) -> list[float]:
    """Size of the group in each session: the median of distance / usual (1 with too few measurements)."""
    iu = np.triu_indices(len(usual), 1)
    out = []
    for pts in sessions:
        d = np.linalg.norm(pts[:, :, None] - pts[:, None], axis=3)
        with np.errstate(invalid="ignore"):
            ratio = (d / usual[None])[:, iu[0], iu[1]]
        ratio = ratio[np.isfinite(ratio)]
        out.append(float(np.median(ratio)) if len(ratio) >= MIN_SIZE_MEASUREMENTS else 1.0)
    return out


def learn_scaled_distances(sessions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """As learn_distances over all sessions, each divided by its own size. Also returns the sizes."""
    sizes = [1.0] * len(sessions)
    for _ in range(3):
        usual, spread, frames = learn_distances(np.concatenate([p / s for p, s in zip(sessions, sizes)]))
        if not np.isfinite(usual).any():
            break
        new = session_sizes(usual, sessions)
        done = np.allclose(new, sizes, rtol=1e-3)
        sizes = new
        if done:
            break
    usual, spread, frames = learn_distances(np.concatenate([p / s for p, s in zip(sessions, sizes)]))
    return usual, spread, frames, sizes


def pair_scores(usual: np.ndarray, spread: np.ndarray, p: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """One frame: distances (G, G) and scores (G, G) of its pairs; score > 1 means off.

    p: (G, 3). NaN where a pair cannot be measured in this frame or was not learned.
    """
    dist = np.linalg.norm(p[:, None] - p[None], axis=2)
    with np.errstate(invalid="ignore"):
        score = np.abs(dist - usual) / (factor * spread)
    np.fill_diagonal(dist, np.nan)
    np.fill_diagonal(score, np.nan)
    return dist, score


def blame(score: np.ndarray, skip: set[int] = frozenset()) -> tuple[int, int | None] | None:
    """The keypoint that is probably wrong, from one frame's pair scores.

    Returns (keypoint, partner). ``partner`` is None when the keypoint is in two or more
    off pairs; with a single off pair the two keypoints cannot be told apart and the
    partner is given (such a pair must be SINGLE_PAIR_FACTOR over the limit). None when no pair is off. Keypoints in ``skip`` are never blamed.
    """
    off = np.nan_to_num(score, nan=0.0) > 1
    counts = off.sum(1)
    weight = np.where(off, np.nan_to_num(score, nan=0.0), 0.0).sum(1)
    cand = [
        k
        for k in range(len(counts))
        if k not in skip and (counts[k] >= 2 or (counts[k] == 1 and weight[k] > SINGLE_PAIR_FACTOR))
    ]
    if not cand:
        return None
    k = max(cand, key=lambda i: (counts[i], weight[i]))
    if counts[k] >= 2:
        return k, None
    return k, int(np.nonzero(off[k])[0][0])


def template_points(usual: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """3D positions (G, 3) whose pairwise distances match ``usual``, to draw the group.

    Only for display: the check itself uses the distances. Keypoints without a learned
    distance are NaN. Distances cannot tell an arrangement from its mirror image, so
    the frames (pts: (N, G, 3)) decide.
    """
    G = len(usual)
    out = np.full((G, 3), np.nan)
    known = np.isfinite(usual)
    np.fill_diagonal(known, False)
    idx = np.nonzero(known.any(axis=1))[0]
    if len(idx) < 3:
        return out
    D = usual[np.ix_(idx, idx)]
    K = known[np.ix_(idx, idx)]
    # Start: classical scaling, unknown distances filled with the shortest path through known ones.
    full = np.where(K, D, np.inf)
    np.fill_diagonal(full, 0.0)
    for m in range(len(idx)):
        full = np.minimum(full, full[:, [m]] + full[[m], :])
    if not np.isfinite(full).all():
        return out
    J = np.eye(len(idx)) - 1.0 / len(idx)
    w, v = np.linalg.eigh(-0.5 * J @ (full**2) @ J)
    order = np.argsort(w)[::-1][:3]
    x0 = v[:, order] * np.sqrt(np.clip(w[order], 1e-12, None))
    ii, jj = np.nonzero(np.triu(K, 1))

    def resid(flat):
        X = flat.reshape(-1, 3)
        return np.linalg.norm(X[ii] - X[jj], axis=1) - D[ii, jj]

    X = least_squares(resid, x0.ravel()).x.reshape(-1, 3)
    X -= X.mean(0)
    # Mirror or not: in which handedness do the frames see the group?
    vote = 0.0
    for p in pts[:, idx]:
        ok = np.isfinite(p).all(axis=1)
        if ok.sum() < 4:
            continue
        H = (X[ok] - X[ok].mean(0)).T @ (p[ok] - p[ok].mean(0))
        vote += float(np.linalg.det(H)) / (np.linalg.norm(H) ** 3 + 1e-30)
    if vote < 0:
        X = X * np.array([1.0, 1.0, -1.0])
    out[idx] = X
    return out
