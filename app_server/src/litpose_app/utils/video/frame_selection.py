"""K-means based frame selection for diverse frame extraction from videos."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from litpose_app.config import Config
from litpose_app.utils.video import video_capture

logger = logging.getLogger(__name__)


def frame_selection_kmeans_impl(
    config: Config,
    video_path: Path,
    n_frames: int,
    beg_frame: int = 0,
    end_frame: int | None = None,
) -> NDArray[np.integer]:
    """
    Reads the frames in [beg_frame, end_frame), computes motion energy, clusters
    into n_frames clusters and picks a frame out of each cluster.

    end_frame=None means the end of the video. Only the frames in the range are
    decoded, so a short segment of a long video is quick. Returned indices are
    frame indices in the whole video.
    """
    with video_capture(video_path) as cap:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # leave room for context at the end of the video
    last = frame_count - config.FRAME_EXTRACT_N_CONTEXT_FRAMES
    end_frame = last if end_frame is None else min(end_frame, last)
    beg_frame = max(0, beg_frame)
    end_frame = max(beg_frame, end_frame)
    if end_frame - beg_frame < n_frames:
        raise ValueError(
            f"Cannot select {n_frames} frames from a range of {end_frame - beg_frame} frames."
        )

    frames = _read_frame_range(config, video_path, beg_frame, end_frame)
    batches = np.reshape(frames, (frames.shape[0], -1))

    # take temporal diffs
    logger.info("Computing motion energy")
    me = np.concatenate([np.zeros((1, batches.shape[1])), np.diff(batches, axis=0)])
    # take absolute values and sum over all pixels to get motion energy
    me = np.sum(np.abs(me), axis=1)

    # find high me frames, defined as those with me larger than nth percentile me
    prctile = 50 if frame_count < 1e5 else 75  # take fewer frames if there are many
    idxs_high_me = np.where(me > np.percentile(me, prctile))[0]
    # just use all frames if the user wants to label a large fraction of the frames
    # (helpful for very short videos)
    if len(idxs_high_me) < n_frames:
        idxs_high_me = np.arange(me.shape[0])

    # compute pca over high me frames
    logger.info("performing pca over high motion energy frames...")
    pca_obj = PCA(n_components=np.min([batches[idxs_high_me].shape[0], 32]))
    embedding = pca_obj.fit_transform(X=batches[idxs_high_me])
    del batches  # free up memory

    # cluster low-d pca embeddings
    logger.info("performing kmeans clustering...")
    _, centers = _run_kmeans(x=embedding, n_clusters=n_frames)
    # centers is initially of shape (n_clusters, n_pcs); reformat
    centers = centers.T[None, :]

    # find high me frame that is closest to each cluster center
    # embedding is shape (n_frames, n_pcs)
    # centers is shape (1, n_pcs, n_clusters)
    dists = np.linalg.norm(embedding[:, :, None] - centers, axis=1)
    # dists is shape (n_frames, n_clusters)
    idxs_prototypes_ = np.argmin(dists, axis=0)
    # now index into high me frames to get overall indices, add offset
    idxs_prototypes = idxs_high_me[idxs_prototypes_] + beg_frame

    return idxs_prototypes


def _read_frame_range(config: Config, video_file: Path, beg: int, end: int) -> np.ndarray:
    """Read frames [beg, end) from video_file, resized like _read_nth_frames."""
    frames = []
    with video_capture(video_file) as cap:
        if beg > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, beg)
        for _ in tqdm(range(end - beg)):
            ret, frame = cap.read()
            if not ret:
                break
            frame_resize = cv2.resize(
                frame, (config.FRAME_EXTRACT_RESIZE_DIMS, config.FRAME_EXTRACT_RESIZE_DIMS)
            )
            frames.append(cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB).astype(np.float16))
    return np.array(frames)


def _read_nth_frames(
    config: Config,
    video_file: Path,
    n: int = 1,
) -> np.ndarray:
    """Read every nth frame from video_file, resized and returned as a float16 array."""
    # Open the video file
    with video_capture(video_file) as cap:
        frames = []
        frame_counter = 0
        frame_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        with tqdm(total=int(frame_total)) as pbar:
            while cap.isOpened():
                # Read the next frame
                ret, frame = cap.read()
                if ret:
                    # If the frame was successfully read, then process it
                    if frame_counter % n == 0:
                        frame_resize = cv2.resize(
                            frame,
                            (
                                config.FRAME_EXTRACT_RESIZE_DIMS,
                                config.FRAME_EXTRACT_RESIZE_DIMS,
                            ),
                        )
                        frame_gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
                        frames.append(frame_gray.astype(np.float16))
                    frame_counter += 1
                    pbar.update(1)
                else:
                    # If we couldn't read a frame, we've probably reached the end
                    break

        return np.array(frames)


def _run_kmeans(x: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit k-means on x and return (cluster_labels, cluster_centers)."""
    kmeans_obj = KMeans(n_clusters, n_init="auto")
    kmeans_obj.fit(x)
    cluster_labels = kmeans_obj.labels_
    cluster_centers = kmeans_obj.cluster_centers_
    return cluster_labels, cluster_centers
