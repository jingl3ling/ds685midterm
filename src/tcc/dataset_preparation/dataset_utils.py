# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core dataset-preparation utilities -- zero TensorFlow dependencies.

Functions for extracting frames from video files, merging per-annotator
labels, assigning labels to timestamps, and saving frames in the
image-folder layout expected by :class:`tcc.datasets.VideoDataset`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def video_to_frames(
    video_path: Union[str, Path],
    rotate: bool = False,
    fps: float = 0,
    resize: bool = True,
    width: int = 224,
    height: int = 224,
) -> Tuple[List[np.ndarray], List[float], float]:
    """Extract frames from a video file using OpenCV.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    rotate:
        If ``True``, rotate each frame 90 degrees clockwise.
    fps:
        Target frames-per-second for extraction. ``0`` means use the
        native video FPS (i.e. extract every frame).
    resize:
        If ``True``, resize frames to *width* x *height*.
    width, height:
        Target dimensions when *resize* is ``True``.

    Returns
    -------
    frames:
        List of uint8 RGB numpy arrays with shape ``(H, W, 3)``.
    timestamps:
        List of timestamps (seconds) for each extracted frame.
    native_fps:
        The native FPS of the source video.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 30.0  # sensible fallback

    # Determine frame step
    if fps > 0 and fps < native_fps:
        frame_step = int(round(native_fps / fps))
    else:
        frame_step = 1
        fps = native_fps

    frames: List[np.ndarray] = []
    timestamps: List[float] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if resize:
                frame = cv2.resize(
                    frame, (width, height), interpolation=cv2.INTER_AREA
                )
            frames.append(frame)
            timestamps.append(frame_idx / native_fps)
        frame_idx += 1

    cap.release()
    return frames, timestamps, native_fps


def merge_annotations(
    labels: np.ndarray,
    expected_n: int,
) -> np.ndarray:
    """Merge per-annotator label arrays via majority voting.

    Parameters
    ----------
    labels:
        Array of shape ``(num_annotators, num_frames)`` containing
        integer class labels from each annotator.
    expected_n:
        Expected number of frames. If the label arrays are shorter or
        longer, they are truncated or zero-padded to this length.

    Returns
    -------
    merged:
        Integer array of shape ``(expected_n,)`` with the majority-vote
        label for each frame.
    """
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels[np.newaxis, :]

    num_annotators, num_frames = labels.shape

    # Adjust to expected length
    if num_frames > expected_n:
        labels = labels[:, :expected_n]
    elif num_frames < expected_n:
        padding = np.zeros(
            (num_annotators, expected_n - num_frames), dtype=labels.dtype
        )
        labels = np.concatenate([labels, padding], axis=1)

    # Majority vote per frame
    merged = np.zeros(expected_n, dtype=np.int64)
    for i in range(expected_n):
        votes = labels[:, i]
        counts = np.bincount(votes.astype(np.int64))
        merged[i] = np.argmax(counts)

    return merged


def label_timestamps(
    timestamps: Sequence[float],
    annotations: np.ndarray,
) -> np.ndarray:
    """Assign frame labels based on timestamps and annotation array.

    Each frame is assigned the label from *annotations* at the index
    corresponding to its position. If the annotations array is shorter
    than the number of timestamps, excess frames receive label 0.

    Parameters
    ----------
    timestamps:
        Sequence of frame timestamps (used only for length).
    annotations:
        1-D integer array of per-frame labels.

    Returns
    -------
    labels:
        Integer array of shape ``(len(timestamps),)`` with a label per
        frame.
    """
    n = len(timestamps)
    annotations = np.asarray(annotations).ravel()
    labels = np.zeros(n, dtype=np.int64)
    copy_len = min(n, len(annotations))
    labels[:copy_len] = annotations[:copy_len]
    return labels


def save_frames(
    frames: Sequence[np.ndarray],
    output_dir: Union[str, Path],
    name: str,
    frame_labels: Optional[np.ndarray] = None,
    seq_label: Optional[Union[int, np.ndarray]] = None,
) -> Path:
    """Save extracted frames to the image-folder layout.

    Creates the directory ``output_dir/name/`` and writes:

    - ``frame_NNNNNN.jpg`` for each frame
    - ``frame_labels.npy`` if *frame_labels* is provided
    - ``seq_label.npy`` if *seq_label* is provided

    Parameters
    ----------
    frames:
        List of uint8 RGB numpy arrays, each ``(H, W, 3)``.
    output_dir:
        Root output directory.
    name:
        Name for this video sequence (becomes a sub-directory).
    frame_labels:
        Optional 1-D integer array of per-frame labels.
    seq_label:
        Optional sequence-level label (scalar int).

    Returns
    -------
    video_dir:
        Path to the created video directory.
    """
    video_dir = Path(output_dir) / name
    video_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(video_dir / f"frame_{i:06d}.jpg", quality=95)

    if frame_labels is not None:
        np.save(video_dir / "frame_labels.npy", np.asarray(frame_labels))

    if seq_label is not None:
        np.save(video_dir / "seq_label.npy", np.asarray(seq_label))

    return video_dir
