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

"""Common utilities for TCC evaluation tasks — pure numpy, no TF."""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def regression_labels_for_class(
    labels: np.ndarray,
    class_idx: int,
) -> Optional[np.ndarray]:
    """Compute normalized regression labels for a single class.

    Find the first frame where the label transitions to ``class_idx``,
    then produce a linearly-interpolated target from 0.0 at the start of
    the sequence to 1.0 at the transition frame.  Frames after the
    transition frame are set to 1.0.

    Parameters
    ----------
    labels:
        1-D integer array of per-frame class labels.
    class_idx:
        The target class to find the transition for.

    Returns
    -------
    np.ndarray or None:
        1-D float array of the same length as *labels*, or ``None`` if
        the class never appears.
    """
    indices = np.where(labels == class_idx)[0]
    if len(indices) == 0:
        return None

    transition_frame = indices[0]
    seq_len = len(labels)
    regression = np.zeros(seq_len, dtype=np.float32)

    if transition_frame == 0:
        regression[:] = 1.0
    else:
        regression[:transition_frame] = np.linspace(
            0.0, 1.0, transition_frame, endpoint=False, dtype=np.float32
        )
        regression[transition_frame:] = 1.0

    return regression


def get_regression_labels(
    class_labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Stack regression labels for all classes.

    Parameters
    ----------
    class_labels:
        1-D integer array of per-frame class labels.
    num_classes:
        Total number of classes.

    Returns
    -------
    np.ndarray:
        Shape ``[seq_len, num_classes]``.  Columns for classes that
        never appear in *class_labels* are filled with zeros.
    """
    seq_len = len(class_labels)
    result = np.zeros((seq_len, num_classes), dtype=np.float32)

    for c in range(num_classes):
        col = regression_labels_for_class(class_labels, c)
        if col is not None:
            result[:, c] = col

    return result


def get_targets_from_labels(
    all_class_labels: List[np.ndarray],
    num_classes: int,
) -> List[np.ndarray]:
    """Map per-video class labels to regression targets.

    Parameters
    ----------
    all_class_labels:
        List of 1-D integer arrays, one per video.
    num_classes:
        Total number of classes.

    Returns
    -------
    list of np.ndarray:
        Each element has shape ``[seq_len_i, num_classes]``.
    """
    return [
        get_regression_labels(labels, num_classes) for labels in all_class_labels
    ]


def unnormalize(preds: np.ndarray, seq_len: int) -> np.ndarray:
    """Convert normalized [0, 1] predictions back to frame indices.

    Parameters
    ----------
    preds:
        Array of normalized predictions in [0, 1].
    seq_len:
        Original sequence length.

    Returns
    -------
    np.ndarray:
        Integer frame indices clipped to [0, seq_len - 1].
    """
    frame_indices = preds * (seq_len - 1)
    return np.clip(np.round(frame_indices), 0, seq_len - 1).astype(np.int64)


def softmax(w: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Parameters
    ----------
    w:
        Input logits array.
    t:
        Temperature parameter.

    Returns
    -------
    np.ndarray:
        Softmax probabilities, same shape as *w*.
    """
    e = np.exp((w - np.max(w, axis=-1, keepdims=True)) / t)
    return e / np.sum(e, axis=-1, keepdims=True)
