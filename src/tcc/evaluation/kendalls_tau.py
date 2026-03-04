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

"""Kendall's Tau evaluation task — PyTorch port."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau

from tcc.evaluation.task import Task

logger = logging.getLogger(__name__)


def _get_kendalls_tau(
    embs_list: List[np.ndarray],
    stride: int = 5,
    distance: str = "sqeuclidean",
) -> float:
    """Compute mean pairwise Kendall's Tau across all sequence pairs.

    For each pair of embedding sequences, compute the nearest-neighbour
    alignment and then measure how well the resulting correspondence
    preserves temporal ordering via Kendall's Tau.

    Parameters
    ----------
    embs_list:
        List of embedding arrays, each of shape ``[T_i, D]``.
    stride:
        Sub-sampling stride applied to sequences before computing
        distances (reduces computational cost).
    distance:
        Distance metric passed to ``scipy.spatial.distance.cdist``.

    Returns
    -------
    float:
        Mean Kendall's Tau over all pairs.
    """
    if len(embs_list) < 2:
        return 0.0

    taus = []

    for i in range(len(embs_list)):
        for j in range(i + 1, len(embs_list)):
            emb_i = embs_list[i][::stride]
            emb_j = embs_list[j][::stride]

            # Pairwise distance matrix
            dist_matrix = cdist(emb_i, emb_j, metric=distance)

            # Nearest neighbour alignment: for each frame in i, find
            # the closest frame in j
            nn_j = np.argmin(dist_matrix, axis=1)

            # Ground-truth ordering (identity)
            gt = np.arange(len(emb_i))

            # Kendall's Tau between the NN alignment and the GT ordering
            tau, _ = kendalltau(gt, nn_j)

            if not np.isnan(tau):
                taus.append(tau)

    if len(taus) == 0:
        return 0.0

    return float(np.mean(taus))


class KendallsTau(Task):
    """Kendall's Tau evaluation on embedding sequences.

    Measures how well the learned embeddings preserve temporal ordering
    across different videos.  This is an embedding-based downstream task.
    """

    def __init__(self) -> None:
        super().__init__(downstream_task=True)

    def evaluate_embeddings(
        self,
        algo: Any,
        global_step: int,
        cfg: Any,
        datasets: Dict[str, Any],
        writer: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Compute Kendall's Tau for each split.

        Parameters
        ----------
        algo:
            Algorithm (unused; embeddings are pre-extracted).
        global_step:
            Current training step.
        cfg:
            TCCConfig instance.
        datasets:
            Dict mapping split name to dict with key
            ``'embeddings_list'`` — a list of per-video embedding arrays
            each of shape ``[T_i, D]``.
        writer:
            Optional TensorBoard SummaryWriter.

        Returns
        -------
        dict:
            Mapping of metric names to tau values.
        """
        metrics: Dict[str, float] = {}
        stride = cfg.eval.kendalls_tau_stride
        distance = cfg.eval.kendalls_tau_distance

        for split, data in datasets.items():
            embs_list = data.get("embeddings_list", [])
            if len(embs_list) < 2:
                logger.warning(
                    "Kendall's Tau requires >= 2 sequences for split '%s'.",
                    split,
                )
                continue

            tau = _get_kendalls_tau(embs_list, stride=stride, distance=distance)
            key = f"kendalls_tau/{split}"
            metrics[key] = tau
            logger.info("Step %d | %s = %.4f", global_step, key, tau)
            if writer is not None:
                writer.add_scalar(key, tau, global_step)

        return metrics

    def evaluate_iterators(
        self,
        algo: Any,
        global_step: int,
        cfg: Any,
        iterators: Dict[str, Any],
        writer: Optional[Any] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError(
            "KendallsTau uses embeddings, not iterators."
        )
