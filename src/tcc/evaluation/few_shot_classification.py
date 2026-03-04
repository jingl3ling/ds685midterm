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

"""Few-shot classification evaluation task — PyTorch port."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

from tcc.evaluation.task import Task

logger = logging.getLogger(__name__)


def _run_few_shot_episode(
    train_embs: np.ndarray,
    train_labels: np.ndarray,
    val_embs: np.ndarray,
    val_labels: np.ndarray,
    k: int,
    rng: np.random.RandomState,
) -> float:
    """Run a single few-shot classification episode.

    Randomly sample *k* labeled examples per class from the training
    set, fit a logistic regression, and return validation accuracy.
    """
    classes = np.unique(train_labels)
    indices = []
    for c in classes:
        class_indices = np.where(train_labels == c)[0]
        if len(class_indices) < k:
            chosen = class_indices
        else:
            chosen = rng.choice(class_indices, k, replace=False)
        indices.extend(chosen.tolist())

    indices = np.array(indices)
    sub_embs = train_embs[indices]
    sub_labels = train_labels[indices]

    # Need at least 2 classes
    if len(np.unique(sub_labels)) < 2:
        return float("nan")

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
    )
    clf.fit(sub_embs, sub_labels)
    return float(clf.score(val_embs, val_labels))


class FewShotClassification(Task):
    """Few-shot classification evaluation.

    For each value of *k* (number of labeled examples per class),
    run multiple episodes and report mean accuracy with 95% confidence
    interval.  This is an embedding-based downstream task.
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
        """Run few-shot classification at each k.

        Parameters
        ----------
        algo:
            Algorithm (unused).
        global_step:
            Current training step.
        cfg:
            TCCConfig instance.
        datasets:
            Dict mapping split name to dict with keys:
            - ``'embeddings'``: np.ndarray of shape ``[N, D]``
            - ``'labels'``: np.ndarray of shape ``[N]``
        writer:
            Optional TensorBoard SummaryWriter.

        Returns
        -------
        dict:
            Mapping of metric names to values.
        """
        metrics: Dict[str, float] = {}

        if "train" not in datasets or "val" not in datasets:
            logger.warning(
                "FewShotClassification requires 'train' and 'val' splits. "
                "Available: %s",
                list(datasets.keys()),
            )
            return metrics

        train_data = datasets["train"]
        val_data = datasets["val"]
        train_embs = train_data["embeddings"]
        train_labels = train_data["labels"]
        val_embs = val_data["embeddings"]
        val_labels = val_data["labels"]

        if len(np.unique(train_labels)) < 2:
            logger.warning(
                "Skipping few-shot classification: fewer than 2 classes."
            )
            return metrics

        num_episodes = cfg.eval.few_shot_num_episodes
        k_values = cfg.eval.few_shot_num_labeled

        for k in k_values:
            accs = []
            for episode in range(num_episodes):
                rng = np.random.RandomState(episode)
                acc = _run_few_shot_episode(
                    train_embs, train_labels,
                    val_embs, val_labels,
                    k=k, rng=rng,
                )
                if not np.isnan(acc):
                    accs.append(acc)

            if len(accs) == 0:
                continue

            mean_acc = float(np.mean(accs))
            # 95% confidence interval
            if len(accs) > 1:
                sem = stats.sem(accs)
                ci_95 = 1.96 * sem
            else:
                ci_95 = 0.0

            key_mean = f"few_shot/k={k}/mean_accuracy"
            key_ci = f"few_shot/k={k}/ci_95"
            metrics[key_mean] = mean_acc
            metrics[key_ci] = ci_95
            logger.info(
                "Step %d | %s = %.4f +/- %.4f",
                global_step, key_mean, mean_acc, ci_95,
            )
            if writer is not None:
                writer.add_scalar(key_mean, mean_acc, global_step)
                writer.add_scalar(key_ci, ci_95, global_step)

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
            "FewShotClassification uses embeddings, not iterators."
        )
