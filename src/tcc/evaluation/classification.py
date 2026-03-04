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

"""Linear classification evaluation task — PyTorch port."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from tcc.evaluation.task import Task

logger = logging.getLogger(__name__)


def fit_linear_model(
    train_embs: np.ndarray,
    train_labels: np.ndarray,
    val_embs: np.ndarray,
    val_labels: np.ndarray,
) -> float:
    """Fit a logistic regression model and return validation accuracy.

    Parameters
    ----------
    train_embs:
        Training embeddings, shape ``[N_train, D]``.
    train_labels:
        Training labels, shape ``[N_train]``.
    val_embs:
        Validation embeddings, shape ``[N_val, D]``.
    val_labels:
        Validation labels, shape ``[N_val]``.

    Returns
    -------
    float:
        Accuracy on the validation set.
    """
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
    )
    clf.fit(train_embs, train_labels)
    return float(clf.score(val_embs, val_labels))


def fit_svm_model(
    train_embs: np.ndarray,
    train_labels: np.ndarray,
    val_embs: np.ndarray,
    val_labels: np.ndarray,
) -> float:
    """Fit an SVM classifier and return validation accuracy.

    Parameters
    ----------
    train_embs:
        Training embeddings, shape ``[N_train, D]``.
    train_labels:
        Training labels, shape ``[N_train]``.
    val_embs:
        Validation embeddings, shape ``[N_val, D]``.
    val_labels:
        Validation labels, shape ``[N_val]``.

    Returns
    -------
    float:
        Accuracy on the validation set.
    """
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(train_embs, train_labels)
    return float(clf.score(val_embs, val_labels))


class Classification(Task):
    """Linear classification on top of frozen embeddings.

    This is an embedding-based downstream task.
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
        """Fit linear models at different data fractions.

        Parameters
        ----------
        algo:
            Algorithm (unused; embeddings are pre-extracted).
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
            Mapping of metric names to accuracy values.
        """
        metrics: Dict[str, float] = {}

        if "train" not in datasets or "val" not in datasets:
            logger.warning(
                "Classification requires 'train' and 'val' splits. "
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

        # Skip if labels have only one class
        if len(np.unique(train_labels)) < 2:
            logger.warning("Skipping classification: fewer than 2 classes.")
            return metrics

        fractions = cfg.eval.classification_fractions

        for frac in fractions:
            n_train = max(1, int(len(train_embs) * frac))
            # Randomly subsample training data
            rng = np.random.RandomState(42)
            indices = rng.choice(len(train_embs), n_train, replace=False)
            sub_embs = train_embs[indices]
            sub_labels = train_labels[indices]

            # Ensure at least 2 classes in subsample
            if len(np.unique(sub_labels)) < 2:
                continue

            # Logistic Regression
            lr_acc = fit_linear_model(sub_embs, sub_labels, val_embs, val_labels)
            key_lr = f"classification/logistic_regression/frac_{frac:.2f}"
            metrics[key_lr] = lr_acc
            logger.info("Step %d | %s = %.4f", global_step, key_lr, lr_acc)
            if writer is not None:
                writer.add_scalar(key_lr, lr_acc, global_step)

            # SVM
            svm_acc = fit_svm_model(sub_embs, sub_labels, val_embs, val_labels)
            key_svm = f"classification/svm/frac_{frac:.2f}"
            metrics[key_svm] = svm_acc
            logger.info("Step %d | %s = %.4f", global_step, key_svm, svm_acc)
            if writer is not None:
                writer.add_scalar(key_svm, svm_acc, global_step)

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
            "Classification uses embeddings, not iterators."
        )
