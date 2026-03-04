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

"""Event completion evaluation task — PyTorch port."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from tcc.evaluation.task import Task
from tcc.evaluation.task_utils import get_targets_from_labels

logger = logging.getLogger(__name__)


class VectorRegression(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for multi-output linear regression.

    Fits one :class:`~sklearn.linear_model.LinearRegression` per output
    dimension and stacks the predictions.
    """

    def __init__(self) -> None:
        self.models_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VectorRegression":
        """Fit one linear model per output column.

        Parameters
        ----------
        X:
            Feature matrix of shape ``[N, D]``.
        y:
            Target matrix of shape ``[N, K]``.
        """
        self.models_ = []
        if y.ndim == 1:
            y = y[:, np.newaxis]
        for k in range(y.shape[1]):
            model = LinearRegression()
            model.fit(X, y[:, k])
            self.models_.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for each output column and stack.

        Returns
        -------
        np.ndarray:
            Predictions of shape ``[N, K]``.
        """
        preds = [m.predict(X) for m in self.models_]
        return np.column_stack(preds)


def fit_model(
    train_embs: np.ndarray,
    train_labels: list,
    val_embs: np.ndarray,
    val_labels: list,
    num_classes: int,
) -> float:
    """Fit a linear regression to predict event completion fraction.

    Parameters
    ----------
    train_embs:
        Training embeddings, shape ``[N_train, D]``.
    train_labels:
        List of 1-D integer label arrays, one per video.
        Will be concatenated and converted to regression targets.
    val_embs:
        Validation embeddings, shape ``[N_val, D]``.
    val_labels:
        List of 1-D integer label arrays for validation.
    num_classes:
        Number of event classes.

    Returns
    -------
    float:
        Mean absolute error on the validation set.
    """
    train_targets_list = get_targets_from_labels(train_labels, num_classes)
    val_targets_list = get_targets_from_labels(val_labels, num_classes)

    # Stack all targets
    train_targets = np.concatenate(train_targets_list, axis=0)
    val_targets = np.concatenate(val_targets_list, axis=0)

    model = VectorRegression()
    model.fit(train_embs, train_targets)
    preds = model.predict(val_embs)

    mae = np.mean(np.abs(preds - val_targets))
    return float(mae)


class EventCompletion(Task):
    """Event completion prediction via linear regression.

    Fits a multi-output linear regression to predict the fraction of
    each event that has been completed at each frame.  This is an
    embedding-based downstream task.
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
        """Fit event completion model and report MAE.

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
            - ``'labels_list'``: list of per-video 1-D label arrays
            - ``'num_classes'``: int
        writer:
            Optional TensorBoard SummaryWriter.

        Returns
        -------
        dict:
            Mapping of metric names to MAE values.
        """
        metrics: Dict[str, float] = {}

        if "train" not in datasets or "val" not in datasets:
            logger.warning(
                "EventCompletion requires 'train' and 'val' splits. "
                "Available: %s",
                list(datasets.keys()),
            )
            return metrics

        train_data = datasets["train"]
        val_data = datasets["val"]

        num_classes = train_data.get("num_classes", 1)
        if num_classes < 1:
            logger.warning("Skipping event completion: num_classes < 1.")
            return metrics

        mae = fit_model(
            train_embs=train_data["embeddings"],
            train_labels=train_data["labels_list"],
            val_embs=val_data["embeddings"],
            val_labels=val_data["labels_list"],
            num_classes=num_classes,
        )

        key = "event_completion/mae"
        metrics[key] = mae
        logger.info("Step %d | %s = %.4f", global_step, key, mae)
        if writer is not None:
            writer.add_scalar(key, mae, global_step)

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
            "EventCompletion uses embeddings, not iterators."
        )
