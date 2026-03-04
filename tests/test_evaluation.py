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

"""Tests for the TCC evaluation system — PyTorch port.

All tests use small numpy arrays and do not require a GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from tcc.config import TCCConfig
from tcc.evaluation.algo_loss import AlgoLoss
from tcc.evaluation.classification import Classification, fit_linear_model, fit_svm_model
from tcc.evaluation.event_completion import EventCompletion, VectorRegression
from tcc.evaluation.few_shot_classification import FewShotClassification
from tcc.evaluation.kendalls_tau import KendallsTau, _get_kendalls_tau
from tcc.evaluation.registry import get_tasks
from tcc.evaluation.task import Task
from tcc.evaluation.task_utils import (
    get_regression_labels,
    get_targets_from_labels,
    regression_labels_for_class,
    softmax,
    unnormalize,
)


# ---------------------------------------------------------------------------
# Task base class dispatch tests
# ---------------------------------------------------------------------------


class _DummyEmbeddingTask(Task):
    """Concrete embedding-based task for testing dispatch."""

    def __init__(self):
        super().__init__(downstream_task=True)

    def evaluate_embeddings(self, algo, global_step, cfg, datasets, writer=None):
        return {"dummy_emb_metric": 1.0}

    def evaluate_iterators(self, algo, global_step, cfg, iterators, writer=None):
        raise NotImplementedError


class _DummyIteratorTask(Task):
    """Concrete iterator-based task for testing dispatch."""

    def __init__(self):
        super().__init__(downstream_task=False)

    def evaluate_embeddings(self, algo, global_step, cfg, datasets, writer=None):
        raise NotImplementedError

    def evaluate_iterators(self, algo, global_step, cfg, iterators, writer=None):
        return {"dummy_iter_metric": 2.0}


class TestTaskDispatch:
    """Test that Task.evaluate dispatches correctly."""

    def test_embedding_task_dispatches_to_evaluate_embeddings(self):
        task = _DummyEmbeddingTask()
        result = task.evaluate(
            algo=None, global_step=0, cfg=None,
            embeddings_dataset={"train": {}, "val": {}},
        )
        assert result == {"dummy_emb_metric": 1.0}

    def test_iterator_task_dispatches_to_evaluate_iterators(self):
        task = _DummyIteratorTask()
        result = task.evaluate(
            algo=None, global_step=0, cfg=None,
            iterators={"val": []},
        )
        assert result == {"dummy_iter_metric": 2.0}

    def test_embedding_task_raises_without_embeddings(self):
        task = _DummyEmbeddingTask()
        with pytest.raises(ValueError, match="requires embeddings_dataset"):
            task.evaluate(algo=None, global_step=0, cfg=None)

    def test_iterator_task_raises_without_iterators(self):
        task = _DummyIteratorTask()
        with pytest.raises(ValueError, match="requires iterators"):
            task.evaluate(algo=None, global_step=0, cfg=None)


# ---------------------------------------------------------------------------
# task_utils tests
# ---------------------------------------------------------------------------


class TestTaskUtils:
    """Test pure numpy utility functions."""

    def test_regression_labels_for_class_basic(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        result = regression_labels_for_class(labels, 1)
        assert result is not None
        assert len(result) == 6
        # Before transition frame (index 2): linearly increasing
        assert result[0] == pytest.approx(0.0)
        # At and after transition: 1.0
        assert result[2] == pytest.approx(1.0)
        assert result[5] == pytest.approx(1.0)

    def test_regression_labels_for_missing_class(self):
        labels = np.array([0, 0, 1, 1])
        result = regression_labels_for_class(labels, 5)
        assert result is None

    def test_regression_labels_class_at_start(self):
        labels = np.array([2, 2, 2])
        result = regression_labels_for_class(labels, 2)
        assert result is not None
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])

    def test_get_regression_labels_shape(self):
        labels = np.array([0, 0, 1, 1, 2])
        result = get_regression_labels(labels, 3)
        assert result.shape == (5, 3)

    def test_get_targets_from_labels(self):
        all_labels = [np.array([0, 1, 2]), np.array([0, 0, 1])]
        result = get_targets_from_labels(all_labels, 3)
        assert len(result) == 2
        assert result[0].shape == (3, 3)
        assert result[1].shape == (3, 3)

    def test_unnormalize(self):
        preds = np.array([0.0, 0.5, 1.0])
        result = unnormalize(preds, seq_len=11)
        np.testing.assert_array_equal(result, [0, 5, 10])

    def test_unnormalize_clipping(self):
        preds = np.array([-0.1, 1.5])
        result = unnormalize(preds, seq_len=10)
        assert result[0] == 0
        assert result[1] == 9

    def test_softmax_uniform(self):
        w = np.array([1.0, 1.0, 1.0])
        result = softmax(w)
        np.testing.assert_array_almost_equal(result, [1 / 3, 1 / 3, 1 / 3])

    def test_softmax_temperature(self):
        w = np.array([1.0, 2.0])
        # High temperature -> more uniform
        high_t = softmax(w, t=100.0)
        low_t = softmax(w, t=0.1)
        # High temp should be more uniform
        assert abs(high_t[0] - high_t[1]) < abs(low_t[0] - low_t[1])

    def test_softmax_2d(self):
        w = np.array([[1.0, 2.0], [3.0, 1.0]])
        result = softmax(w)
        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result.sum(axis=-1), [1.0, 1.0])


# ---------------------------------------------------------------------------
# Kendall's Tau tests
# ---------------------------------------------------------------------------


class TestKendallsTau:
    """Test Kendall's Tau with synthetic embeddings."""

    def test_perfectly_ordered_embeddings(self):
        """Monotonically increasing 1-D embeddings should yield tau ~ 1.0."""
        rng = np.random.RandomState(42)
        embs1 = np.arange(20).reshape(-1, 1).astype(np.float32)
        embs2 = np.arange(20).reshape(-1, 1).astype(np.float32) + rng.randn(20, 1) * 0.01
        tau = _get_kendalls_tau([embs1, embs2], stride=1, distance="sqeuclidean")
        assert tau > 0.9, f"Expected tau > 0.9, got {tau}"

    def test_reversed_embeddings(self):
        """Reversed embeddings should yield negative tau."""
        embs1 = np.arange(20).reshape(-1, 1).astype(np.float32)
        embs2 = np.arange(19, -1, -1).reshape(-1, 1).astype(np.float32)
        tau = _get_kendalls_tau([embs1, embs2], stride=1, distance="sqeuclidean")
        assert tau < -0.5, f"Expected tau < -0.5, got {tau}"

    def test_single_sequence_returns_zero(self):
        embs = [np.arange(10).reshape(-1, 1).astype(np.float32)]
        tau = _get_kendalls_tau(embs, stride=1)
        assert tau == 0.0

    def test_kendalls_tau_task(self):
        """Test the KendallsTau task class end-to-end."""
        cfg = TCCConfig()
        task = KendallsTau()
        assert task.downstream_task is True

        embs_list = [
            np.arange(30).reshape(-1, 1).astype(np.float32),
            np.arange(30).reshape(-1, 1).astype(np.float32) + 0.01,
        ]
        datasets = {
            "val": {"embeddings_list": embs_list},
        }
        metrics = task.evaluate(
            algo=None, global_step=100, cfg=cfg,
            embeddings_dataset=datasets,
        )
        assert "kendalls_tau/val" in metrics
        assert metrics["kendalls_tau/val"] > 0.9


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassification:
    """Test linear classification with linearly separable data."""

    def _make_separable_data(self, n=100, d=10, seed=42):
        rng = np.random.RandomState(seed)
        # Two clusters well separated along dim 0
        embs_0 = rng.randn(n, d).astype(np.float32)
        embs_0[:, 0] -= 5.0
        embs_1 = rng.randn(n, d).astype(np.float32)
        embs_1[:, 0] += 5.0
        embs = np.concatenate([embs_0, embs_1], axis=0)
        labels = np.array([0] * n + [1] * n)
        return embs, labels

    def test_fit_linear_model(self):
        train_embs, train_labels = self._make_separable_data(n=50)
        val_embs, val_labels = self._make_separable_data(n=20, seed=99)
        acc = fit_linear_model(train_embs, train_labels, val_embs, val_labels)
        assert acc > 0.95

    def test_fit_svm_model(self):
        train_embs, train_labels = self._make_separable_data(n=50)
        val_embs, val_labels = self._make_separable_data(n=20, seed=99)
        acc = fit_svm_model(train_embs, train_labels, val_embs, val_labels)
        assert acc > 0.95

    def test_classification_task(self):
        cfg = TCCConfig()
        cfg.eval.classification_fractions = [1.0]
        task = Classification()

        train_embs, train_labels = self._make_separable_data(n=50)
        val_embs, val_labels = self._make_separable_data(n=20, seed=99)

        datasets = {
            "train": {"embeddings": train_embs, "labels": train_labels},
            "val": {"embeddings": val_embs, "labels": val_labels},
        }
        metrics = task.evaluate(
            algo=None, global_step=0, cfg=cfg,
            embeddings_dataset=datasets,
        )
        assert "classification/logistic_regression/frac_1.00" in metrics
        assert metrics["classification/logistic_regression/frac_1.00"] > 0.95
        assert "classification/svm/frac_1.00" in metrics


# ---------------------------------------------------------------------------
# EventCompletion tests
# ---------------------------------------------------------------------------


class TestEventCompletion:
    """Test event completion with simple regression data."""

    def test_vector_regression(self):
        X = np.array([[1], [2], [3], [4]], dtype=np.float32)
        y = np.array([[0.0, 0.0], [0.25, 0.5], [0.5, 1.0], [0.75, 1.0]])
        model = VectorRegression()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (4, 2)
        # Predictions should be close to targets for linear data
        assert np.mean(np.abs(preds - y)) < 0.2

    def test_event_completion_task(self):
        cfg = TCCConfig()
        task = EventCompletion()
        assert task.downstream_task is True

        rng = np.random.RandomState(42)
        d = 8
        # Training data: embeddings linearly correlated with event progress
        n_vids_train = 5
        train_embs_parts = []
        train_labels_list = []
        for _ in range(n_vids_train):
            seq_len = 20
            t = np.linspace(0, 1, seq_len).reshape(-1, 1)
            emb = np.hstack([t, rng.randn(seq_len, d - 1) * 0.1]).astype(np.float32)
            labels = (t.flatten() * 2).astype(int).clip(0, 2)
            train_embs_parts.append(emb)
            train_labels_list.append(labels)

        train_embs = np.concatenate(train_embs_parts, axis=0)

        n_vids_val = 3
        val_embs_parts = []
        val_labels_list = []
        for _ in range(n_vids_val):
            seq_len = 20
            t = np.linspace(0, 1, seq_len).reshape(-1, 1)
            emb = np.hstack([t, rng.randn(seq_len, d - 1) * 0.1]).astype(np.float32)
            labels = (t.flatten() * 2).astype(int).clip(0, 2)
            val_embs_parts.append(emb)
            val_labels_list.append(labels)

        val_embs = np.concatenate(val_embs_parts, axis=0)

        datasets = {
            "train": {
                "embeddings": train_embs,
                "labels_list": train_labels_list,
                "num_classes": 3,
            },
            "val": {
                "embeddings": val_embs,
                "labels_list": val_labels_list,
                "num_classes": 3,
            },
        }
        metrics = task.evaluate(
            algo=None, global_step=0, cfg=cfg,
            embeddings_dataset=datasets,
        )
        assert "event_completion/mae" in metrics
        assert metrics["event_completion/mae"] < 1.0  # Should be reasonable


# ---------------------------------------------------------------------------
# FewShotClassification tests
# ---------------------------------------------------------------------------


class TestFewShotClassification:
    """Test few-shot classification with easy separable data."""

    def test_few_shot_task(self):
        cfg = TCCConfig()
        cfg.eval.few_shot_num_labeled = [5]
        cfg.eval.few_shot_num_episodes = 10

        task = FewShotClassification()
        assert task.downstream_task is True

        rng = np.random.RandomState(42)
        d = 10
        n = 100
        # Well-separated clusters
        embs_0 = rng.randn(n, d).astype(np.float32) - 5.0
        embs_1 = rng.randn(n, d).astype(np.float32) + 5.0
        train_embs = np.concatenate([embs_0, embs_1])
        train_labels = np.array([0] * n + [1] * n)

        val_embs_0 = rng.randn(20, d).astype(np.float32) - 5.0
        val_embs_1 = rng.randn(20, d).astype(np.float32) + 5.0
        val_embs = np.concatenate([val_embs_0, val_embs_1])
        val_labels = np.array([0] * 20 + [1] * 20)

        datasets = {
            "train": {"embeddings": train_embs, "labels": train_labels},
            "val": {"embeddings": val_embs, "labels": val_labels},
        }

        metrics = task.evaluate(
            algo=None, global_step=0, cfg=cfg,
            embeddings_dataset=datasets,
        )
        assert "few_shot/k=5/mean_accuracy" in metrics
        assert metrics["few_shot/k=5/mean_accuracy"] > 0.8
        assert "few_shot/k=5/ci_95" in metrics


# ---------------------------------------------------------------------------
# AlgoLoss tests
# ---------------------------------------------------------------------------


class _FakeAlgo(nn.Module):
    """Minimal fake Algorithm for testing AlgoLoss."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, data, steps, seq_lens, training=False):
        b = data["frames"].shape[0]
        t = data["frames"].shape[1]
        return torch.randn(b, t, 8)

    def compute_loss(self, embs, steps, seq_lens, global_step, training=False,
                     frame_labels=None, seq_labels=None):
        return torch.tensor(0.5)


class TestAlgoLoss:
    """Test AlgoLoss returns a finite scalar."""

    def test_algo_loss_returns_finite(self):
        cfg = TCCConfig()
        cfg.eval.val_iters = 2
        cfg.eval.num_frames = 4

        task = AlgoLoss()
        assert task.downstream_task is False

        algo = _FakeAlgo()

        # Create fake batches
        fake_batches = []
        for _ in range(3):
            frames = torch.randn(2, 4, 3, 8, 8)
            frame_labels = torch.zeros(2, 4, dtype=torch.long)
            seq_labels = torch.zeros(2, dtype=torch.long)
            seq_lens = torch.tensor([4, 4], dtype=torch.long)
            names = ["vid1", "vid2"]
            fake_batches.append((frames, frame_labels, seq_labels, seq_lens, names))

        iterators = {"val": fake_batches}

        metrics = task.evaluate(
            algo=algo, global_step=0, cfg=cfg,
            iterators=iterators,
        )
        assert "algo_loss/val" in metrics
        assert np.isfinite(metrics["algo_loss/val"])
        assert metrics["algo_loss/val"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    """Test get_tasks factory."""

    def test_get_tasks_splits_correctly(self):
        iterator_tasks, embedding_tasks = get_tasks([
            "algo_loss",
            "classification",
            "kendalls_tau",
            "event_completion",
            "few_shot_classification",
        ])
        assert "algo_loss" in iterator_tasks
        assert "algo_loss" not in embedding_tasks

        assert "classification" in embedding_tasks
        assert "kendalls_tau" in embedding_tasks
        assert "event_completion" in embedding_tasks
        assert "few_shot_classification" in embedding_tasks

    def test_get_tasks_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_tasks(["nonexistent_task"])

    def test_get_tasks_empty(self):
        iterator_tasks, embedding_tasks = get_tasks([])
        assert len(iterator_tasks) == 0
        assert len(embedding_tasks) == 0

    def test_get_tasks_single_iterator(self):
        iterator_tasks, embedding_tasks = get_tasks(["algo_loss"])
        assert len(iterator_tasks) == 1
        assert len(embedding_tasks) == 0

    def test_get_tasks_single_embedding(self):
        iterator_tasks, embedding_tasks = get_tasks(["classification"])
        assert len(iterator_tasks) == 0
        assert len(embedding_tasks) == 1
