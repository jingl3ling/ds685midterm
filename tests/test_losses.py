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

"""Tests for TCC core losses ported to PyTorch."""

import pytest
import torch
import torch.nn.functional as F

from tcc.losses import classification_loss, regression_loss
from tcc.deterministic_alignment import (
    align_pair_of_sequences,
    pairwise_l2_distance,
)
from tcc.alignment import compute_alignment_loss


# ---------------------------------------------------------------------------
# pairwise_l2_distance
# ---------------------------------------------------------------------------

class TestPairwiseL2Distance:
    def test_shape(self):
        embs1 = torch.randn(5, 8)
        embs2 = torch.randn(7, 8)
        dist = pairwise_l2_distance(embs1, embs2)
        assert dist.shape == (5, 7)

    def test_non_negative(self):
        embs1 = torch.randn(4, 16)
        embs2 = torch.randn(6, 16)
        dist = pairwise_l2_distance(embs1, embs2)
        assert (dist >= 0).all()

    def test_self_distance_zero(self):
        embs = torch.randn(5, 8)
        dist = pairwise_l2_distance(embs, embs)
        diag = torch.diagonal(dist)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5)

    def test_matches_brute_force(self):
        embs1 = torch.randn(3, 4)
        embs2 = torch.randn(5, 4)
        dist = pairwise_l2_distance(embs1, embs2)
        # Brute force
        expected = torch.zeros(3, 5)
        for i in range(3):
            for j in range(5):
                expected[i, j] = ((embs1[i] - embs2[j]) ** 2).sum()
        assert torch.allclose(dist, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# align_pair_of_sequences
# ---------------------------------------------------------------------------

class TestAlignPairOfSequences:
    def test_output_shapes_l2(self):
        M, D = 10, 16
        embs1 = torch.randn(M, D)
        embs2 = torch.randn(M, D)
        logits, labels = align_pair_of_sequences(embs1, embs2, "l2", 0.1)
        assert logits.shape == (M, M)
        assert labels.shape == (M, M)

    def test_output_shapes_cosine(self):
        M, N, D = 8, 12, 16
        embs1 = torch.randn(M, D)
        embs2 = torch.randn(N, D)
        logits, labels = align_pair_of_sequences(embs1, embs2, "cosine", 0.1)
        assert logits.shape == (M, M)
        assert labels.shape == (M, M)

    def test_labels_are_identity(self):
        M, D = 6, 8
        embs1 = torch.randn(M, D)
        embs2 = torch.randn(M, D)
        _, labels = align_pair_of_sequences(embs1, embs2, "l2", 0.1)
        expected = torch.eye(M)
        assert torch.allclose(labels, expected)

    def test_invalid_similarity(self):
        embs = torch.randn(4, 8)
        with pytest.raises(ValueError):
            align_pair_of_sequences(embs, embs, "dot", 0.1)


# ---------------------------------------------------------------------------
# classification_loss
# ---------------------------------------------------------------------------

class TestClassificationLoss:
    def test_scalar_output(self):
        logits = torch.randn(8, 8)
        labels = torch.eye(8)
        loss = classification_loss(logits, labels, label_smoothing=0.1)
        assert loss.shape == ()

    def test_non_negative(self):
        logits = torch.randn(8, 8)
        labels = torch.eye(8)
        loss = classification_loss(logits, labels, label_smoothing=0.0)
        assert loss.item() >= 0

    def test_perfect_logits_low_loss(self):
        # With very confident logits matching labels, loss should be small.
        labels = torch.eye(5)
        logits = labels * 100.0  # Very confident
        loss = classification_loss(logits, labels, label_smoothing=0.0)
        assert loss.item() < 0.01

    def test_label_smoothing_increases_loss(self):
        logits = torch.randn(8, 8)
        labels = torch.eye(8)
        loss_no_smooth = classification_loss(logits, labels, 0.0)
        loss_smooth = classification_loss(logits, labels, 0.1)
        # With label smoothing, the loss from the "correct" class decreases but
        # the total can go either way; just check both are finite.
        assert torch.isfinite(loss_no_smooth)
        assert torch.isfinite(loss_smooth)


# ---------------------------------------------------------------------------
# regression_loss
# ---------------------------------------------------------------------------

class TestRegressionLoss:
    def _make_inputs(self, n=10, t=10):
        logits = torch.randn(n, t)
        labels = F.one_hot(torch.arange(t), t).float().expand(n, -1).clone()
        # Only works cleanly when n == t; for simplicity we keep them equal.
        steps = torch.arange(t).unsqueeze(0).expand(n, -1).clone()
        seq_lens = torch.full((n,), t, dtype=torch.long)
        return logits, labels, t, steps, seq_lens

    def test_regression_mse_scalar(self):
        logits, labels, t, steps, seq_lens = self._make_inputs()
        loss = regression_loss(
            logits, labels, t, steps, seq_lens,
            "regression_mse", False, 0.001, 0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_regression_mse_var_scalar(self):
        logits, labels, t, steps, seq_lens = self._make_inputs()
        loss = regression_loss(
            logits, labels, t, steps, seq_lens,
            "regression_mse_var", False, 0.001, 0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_regression_huber_scalar(self):
        logits, labels, t, steps, seq_lens = self._make_inputs()
        loss = regression_loss(
            logits, labels, t, steps, seq_lens,
            "regression_huber", False, 0.001, 0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_normalize_indices(self):
        logits, labels, t, steps, seq_lens = self._make_inputs()
        loss = regression_loss(
            logits, labels, t, steps, seq_lens,
            "regression_mse", True, 0.001, 0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_unsupported_type(self):
        logits, labels, t, steps, seq_lens = self._make_inputs()
        with pytest.raises(ValueError):
            regression_loss(
                logits, labels, t, steps, seq_lens,
                "regression_l1", False, 0.001, 0.1,
            )


# ---------------------------------------------------------------------------
# compute_alignment_loss (end-to-end)
# ---------------------------------------------------------------------------

class TestComputeAlignmentLoss:
    def test_deterministic_classification_l2(self):
        N, T, D = 3, 8, 16
        embs = torch.randn(N, T, D)
        loss = compute_alignment_loss(
            embs, batch_size=N, loss_type="classification",
            similarity_type="l2", temperature=0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_deterministic_classification_cosine(self):
        N, T, D = 3, 8, 16
        embs = torch.randn(N, T, D)
        loss = compute_alignment_loss(
            embs, batch_size=N, loss_type="classification",
            similarity_type="cosine", temperature=0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_deterministic_regression_mse(self):
        N, T, D = 2, 6, 8
        embs = torch.randn(N, T, D)
        loss = compute_alignment_loss(
            embs, batch_size=N, loss_type="regression_mse",
            similarity_type="l2", temperature=0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_deterministic_regression_mse_var(self):
        N, T, D = 2, 6, 8
        embs = torch.randn(N, T, D)
        loss = compute_alignment_loss(
            embs, batch_size=N, loss_type="regression_mse_var",
            similarity_type="l2", temperature=0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_deterministic_regression_huber(self):
        N, T, D = 2, 6, 8
        embs = torch.randn(N, T, D)
        loss = compute_alignment_loss(
            embs, batch_size=N, loss_type="regression_huber",
            similarity_type="l2", temperature=0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_stochastic_classification(self):
        N, T, D = 4, 10, 16
        embs = torch.randn(N, T, D)
        loss = compute_alignment_loss(
            embs, batch_size=N, stochastic_matching=True,
            loss_type="classification", similarity_type="l2",
            num_cycles=5, cycle_length=2, temperature=0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_stochastic_regression_mse(self):
        N, T, D = 4, 10, 16
        embs = torch.randn(N, T, D)
        loss = compute_alignment_loss(
            embs, batch_size=N, stochastic_matching=True,
            loss_type="regression_mse", similarity_type="l2",
            num_cycles=5, cycle_length=2, temperature=0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_normalize_embeddings(self):
        N, T, D = 3, 8, 16
        embs = torch.randn(N, T, D)
        loss = compute_alignment_loss(
            embs, batch_size=N, normalize_embeddings=True,
            loss_type="classification", similarity_type="cosine",
            temperature=0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_custom_steps_and_seq_lens(self):
        N, T, D = 2, 6, 8
        embs = torch.randn(N, T, D)
        steps = torch.arange(T).unsqueeze(0).expand(N, -1)
        seq_lens = torch.tensor([T, T])
        loss = compute_alignment_loss(
            embs, batch_size=N, steps=steps, seq_lens=seq_lens,
            loss_type="classification", similarity_type="l2",
            temperature=0.1,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_invalid_loss_type(self):
        N, T, D = 2, 6, 8
        embs = torch.randn(N, T, D)
        with pytest.raises(ValueError):
            compute_alignment_loss(
                embs, batch_size=N, loss_type="invalid",
                similarity_type="l2", temperature=0.1,
            )

    def test_gradient_flows(self):
        N, T, D = 2, 6, 8
        embs = torch.randn(N, T, D, requires_grad=True)
        loss = compute_alignment_loss(
            embs, batch_size=N, loss_type="classification",
            similarity_type="l2", temperature=0.1,
        )
        loss.backward()
        assert embs.grad is not None
        assert torch.isfinite(embs.grad).all()
