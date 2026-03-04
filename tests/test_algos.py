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

"""Tests for tcc.algos — algorithm implementations."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tcc.algos import (
    Alignment,
    AlignmentSaLTCN,
    Classification,
    SAL,
    TCN,
    get_algo,
)
from tcc.algos.sal import get_shuffled_indices_and_labels, sample_batch
from tcc.config import get_default_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T, D = 2, 4, 128


def _make_cfg():
    """Return a config suitable for small CPU tests."""
    cfg = get_default_config()
    cfg.train.num_frames = T
    cfg.eval.num_frames = T
    cfg.train.batch_size = B
    return cfg


def _make_dummy_model(cfg=None):
    """Build a tiny surrogate model dict (no ResNet) for fast CPU tests."""
    if cfg is None:
        cfg = _make_cfg()

    emb_size = cfg.model.conv_embedder.embedding_size

    class DummyCNN(nn.Module):
        """Identity-like CNN that outputs [B, T, C, 1, 1]."""
        def forward(self, x):
            # x: [B, T, 3, H, W]
            b, t = x.shape[:2]
            return torch.randn(b, t, 64, 1, 1)

    class DummyEmb(nn.Module):
        """Maps [B, T, C, 1, 1] -> [B*num_frames, emb_size]."""
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, emb_size)

        def forward(self, x, num_frames):
            b = x.shape[0]
            # Flatten spatial dims
            x = x.squeeze(-1).squeeze(-1)  # [B, T, 64]
            # Take num_frames frames (subsample or pad)
            t = x.shape[1]
            if t > num_frames:
                x = x[:, :num_frames, :]
            elif t < num_frames:
                pad = torch.zeros(b, num_frames - t, 64, device=x.device)
                x = torch.cat([x, pad], dim=1)
            x = x.reshape(b * num_frames, 64)
            return self.linear(x)

    return {"cnn": DummyCNN(), "emb": DummyEmb()}


def _make_data(num_frames=T, channels=3, h=4, w=4):
    """Create a dummy data dict."""
    frames = torch.randn(B, num_frames, channels, h, w)
    return {"frames": frames}


def _make_steps(num_frames=T):
    return torch.arange(num_frames).unsqueeze(0).expand(B, -1)


def _make_seq_lens(num_frames=T):
    return torch.full((B,), num_frames, dtype=torch.long)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_forward_shape(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = Alignment(cfg=cfg, model=model)
        algo.eval()

        data = _make_data()
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        embs = algo(data, steps, seq_lens, training=False)
        assert embs.shape == (B, T, D)

    def test_compute_loss_scalar(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = Alignment(cfg=cfg, model=model)

        embs = torch.randn(B, T, D, requires_grad=True)
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        loss = algo.compute_loss(embs, steps, seq_lens, global_step=0, training=True)
        assert loss.dim() == 0
        assert loss.requires_grad


# ---------------------------------------------------------------------------
# SAL
# ---------------------------------------------------------------------------


class TestSAL:
    def test_forward_shape(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = SAL(cfg=cfg, model=model)
        algo.eval()

        data = _make_data()
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        embs = algo(data, steps, seq_lens, training=False)
        assert embs.shape == (B, T, D)

    def test_compute_loss_scalar(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = SAL(cfg=cfg, model=model)

        embs = torch.randn(B, T, D, requires_grad=True)
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        loss = algo.compute_loss(embs, steps, seq_lens, global_step=0, training=True)
        assert loss.dim() == 0

    def test_shuffled_indices_shape(self):
        indices, labels = get_shuffled_indices_and_labels(
            batch_size=B, num_samples=8, shuffle_fraction=0.75, num_steps=T,
        )
        assert indices.shape == (B * 8, 3)
        assert labels.shape == (B * 8, 2)

    def test_sample_batch_shape(self):
        cfg = _make_cfg()
        embs = torch.randn(B, T, D)
        concat, labels = sample_batch(embs, B, T, cfg.sal)
        assert concat.shape == (B * cfg.sal.num_samples, 3 * D)
        assert labels.shape == (B * cfg.sal.num_samples, 2)

    def test_algo_params_not_empty(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = SAL(cfg=cfg, model=model)
        assert len(algo.get_algo_params()) > 0


# ---------------------------------------------------------------------------
# TCN
# ---------------------------------------------------------------------------


class TestTCN:
    def test_forward_shape(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = TCN(cfg=cfg, model=model)
        algo.eval()

        # TCN forward expects doubled frames
        data = _make_data(num_frames=2 * T)
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        embs = algo(data, steps, seq_lens, training=False)
        assert embs.shape == (B, 2 * T, D)

    def test_compute_loss_scalar(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = TCN(cfg=cfg, model=model)

        embs = torch.randn(B, 2 * T, D, requires_grad=True)
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        loss = algo.compute_loss(embs, steps, seq_lens, global_step=0, training=True)
        assert loss.dim() == 0

    def test_npairs_loss(self):
        labels = torch.arange(4)
        anchors = torch.randn(4, D, requires_grad=True)
        positives = torch.randn(4, D, requires_grad=True)
        loss = TCN._npairs_loss(labels, anchors, positives, reg_lambda=0.002)
        assert loss.dim() == 0
        loss.backward()
        assert anchors.grad is not None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_forward_shape(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = Classification(cfg=cfg, model=model, num_classes=5)
        algo.eval()

        data = _make_data()
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        embs = algo(data, steps, seq_lens, training=False)
        assert embs.shape == (B, T, D)

    def test_compute_loss_scalar(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        num_classes = 5
        algo = Classification(cfg=cfg, model=model, num_classes=num_classes)

        embs = torch.randn(B, T, D, requires_grad=True)
        steps = _make_steps()
        seq_lens = _make_seq_lens()
        frame_labels = torch.randint(0, num_classes, (B, T))

        loss = algo.compute_loss(
            embs, steps, seq_lens, global_step=0, training=True,
            frame_labels=frame_labels,
        )
        assert loss.dim() == 0

    def test_missing_labels_raises(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = Classification(cfg=cfg, model=model, num_classes=5)

        embs = torch.randn(B, T, D)
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        with pytest.raises(ValueError, match="frame_labels"):
            algo.compute_loss(
                embs, steps, seq_lens, global_step=0, training=True,
            )


# ---------------------------------------------------------------------------
# AlignmentSaLTCN
# ---------------------------------------------------------------------------


class TestAlignmentSaLTCN:
    def test_forward_shape(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = AlignmentSaLTCN(cfg=cfg, model=model)
        algo.eval()

        data = _make_data(num_frames=2 * T)
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        embs = algo(data, steps, seq_lens, training=False)
        assert embs.shape == (B, 2 * T, D)

    def test_compute_loss_scalar(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = AlignmentSaLTCN(cfg=cfg, model=model)

        embs = torch.randn(B, 2 * T, D, requires_grad=True)
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        loss = algo.compute_loss(embs, steps, seq_lens, global_step=0, training=True)
        assert loss.dim() == 0

    def test_weight_validation_negative(self):
        cfg = _make_cfg()
        cfg.alignment_sal_tcn.alignment_loss_weight = -0.1
        cfg.alignment_sal_tcn.sal_loss_weight = 0.5
        model = _make_dummy_model(cfg)

        with pytest.raises(ValueError, match="non-negative"):
            AlignmentSaLTCN(cfg=cfg, model=model)

    def test_weight_validation_sum_exceeds_one(self):
        cfg = _make_cfg()
        cfg.alignment_sal_tcn.alignment_loss_weight = 0.7
        cfg.alignment_sal_tcn.sal_loss_weight = 0.5
        model = _make_dummy_model(cfg)

        with pytest.raises(ValueError, match="<= 1.0"):
            AlignmentSaLTCN(cfg=cfg, model=model)

    def test_valid_custom_weights(self):
        cfg = _make_cfg()
        cfg.alignment_sal_tcn.alignment_loss_weight = 0.5
        cfg.alignment_sal_tcn.sal_loss_weight = 0.3
        model = _make_dummy_model(cfg)

        algo = AlignmentSaLTCN(cfg=cfg, model=model)
        assert abs(algo.tcn_weight - 0.2) < 1e-6


# ---------------------------------------------------------------------------
# Registry / Factory
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_get_algo_alignment(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = get_algo("alignment", cfg=cfg, model=model)
        assert isinstance(algo, Alignment)

    def test_get_algo_sal(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = get_algo("sal", cfg=cfg, model=model)
        assert isinstance(algo, SAL)

    def test_get_algo_tcn(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = get_algo("tcn", cfg=cfg, model=model)
        assert isinstance(algo, TCN)

    def test_get_algo_classification(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = get_algo("classification", cfg=cfg, model=model, num_classes=5)
        assert isinstance(algo, Classification)

    def test_get_algo_alignment_sal_tcn(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = get_algo("alignment_sal_tcn", cfg=cfg, model=model)
        assert isinstance(algo, AlignmentSaLTCN)

    def test_invalid_algo_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_algo("nonexistent")


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_alignment_gradient_flow(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = Alignment(cfg=cfg, model=model)
        algo.train()

        embs = torch.randn(B, T, D, requires_grad=True)
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        loss = algo.compute_loss(embs, steps, seq_lens, global_step=0, training=True)
        loss.backward()
        assert embs.grad is not None
        assert torch.any(embs.grad != 0)

    def test_sal_gradient_flow(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = SAL(cfg=cfg, model=model)
        algo.train()

        embs = torch.randn(B, T, D, requires_grad=True)
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        loss = algo.compute_loss(embs, steps, seq_lens, global_step=0, training=True)
        loss.backward()
        assert embs.grad is not None

    def test_tcn_gradient_flow(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = TCN(cfg=cfg, model=model)
        algo.train()

        embs = torch.randn(B, 2 * T, D, requires_grad=True)
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        loss = algo.compute_loss(embs, steps, seq_lens, global_step=0, training=True)
        loss.backward()
        assert embs.grad is not None

    def test_classification_gradient_flow(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        num_classes = 5
        algo = Classification(cfg=cfg, model=model, num_classes=num_classes)
        algo.train()

        embs = torch.randn(B, T, D, requires_grad=True)
        steps = _make_steps()
        seq_lens = _make_seq_lens()
        frame_labels = torch.randint(0, num_classes, (B, T))

        loss = algo.compute_loss(
            embs, steps, seq_lens, global_step=0, training=True,
            frame_labels=frame_labels,
        )
        loss.backward()
        assert embs.grad is not None

    def test_alignment_sal_tcn_gradient_flow(self):
        cfg = _make_cfg()
        model = _make_dummy_model(cfg)
        algo = AlignmentSaLTCN(cfg=cfg, model=model)
        algo.train()

        embs = torch.randn(B, 2 * T, D, requires_grad=True)
        steps = _make_steps()
        seq_lens = _make_seq_lens()

        loss = algo.compute_loss(embs, steps, seq_lens, global_step=0, training=True)
        loss.backward()
        assert embs.grad is not None
