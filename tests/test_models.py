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

"""Tests for TCC PyTorch models."""

import pytest
import torch

from tcc.models import BaseModel, ConvEmbedder, ConvGRUEmbedder, Classifier, get_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
B, T, H, W = 2, 4, 224, 224


@pytest.fixture()
def base_model():
    return BaseModel()


@pytest.fixture()
def conv_embedder():
    return ConvEmbedder()


# ---------------------------------------------------------------------------
# BaseModel
# ---------------------------------------------------------------------------

class TestBaseModel:
    def test_forward_shape(self, base_model):
        """Output should be [B, T, 1024, 14, 14] for layer3 with 224x224."""
        x = torch.randn(B, T, 3, H, W)
        with torch.no_grad():
            out = base_model(x)
        assert out.shape == (B, T, 1024, 14, 14)

    def test_frozen_mode_no_grad(self):
        model = BaseModel(cfg={"train_base": "frozen"})
        for p in model.backbone.parameters():
            assert not p.requires_grad

    def test_only_bn_mode(self):
        model = BaseModel(cfg={"train_base": "only_bn"})
        # Collect BN parameter ids via isinstance (matches model logic)
        bn_param_ids = set()
        for m in model.backbone.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                for p in m.parameters():
                    bn_param_ids.add(id(p))
        assert len(bn_param_ids) > 0, "Should have BN parameters"
        for name, p in model.backbone.named_parameters():
            if id(p) in bn_param_ids:
                assert p.requires_grad, f"BN param {name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_train_all_mode(self):
        model = BaseModel(cfg={"train_base": "train_all"})
        for p in model.backbone.parameters():
            assert p.requires_grad


# ---------------------------------------------------------------------------
# ConvEmbedder
# ---------------------------------------------------------------------------

class TestConvEmbedder:
    def test_output_shape_128d(self, conv_embedder):
        """Embedding output should be 128-dimensional."""
        num_frames = 4
        num_context = 2  # default num_steps
        T_total = num_frames * num_context
        # Simulate CNN features: [B, T_total, 1024, 14, 14]
        x = torch.randn(B, T_total, 1024, 14, 14)
        with torch.no_grad():
            out = conv_embedder(x, num_frames=num_frames)
        assert out.shape == (B * num_frames, 128)

    def test_avg_pool_method(self):
        emb = ConvEmbedder(cfg={"conv_embedder_flatten_method": "avg_pool"})
        x = torch.randn(1, 2, 1024, 14, 14)
        with torch.no_grad():
            out = emb(x, num_frames=1)
        assert out.shape[-1] == 128

    def test_l2_normalize(self):
        emb = ConvEmbedder(cfg={"conv_embedder_l2_normalize": True})
        x = torch.randn(B, 2, 1024, 14, 14)
        with torch.no_grad():
            out = emb(x, num_frames=1)
        norms = torch.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            "L2-normalized embeddings should have unit norm"
        )


# ---------------------------------------------------------------------------
# ConvGRUEmbedder
# ---------------------------------------------------------------------------

class TestConvGRUEmbedder:
    def test_output_shape(self):
        cfg = {"num_steps": 1}
        emb = ConvGRUEmbedder(cfg=cfg)
        x = torch.randn(B, T, 1024, 14, 14)
        with torch.no_grad():
            out = emb(x, num_frames=T)
        # Default GRU hidden = 128
        assert out.shape == (B * T, 128)

    def test_rejects_context_frames(self):
        with pytest.raises(ValueError, match="context frames"):
            ConvGRUEmbedder(cfg={"num_steps": 2})


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_output_shape(self):
        fc_layers = [(128, True), (64, True), (2, False)]
        clf = Classifier(fc_layers=fc_layers, dropout_rate=0.0, in_features=128)
        x = torch.randn(B, 128)
        out = clf(x)
        assert out.shape == (B, 2)


# ---------------------------------------------------------------------------
# get_model factory
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_returns_dict_with_cnn_and_emb(self):
        model = get_model()
        assert isinstance(model, dict)
        assert "cnn" in model
        assert "emb" in model
        assert isinstance(model["cnn"], BaseModel)
        assert isinstance(model["emb"], ConvEmbedder)

    def test_convgru_embedder_type(self):
        cfg = {"embedder_type": "convgru", "num_steps": 1}
        model = get_model(cfg=cfg)
        assert isinstance(model["emb"], ConvGRUEmbedder)

    def test_invalid_embedder_type(self):
        with pytest.raises(ValueError, match="Unsupported embedder_type"):
            get_model(cfg={"embedder_type": "transformer"})
