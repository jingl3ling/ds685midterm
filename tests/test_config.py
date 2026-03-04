"""Tests for the TCC configuration system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tcc.config import (
    AlignmentConfig,
    BaseModelConfig,
    ConvEmbedderConfig,
    DataConfig,
    EvalConfig,
    ModelConfig,
    OptimizerConfig,
    TCCConfig,
    TrainConfig,
    config_to_dict,
    dict_to_config,
    get_default_config,
    load_config,
    save_config,
)


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------


class TestGetDefaultConfig:
    """Verify that get_default_config() returns correct defaults."""

    def test_returns_tcc_config(self):
        cfg = get_default_config()
        assert isinstance(cfg, TCCConfig)

    def test_train_defaults(self):
        cfg = get_default_config()
        assert cfg.train.max_iters == 150000
        assert cfg.train.batch_size == 2
        assert cfg.train.num_frames == 20
        assert cfg.train.visualize_interval == 200

    def test_eval_defaults(self):
        cfg = get_default_config()
        assert cfg.eval.batch_size == 2
        assert cfg.eval.num_frames == 20
        assert cfg.eval.val_iters == 20
        assert cfg.eval.kendalls_tau_stride == 5
        assert cfg.eval.classification_fractions == [0.1, 0.5, 1.0]
        assert "algo_loss" in cfg.eval.tasks

    def test_model_defaults(self):
        cfg = get_default_config()
        assert cfg.model.embedder_type == "conv"
        assert cfg.model.base_model.network == "resnet50"
        assert cfg.model.base_model.layer == "conv4_block3_out"
        assert cfg.model.base_model.train_base == "only_bn"
        assert cfg.model.conv_embedder.embedding_size == 128
        assert cfg.model.conv_embedder.l2_normalize is False
        assert cfg.model.l2_reg_weight == pytest.approx(1e-5)

    def test_alignment_defaults(self):
        cfg = get_default_config()
        assert cfg.alignment.loss_type == "regression_mse_var"
        assert cfg.alignment.similarity_type == "l2"
        assert cfg.alignment.temperature == pytest.approx(0.1)
        assert cfg.alignment.cycle_length == 2
        assert cfg.alignment.stochastic_matching is False
        assert cfg.alignment.normalize_indices is True

    def test_optimizer_defaults(self):
        cfg = get_default_config()
        assert cfg.optimizer.type == "adam"
        assert cfg.optimizer.lr.initial_lr == pytest.approx(0.0001)
        assert cfg.optimizer.lr.decay_type == "fixed"

    def test_data_defaults(self):
        cfg = get_default_config()
        assert cfg.data.sampling_strategy == "offset_uniform"
        assert cfg.data.stride == 16
        assert cfg.data.num_steps == 2
        assert cfg.data.frame_stride == 15
        assert cfg.data.image_size == 224

    def test_top_level_defaults(self):
        cfg = get_default_config()
        assert cfg.training_algo == "alignment"
        assert cfg.datasets == ["pouring"]
        assert cfg.logdir == "/tmp/alignment_logs/"

    def test_sal_defaults(self):
        cfg = get_default_config()
        assert cfg.sal.shuffle_fraction == 0.75
        assert cfg.sal.num_samples == 8

    def test_tcn_defaults(self):
        cfg = get_default_config()
        assert cfg.tcn.positive_window == 5
        assert cfg.tcn.reg_lambda == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# Round-trip: config -> dict -> config
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """config_to_dict / dict_to_config preserve values."""

    def test_dict_roundtrip(self):
        cfg = get_default_config()
        d = config_to_dict(cfg)
        cfg2 = dict_to_config(d)
        assert config_to_dict(cfg) == config_to_dict(cfg2)

    def test_save_load_roundtrip(self):
        cfg = get_default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_config.yaml")
            save_config(cfg, path)
            cfg2 = load_config(path)
        assert config_to_dict(cfg) == config_to_dict(cfg2)


# ---------------------------------------------------------------------------
# Partial / nested overrides
# ---------------------------------------------------------------------------


class TestOverrides:
    """Loading a partial YAML merges correctly with defaults."""

    def test_partial_override(self):
        """Only override train.max_iters; everything else stays default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "partial.yaml"
            path.write_text("train:\n  max_iters: 42\n")
            cfg = load_config(str(path))

        assert cfg.train.max_iters == 42
        # The rest should be untouched.
        assert cfg.train.batch_size == 2
        assert cfg.eval.val_iters == 20
        assert cfg.model.conv_embedder.embedding_size == 128

    def test_nested_model_override(self):
        """Override a deeply nested model field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested.yaml"
            path.write_text(
                "model:\n  base_model:\n    network: vggm\n"
                "  l2_reg_weight: 0.001\n"
            )
            cfg = load_config(str(path))

        assert cfg.model.base_model.network == "vggm"
        assert cfg.model.l2_reg_weight == pytest.approx(0.001)
        # Other model defaults remain.
        assert cfg.model.embedder_type == "conv"
        assert cfg.model.base_model.layer == "conv4_block3_out"

    def test_optimizer_lr_override(self):
        """Override nested optimizer.lr fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "opt.yaml"
            path.write_text(
                "optimizer:\n  type: sgd\n  lr:\n    initial_lr: 0.01\n"
            )
            cfg = load_config(str(path))

        assert cfg.optimizer.type == "sgd"
        assert cfg.optimizer.lr.initial_lr == pytest.approx(0.01)
        assert cfg.optimizer.lr.decay_type == "fixed"

    def test_list_override_replaces(self):
        """Lists are replaced, not merged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "list.yaml"
            path.write_text("datasets:\n- baseball_pitch\n- golf_swing\n")
            cfg = load_config(str(path))

        assert cfg.datasets == ["baseball_pitch", "golf_swing"]


# ---------------------------------------------------------------------------
# Loading bundled YAML files
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestBundledConfigs:
    """Validate configs shipped with the repo."""

    def test_load_default_yaml(self):
        path = _REPO_ROOT / "configs" / "default.yaml"
        if not path.exists():
            pytest.skip("configs/default.yaml not found")
        cfg = load_config(str(path))
        assert isinstance(cfg, TCCConfig)
        assert cfg.train.max_iters == 150000

    def test_load_demo_yaml(self):
        path = _REPO_ROOT / "configs" / "demo.yaml"
        if not path.exists():
            pytest.skip("configs/demo.yaml not found")
        cfg = load_config(str(path))
        assert isinstance(cfg, TCCConfig)
        # Demo overrides
        assert cfg.train.max_iters == 10
        assert cfg.train.num_frames == 5
        assert cfg.data.image_size == 112
        # Defaults preserved
        assert cfg.alignment.loss_type == "regression_mse_var"
        assert cfg.model.conv_embedder.embedding_size == 128
