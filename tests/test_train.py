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

"""Tests for tcc.train -- training utilities and loop."""

from __future__ import annotations

import math
import os
import tempfile
from unittest import mock

import pytest
import torch
import torch.nn as nn

from tcc.algos.algorithm import Algorithm
from tcc.config import TCCConfig, get_default_config, load_config
from tcc.train import (
    _WarmupScheduler,
    _find_latest_checkpoint,
    get_lr_scheduler,
    get_optimizer,
    load_checkpoint,
    main,
    maybe_restore_checkpoint,
    save_checkpoint,
    train,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_cfg() -> TCCConfig:
    """Return a small config suitable for fast CPU tests."""
    cfg = get_default_config()
    cfg.train.max_iters = 2
    cfg.train.batch_size = 1
    cfg.train.num_frames = 4
    cfg.eval.num_frames = 4
    cfg.logging.report_interval = 1
    cfg.checkpoint.save_interval = 1
    cfg.model.l2_reg_weight = 0.0
    cfg.model.base_model.train_base = "frozen"
    return cfg


class _DummyAlgorithm(Algorithm):
    """Minimal concrete algorithm for testing (avoids loading ResNet)."""

    def __init__(self, cfg=None):
        # Bypass Algorithm.__init__ which calls get_model()
        nn.Module.__init__(self)
        self.cfg = cfg if cfg is not None else get_default_config()
        self.linear = nn.Linear(8, 4)
        # Provide cnn/emb stubs so get_trainable_params works
        self.cnn = nn.Linear(1, 1)
        self.emb = nn.Linear(1, 1)

    def forward(self, data, steps, seq_lens, training=True):
        frames = data["frames"]
        B = frames.shape[0]
        return self.linear(torch.randn(B, 4, 8, device=frames.device))

    def compute_loss(self, embs, steps, seq_lens, global_step, training,
                     frame_labels=None, seq_labels=None):
        return embs.sum().abs() + 1.0

    def get_trainable_params(self):
        return list(self.linear.parameters())


# ---------------------------------------------------------------------------
# Test: get_optimizer
# ---------------------------------------------------------------------------


class TestGetOptimizer:
    def test_adam(self):
        cfg = _tiny_cfg()
        cfg.optimizer.type = "adam"
        cfg.optimizer.lr.initial_lr = 0.001
        algo = _DummyAlgorithm(cfg)
        opt = get_optimizer(algo, cfg)
        assert isinstance(opt, torch.optim.Adam)
        assert opt.defaults["lr"] == 0.001

    def test_sgd(self):
        cfg = _tiny_cfg()
        cfg.optimizer.type = "sgd"
        cfg.optimizer.lr.initial_lr = 0.01
        algo = _DummyAlgorithm(cfg)
        opt = get_optimizer(algo, cfg)
        assert isinstance(opt, torch.optim.SGD)
        assert opt.defaults["lr"] == 0.01
        assert opt.defaults["momentum"] == 0.9

    def test_unknown_raises(self):
        cfg = _tiny_cfg()
        cfg.optimizer.type = "rmsprop"
        algo = _DummyAlgorithm(cfg)
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            get_optimizer(algo, cfg)


# ---------------------------------------------------------------------------
# Test: get_lr_scheduler
# ---------------------------------------------------------------------------


class TestGetLrScheduler:
    def _make_opt(self, lr=0.01):
        model = nn.Linear(4, 2)
        return torch.optim.Adam(model.parameters(), lr=lr)

    def test_fixed_returns_none(self):
        cfg = _tiny_cfg()
        cfg.optimizer.lr.decay_type = "fixed"
        opt = self._make_opt()
        assert get_lr_scheduler(opt, cfg) is None

    def test_exp_decay(self):
        cfg = _tiny_cfg()
        cfg.optimizer.lr.decay_type = "exp_decay"
        cfg.optimizer.lr.exp_decay_rate = 0.96
        opt = self._make_opt()
        sched = get_lr_scheduler(opt, cfg)
        assert isinstance(sched, torch.optim.lr_scheduler.ExponentialLR)

    def test_manual(self):
        cfg = _tiny_cfg()
        cfg.optimizer.lr.decay_type = "manual"
        cfg.optimizer.lr.manual_lr_step_boundaries = [5, 10]
        cfg.optimizer.lr.manual_lr_decay_rate = 0.1
        opt = self._make_opt()
        sched = get_lr_scheduler(opt, cfg)
        assert isinstance(sched, torch.optim.lr_scheduler.MultiStepLR)

    def test_poly(self):
        cfg = _tiny_cfg()
        cfg.optimizer.lr.decay_type = "poly"
        opt = self._make_opt()
        sched = get_lr_scheduler(opt, cfg)
        assert isinstance(sched, torch.optim.lr_scheduler.PolynomialLR)

    def test_unknown_raises(self):
        cfg = _tiny_cfg()
        cfg.optimizer.lr.decay_type = "cosine"
        opt = self._make_opt()
        with pytest.raises(ValueError, match="Unsupported lr decay_type"):
            get_lr_scheduler(opt, cfg)

    def test_warmup_wrapper(self):
        cfg = _tiny_cfg()
        cfg.optimizer.lr.decay_type = "fixed"
        cfg.optimizer.lr.num_warmup_steps = 10
        opt = self._make_opt(lr=0.01)
        sched = get_lr_scheduler(opt, cfg)
        assert isinstance(sched, _WarmupScheduler)
        # First step: lr should be < base lr (warmup)
        sched.step()
        assert opt.param_groups[0]["lr"] < 0.01

    def test_warmup_with_inner(self):
        cfg = _tiny_cfg()
        cfg.optimizer.lr.decay_type = "exp_decay"
        cfg.optimizer.lr.exp_decay_rate = 0.99
        cfg.optimizer.lr.num_warmup_steps = 5
        opt = self._make_opt(lr=0.1)
        sched = get_lr_scheduler(opt, cfg)
        assert isinstance(sched, _WarmupScheduler)


# ---------------------------------------------------------------------------
# Test: save_checkpoint / load_checkpoint roundtrip
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_load_roundtrip(self):
        cfg = _tiny_cfg()
        algo = _DummyAlgorithm(cfg)
        opt = get_optimizer(algo, cfg)

        # Do one forward + backward to update optimizer state
        dummy = algo.linear(torch.randn(2, 8))
        dummy.sum().backward()
        opt.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(tmpdir, algo, opt, 42, cfg)
            assert os.path.isfile(path)

            # Create a fresh model and optimizer
            algo2 = _DummyAlgorithm(cfg)
            opt2 = get_optimizer(algo2, cfg)

            step = load_checkpoint(path, algo2, opt2)
            assert step == 42

            # Check model weights match
            for p1, p2 in zip(algo.parameters(), algo2.parameters()):
                assert torch.allclose(p1, p2)

    def test_find_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # No checkpoints
            assert _find_latest_checkpoint(tmpdir) is None

            # Create some dummy checkpoint files
            for step in [10, 50, 30]:
                torch.save({"global_step": step}, os.path.join(tmpdir, f"checkpoint_{step}.pt"))

            latest = _find_latest_checkpoint(tmpdir)
            assert latest is not None
            assert "checkpoint_50" in latest

    def test_maybe_restore_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _tiny_cfg()
            algo = _DummyAlgorithm(cfg)
            opt = get_optimizer(algo, cfg)
            step = maybe_restore_checkpoint(tmpdir, algo, opt)
            assert step == 0


# ---------------------------------------------------------------------------
# Test: train() runs for a few iterations with tiny config
# ---------------------------------------------------------------------------


class TestTrainLoop:
    def _make_fake_batch(self, batch_size=1, num_frames=4):
        """Create a batch tuple matching the collate_fn output."""
        frames = torch.randn(batch_size, num_frames, 3, 32, 32)
        frame_labels = torch.zeros(batch_size, num_frames, dtype=torch.long)
        seq_labels = torch.zeros(batch_size, dtype=torch.long)
        seq_lens = torch.full((batch_size,), num_frames, dtype=torch.long)
        names = ["video_0"] * batch_size
        return frames, frame_labels, seq_labels, seq_lens, names

    def test_train_runs_and_saves_checkpoint(self):
        """Verify that train() completes, saves a checkpoint, and the loss is finite."""
        cfg = _tiny_cfg()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.logdir = tmpdir

            fake_batch = self._make_fake_batch()
            fake_loader = [fake_batch, fake_batch, fake_batch]

            with mock.patch("tcc.train.get_algo") as mock_get_algo, \
                 mock.patch("tcc.train.create_dataset") as mock_create_ds:

                algo = _DummyAlgorithm(cfg)
                mock_get_algo.return_value = algo
                mock_create_ds.return_value = fake_loader

                train(cfg)

                # Check that a final checkpoint was saved
                ckpts = [f for f in os.listdir(tmpdir) if f.endswith(".pt")]
                assert len(ckpts) > 0

                # Check that TensorBoard logs were created
                tb_dir = os.path.join(tmpdir, "train_logs")
                assert os.path.isdir(tb_dir)

                # Check config was saved
                assert os.path.isfile(os.path.join(tmpdir, "config.yaml"))

    def test_train_loss_is_finite(self):
        """Verify that the loss computed during training is always finite."""
        cfg = _tiny_cfg()
        cfg.train.max_iters = 3

        losses = []
        orig_train_one_iter = _DummyAlgorithm.train_one_iter

        def _recording_train_one_iter(self, data, steps, seq_lens, gs, opt):
            loss = orig_train_one_iter(self, data, steps, seq_lens, gs, opt)
            losses.append(loss)
            return loss

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.logdir = tmpdir

            fake_batch = self._make_fake_batch()
            fake_loader = [fake_batch] * 5

            with mock.patch("tcc.train.get_algo") as mock_get_algo, \
                 mock.patch("tcc.train.create_dataset") as mock_create_ds, \
                 mock.patch.object(
                     _DummyAlgorithm, "train_one_iter", _recording_train_one_iter
                 ):

                algo = _DummyAlgorithm(cfg)
                mock_get_algo.return_value = algo
                mock_create_ds.return_value = fake_loader

                train(cfg)

            assert len(losses) >= 2
            for loss_val in losses:
                assert math.isfinite(loss_val), f"Non-finite loss: {loss_val}"

    def test_train_respects_max_iters(self):
        """Verify training stops at max_iters even with more data."""
        cfg = _tiny_cfg()
        cfg.train.max_iters = 3
        cfg.checkpoint.save_interval = 100  # don't save mid-loop

        call_count = 0
        orig_train_one_iter = _DummyAlgorithm.train_one_iter

        def _counting_iter(self, data, steps, seq_lens, gs, opt):
            nonlocal call_count
            call_count += 1
            return orig_train_one_iter(self, data, steps, seq_lens, gs, opt)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.logdir = tmpdir
            fake_batch = self._make_fake_batch()
            # Provide more batches than max_iters
            fake_loader = [fake_batch] * 100

            with mock.patch("tcc.train.get_algo") as mock_get_algo, \
                 mock.patch("tcc.train.create_dataset") as mock_create_ds, \
                 mock.patch.object(
                     _DummyAlgorithm, "train_one_iter", _counting_iter
                 ):
                algo = _DummyAlgorithm(cfg)
                mock_get_algo.return_value = algo
                mock_create_ds.return_value = fake_loader

                train(cfg)

            assert call_count == 3


# ---------------------------------------------------------------------------
# Test: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLI:
    def test_default_args(self):
        with mock.patch("tcc.train.train") as mock_train, \
             mock.patch("sys.argv", ["train.py"]):
            main()
            mock_train.assert_called_once()
            called_cfg = mock_train.call_args[0][0]
            assert called_cfg.logdir == "/tmp/alignment_logs"

    def test_custom_logdir(self):
        with mock.patch("tcc.train.train") as mock_train, \
             mock.patch("sys.argv", ["train.py", "--logdir", "/tmp/my_logs"]):
            main()
            called_cfg = mock_train.call_args[0][0]
            assert called_cfg.logdir == "/tmp/my_logs"

    def test_config_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("training_algo: sal\n")
            f.write("train:\n  max_iters: 500\n")
            config_path = f.name

        try:
            with mock.patch("tcc.train.train") as mock_train, \
                 mock.patch(
                     "sys.argv",
                     ["train.py", "--config", config_path, "--logdir", "/tmp/test"],
                 ):
                main()
                called_cfg = mock_train.call_args[0][0]
                assert called_cfg.training_algo == "sal"
                assert called_cfg.train.max_iters == 500
                assert called_cfg.logdir == "/tmp/test"
        finally:
            os.unlink(config_path)
