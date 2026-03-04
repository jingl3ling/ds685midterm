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

"""TCC training loop -- PyTorch port.

Replaces the TF2 training script.  Key differences from the original:
- No ``tf.distribute`` strategy -- single-GPU PyTorch.
- No absl flags -- uses ``argparse`` + config YAML files.
- ``torch.optim`` instead of ``tf.keras.optimizers``.
- ``torch.utils.tensorboard.SummaryWriter`` for logging.
- Native PyTorch checkpointing (``torch.save`` / ``torch.load``).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from typing import Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ExponentialLR,
    LambdaLR,
    MultiStepLR,
    PolynomialLR,
    _LRScheduler,
)
from torch.utils.tensorboard import SummaryWriter

from tcc.algos.algorithm import Algorithm
from tcc.algos.registry import get_algo
from tcc.config import TCCConfig, get_default_config, load_config, save_config
from tcc.datasets import DataConfig, create_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Learning rate scheduling
# ---------------------------------------------------------------------------

_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pt$")


class _WarmupScheduler(_LRScheduler):
    """Linear warmup wrapper around an inner LR scheduler.

    During the first ``num_warmup_steps`` the learning rate is linearly
    ramped from 0 to the base LR.  Afterwards, the inner scheduler takes
    over.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        inner: Optional[_LRScheduler],
        num_warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.inner = inner
        self.num_warmup_steps = num_warmup_steps
        # Store base LRs *before* super().__init__ calls step()
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):  # noqa: D401 (simple override)
        step = self.last_epoch
        if step < self.num_warmup_steps:
            scale = (step + 1) / max(1, self.num_warmup_steps)
            return [base * scale for base in self._base_lrs]
        # After warmup, delegate to the inner scheduler.
        if self.inner is not None:
            return self.inner.get_lr()
        return self._base_lrs


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    cfg: TCCConfig,
) -> Optional[_LRScheduler]:
    """Return a PyTorch LR scheduler based on ``cfg.optimizer.lr``.

    Supported ``decay_type`` values:
    - ``'fixed'``: no scheduling (returns ``None``).
    - ``'exp_decay'``: :class:`ExponentialLR`.
    - ``'manual'``: :class:`MultiStepLR`.
    - ``'poly'``: :class:`PolynomialLR`.

    If ``cfg.optimizer.lr.num_warmup_steps > 0`` a linear warmup wrapper
    is applied on top.
    """
    lr_cfg = cfg.optimizer.lr
    decay_type = lr_cfg.decay_type
    warmup_steps = lr_cfg.num_warmup_steps

    inner: Optional[_LRScheduler] = None

    if decay_type == "fixed":
        inner = None
    elif decay_type == "exp_decay":
        inner = ExponentialLR(optimizer, gamma=lr_cfg.exp_decay_rate)
    elif decay_type == "manual":
        inner = MultiStepLR(
            optimizer,
            milestones=list(lr_cfg.manual_lr_step_boundaries),
            gamma=lr_cfg.manual_lr_decay_rate,
        )
    elif decay_type == "poly":
        inner = PolynomialLR(
            optimizer,
            total_iters=cfg.train.max_iters,
            power=1.0,
        )
    else:
        raise ValueError(
            f"Unsupported lr decay_type '{decay_type}'. "
            "Choose from: fixed, exp_decay, manual, poly."
        )

    if warmup_steps > 0:
        return _WarmupScheduler(optimizer, inner, warmup_steps)

    return inner


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def get_optimizer(algo: Algorithm, cfg: TCCConfig) -> optim.Optimizer:
    """Create an optimizer for the algorithm's trainable parameters.

    Supported ``cfg.optimizer.type`` values:
    - ``'adam'``: :class:`torch.optim.Adam`.
    - ``'sgd'``: :class:`torch.optim.SGD` with ``momentum=0.9``.
    """
    params = algo.get_trainable_params()
    lr = cfg.optimizer.lr.initial_lr
    wd = cfg.optimizer.weight_decay

    opt_type = cfg.optimizer.type.lower()
    if opt_type == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(
            f"Unsupported optimizer type '{opt_type}'. "
            "Choose from: adam, sgd."
        )


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def save_checkpoint(
    logdir: str,
    algo: Algorithm,
    optimizer: optim.Optimizer,
    global_step: int,
    cfg: TCCConfig,
) -> str:
    """Save a training checkpoint and return the file path."""
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, f"checkpoint_{global_step}.pt")
    torch.save(
        {
            "global_step": global_step,
            "model_state_dict": algo.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    logger.info("Saved checkpoint to %s", path)
    return path


def load_checkpoint(
    path: str,
    algo: Algorithm,
    optimizer: Optional[optim.Optimizer] = None,
) -> int:
    """Load a checkpoint and return the global step.

    Parameters
    ----------
    path:
        Path to a ``.pt`` checkpoint file.
    algo:
        The algorithm whose ``state_dict`` will be restored.
    optimizer:
        If provided, its ``state_dict`` is also restored.

    Returns
    -------
    int
        The global step stored in the checkpoint.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    algo.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    global_step = ckpt.get("global_step", 0)
    logger.info("Restored checkpoint from %s at step %d", path, global_step)
    return global_step


def _find_latest_checkpoint(logdir: str) -> Optional[str]:
    """Return the path to the latest checkpoint in *logdir*, or ``None``."""
    pattern = os.path.join(logdir, "checkpoint_*.pt")
    paths = glob.glob(pattern)
    if not paths:
        return None
    # Sort by step number extracted from filename.
    def _step(p: str) -> int:
        m = _CHECKPOINT_RE.search(os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(paths, key=_step)


def maybe_restore_checkpoint(
    logdir: str,
    algo: Algorithm,
    optimizer: Optional[optim.Optimizer] = None,
) -> int:
    """Restore the latest checkpoint if one exists, returning the global step."""
    ckpt_path = _find_latest_checkpoint(logdir)
    if ckpt_path is not None:
        return load_checkpoint(ckpt_path, algo, optimizer)
    return 0


# ---------------------------------------------------------------------------
# Build a DataConfig from the top-level TCCConfig
# ---------------------------------------------------------------------------


def _build_data_config(cfg: TCCConfig) -> DataConfig:
    """Create a :class:`DataConfig` from the top-level :class:`TCCConfig`."""
    from tcc.datasets import AugmentationConfig as DataAugConfig

    aug = DataAugConfig(
        random_flip=cfg.augmentation.random_flip,
        random_crop=cfg.augmentation.random_crop,
        brightness=cfg.augmentation.brightness,
        brightness_max_delta=cfg.augmentation.brightness_max_delta,
        contrast=cfg.augmentation.contrast,
        contrast_lower=cfg.augmentation.contrast_lower,
        contrast_upper=cfg.augmentation.contrast_upper,
        hue=cfg.augmentation.hue,
        hue_max_delta=cfg.augmentation.hue_max_delta,
        saturation=cfg.augmentation.saturation,
        saturation_lower=cfg.augmentation.saturation_lower,
        saturation_upper=cfg.augmentation.saturation_upper,
    )

    return DataConfig(
        root_dir=cfg.path_to_tfrecords % cfg.datasets[0] if cfg.datasets else "",
        datasets=cfg.datasets,
        sampling_strategy=cfg.data.sampling_strategy,
        num_frames=cfg.train.num_frames,
        stride=cfg.data.stride,
        random_offset=cfg.data.random_offset,
        num_steps=cfg.data.num_steps,
        frame_stride=cfg.data.frame_stride,
        sample_all_stride=cfg.data.sample_all_stride,
        frame_labels=cfg.data.frame_labels,
        image_size=cfg.data.image_size,
        training_algo=cfg.training_algo,
        train_batch_size=cfg.train.batch_size,
        eval_batch_size=cfg.eval.batch_size,
        augmentation=aug,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: TCCConfig) -> None:
    """Run the TCC training loop.

    Parameters
    ----------
    cfg:
        Fully-populated :class:`TCCConfig`.
    """
    logdir = cfg.logdir
    os.makedirs(logdir, exist_ok=True)
    save_config(cfg, os.path.join(logdir, "config.yaml"))

    writer = SummaryWriter(os.path.join(logdir, "train_logs"))

    # Model + algo
    algo = get_algo(cfg.training_algo, cfg=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = algo.to(device)

    # Optimizer + scheduler
    optimizer = get_optimizer(algo, cfg)
    scheduler = get_lr_scheduler(optimizer, cfg)

    # Dataset
    data_cfg = _build_data_config(cfg)
    train_loader = create_dataset(split="train", mode="train", config=data_cfg)

    # Restore checkpoint
    global_step = maybe_restore_checkpoint(logdir, algo, optimizer)

    # Training loop
    algo.train()
    while global_step < cfg.train.max_iters:
        for batch in train_loader:
            if global_step >= cfg.train.max_iters:
                break

            frames, frame_labels, seq_labels, seq_lens, names = batch

            # Move data to device
            frames = frames.to(device)
            frame_labels = frame_labels.to(device)
            seq_labels = seq_labels.to(device)
            seq_lens = seq_lens.to(device)

            # Build the data dict expected by Algorithm.forward / train_one_iter
            num_frames = frames.shape[1]
            chosen_steps = torch.arange(num_frames, device=device).unsqueeze(0).expand(
                frames.shape[0], -1
            )

            data = {
                "frames": frames,
                "frame_labels": frame_labels,
                "seq_labels": seq_labels,
            }
            steps = chosen_steps
            sl = seq_lens

            loss = algo.train_one_iter(data, steps, sl, global_step, optimizer)

            if scheduler is not None:
                scheduler.step()

            # Logging
            if global_step % cfg.logging.report_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("loss", loss, global_step)
                writer.add_scalar("learning_rate", current_lr, global_step)
                logger.info(
                    "Iter[%d/%d], Loss: %.3f, LR: %.6f",
                    global_step,
                    cfg.train.max_iters,
                    loss,
                    current_lr,
                )

            # Save checkpoint
            if global_step % cfg.checkpoint.save_interval == 0 and global_step > 0:
                save_checkpoint(logdir, algo, optimizer, global_step, cfg)

            global_step += 1

    # Final checkpoint
    save_checkpoint(logdir, algo, optimizer, global_step, cfg)
    writer.close()
    logger.info("Training complete. Final step: %d", global_step)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse command-line arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="TCC training script (PyTorch)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="/tmp/alignment_logs",
        help="Directory for checkpoints and TensorBoard logs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = get_default_config()

    cfg.logdir = args.logdir

    train(cfg)


if __name__ == "__main__":
    main()
