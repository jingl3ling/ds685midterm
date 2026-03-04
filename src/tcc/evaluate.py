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

"""Main evaluation script for TCC — PyTorch port.

Usage::

    python -m tcc.evaluate --config path/to/config.yaml --logdir /tmp/logs

Supports one-shot evaluation or continuous evaluation (watching for new
checkpoints in ``logdir``).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from tcc.config import TCCConfig, load_config
from tcc.evaluation.registry import get_tasks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def get_embeddings_dataset(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    max_embs: int = 0,
) -> Dict[str, Any]:
    """Extract embeddings from all data in a DataLoader.

    Parameters
    ----------
    model:
        The algorithm (nn.Module) in eval mode.
    dataloader:
        A DataLoader yielding batches of
        ``(frames, frame_labels, seq_labels, seq_lens, names)``.
    device:
        Device for computation.
    max_embs:
        Maximum number of embedding vectors to collect (0 = unlimited).

    Returns
    -------
    dict:
        Contains:
        - ``'embeddings'``: np.ndarray of shape ``[N, D]``
        - ``'labels'``: np.ndarray of shape ``[N]``
        - ``'embeddings_list'``: list of per-video np.ndarray ``[T_i, D]``
        - ``'labels_list'``: list of per-video np.ndarray ``[T_i]``
        - ``'names'``: list of video names
        - ``'num_classes'``: int
    """
    all_embs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    embs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    names: List[str] = []
    total_frames = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            frames, frame_labels, seq_labels, seq_lens, batch_names = batch
            frames = frames.to(device)
            seq_lens = seq_lens.to(device)

            batch_size = frames.shape[0]
            num_steps = frames.shape[1]

            # Build dummy data dict and steps
            data = {"frames": frames}
            steps = torch.arange(num_steps, device=device).unsqueeze(0).expand(
                batch_size, -1
            )

            embs = model(data, steps, seq_lens, training=False)
            embs_np = embs.cpu().numpy()
            labels_np = frame_labels.numpy()

            for b in range(batch_size):
                sl = int(seq_lens[b].item()) if seq_lens[b].item() > 0 else num_steps
                vid_embs = embs_np[b, :sl]
                vid_labels = labels_np[b, :sl]

                embs_list.append(vid_embs)
                labels_list.append(vid_labels)
                all_embs.append(vid_embs)
                all_labels.append(vid_labels)
                names.append(batch_names[b])

                total_frames += sl

            if max_embs > 0 and total_frames >= max_embs:
                break

    # Concatenate all
    if all_embs:
        embeddings = np.concatenate(all_embs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
    else:
        embeddings = np.zeros((0, 0), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)

    num_classes = int(labels.max()) + 1 if len(labels) > 0 else 0

    return {
        "embeddings": embeddings,
        "labels": labels,
        "embeddings_list": embs_list,
        "labels_list": labels_list,
        "names": names,
        "num_classes": num_classes,
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_once(
    algo: torch.nn.Module,
    cfg: TCCConfig,
    device: torch.device,
    global_step: int,
    iterators: Optional[Dict[str, Any]] = None,
    embeddings_datasets: Optional[Dict[str, Any]] = None,
    writer: Optional[Any] = None,
) -> Dict[str, float]:
    """Run all configured evaluation tasks once.

    Parameters
    ----------
    algo:
        The Algorithm module.
    cfg:
        TCCConfig instance.
    device:
        Torch device.
    global_step:
        Current training step.
    iterators:
        Dict of split name -> DataLoader for iterator-based tasks.
    embeddings_datasets:
        Dict of split name -> embedding dataset dict for downstream tasks.
    writer:
        Optional TensorBoard SummaryWriter.

    Returns
    -------
    dict:
        All metrics collected from all tasks.
    """
    iterator_tasks, embedding_tasks = get_tasks(cfg.eval.tasks)

    all_metrics: Dict[str, float] = {}

    # Run iterator-based tasks
    if iterators and iterator_tasks:
        for name, task in iterator_tasks.items():
            logger.info("Running iterator task: %s", name)
            metrics = task.evaluate(
                algo, global_step, cfg,
                iterators=iterators,
                writer=writer,
            )
            all_metrics.update(metrics)

    # Run embedding-based tasks
    if embeddings_datasets and embedding_tasks:
        for name, task in embedding_tasks.items():
            logger.info("Running embedding task: %s", name)
            metrics = task.evaluate(
                algo, global_step, cfg,
                embeddings_dataset=embeddings_datasets,
                writer=writer,
            )
            all_metrics.update(metrics)

    return all_metrics


def _find_latest_checkpoint(logdir: str) -> Optional[str]:
    """Find the latest checkpoint file in *logdir*.

    Looks for files matching ``*.pt`` or ``*.pth`` and returns the one
    with the highest modification time.
    """
    patterns = [os.path.join(logdir, "*.pt"), os.path.join(logdir, "*.pth")]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def evaluate(cfg: TCCConfig, args: argparse.Namespace) -> None:
    """Main evaluation entry point.

    Optionally watches for new checkpoints when ``--continuous-eval``
    is set.

    Parameters
    ----------
    cfg:
        TCCConfig instance.
    args:
        Parsed command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logdir = args.logdir or cfg.logdir

    logger.info("Evaluation logdir: %s", logdir)
    logger.info("Device: %s", device)

    # Optional TensorBoard writer
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(logdir, "eval_tb")
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
    except ImportError:
        logger.warning("TensorBoard not available; skipping TB logging.")

    evaluated_checkpoints: set = set()

    while True:
        ckpt_path = _find_latest_checkpoint(logdir)

        if ckpt_path is None:
            if not args.continuous_eval:
                logger.error("No checkpoint found in %s", logdir)
                return
            logger.info("No checkpoint found yet. Waiting...")
            time.sleep(10)
            continue

        if ckpt_path in evaluated_checkpoints:
            if not args.continuous_eval:
                return
            time.sleep(10)
            continue

        logger.info("Evaluating checkpoint: %s", ckpt_path)

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        global_step = checkpoint.get("global_step", 0)

        # Build algorithm
        from tcc.algos.registry import get_algo
        algo = get_algo(cfg.training_algo, cfg=cfg)
        if "model_state_dict" in checkpoint:
            algo.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            algo.load_state_dict(checkpoint["state_dict"])
        algo.to(device)
        algo.eval()

        # Run evaluation
        metrics = evaluate_once(
            algo, cfg, device, global_step, writer=writer,
        )

        logger.info("Step %d metrics: %s", global_step, metrics)
        evaluated_checkpoints.add(ckpt_path)

        if not args.continuous_eval:
            break

    if writer is not None:
        writer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="TCC Evaluation (PyTorch)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Directory containing checkpoints. Overrides config logdir.",
    )
    parser.add_argument(
        "--continuous-eval",
        action="store_true",
        default=False,
        help="Continuously watch for new checkpoints.",
    )
    parser.add_argument(
        "--max-embs",
        type=int,
        default=0,
        help="Maximum number of embeddings to extract (0 = unlimited).",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    evaluate(cfg, args)


if __name__ == "__main__":
    main()
