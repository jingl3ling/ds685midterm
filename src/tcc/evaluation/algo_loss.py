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

"""AlgoLoss evaluation task — PyTorch port."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from tcc.evaluation.task import Task

logger = logging.getLogger(__name__)


def get_loss(
    algo: Any,
    iterator: Any,
    global_step: int,
    split: str,
    cfg: Any,
) -> float:
    """Run the algorithm in eval mode for a number of batches and average loss.

    Parameters
    ----------
    algo:
        The Algorithm (nn.Module) to evaluate.
    iterator:
        An iterable (DataLoader) yielding batches.
    global_step:
        Current training step.
    split:
        Name of the data split (e.g. 'val', 'train').
    cfg:
        TCCConfig instance.

    Returns
    -------
    float:
        Average loss over ``cfg.eval.val_iters`` batches.
    """
    algo.eval()
    val_iters = cfg.eval.val_iters
    total_loss = 0.0
    count = 0

    data_iter = iter(iterator)

    with torch.no_grad():
        for _ in range(val_iters):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            frames, frame_labels, seq_labels, seq_lens, names = batch

            # Move to the same device as the model
            device = next(algo.parameters()).device
            frames = frames.to(device)
            seq_lens = seq_lens.to(device)

            # Build data dict and steps
            data = {"frames": frames}
            if frame_labels is not None:
                data["frame_labels"] = frame_labels.to(device)
            if seq_labels is not None:
                data["seq_labels"] = seq_labels.to(device)

            num_steps = cfg.eval.num_frames
            batch_size = frames.shape[0]
            steps = torch.arange(num_steps, device=device).unsqueeze(0).expand(
                batch_size, -1
            )

            embs = algo(data, steps, seq_lens, training=False)

            loss = algo.compute_loss(
                embs,
                steps,
                seq_lens,
                global_step,
                training=False,
                frame_labels=data.get("frame_labels"),
                seq_labels=data.get("seq_labels"),
            )

            total_loss += loss.item()
            count += 1

    algo.train()

    if count == 0:
        return float("nan")

    return total_loss / count


class AlgoLoss(Task):
    """Evaluate the algorithm's own loss on validation data.

    This is an iterator-based task (``downstream_task=False``).
    """

    def __init__(self) -> None:
        super().__init__(downstream_task=False)

    def evaluate_embeddings(
        self,
        algo: Any,
        global_step: int,
        cfg: Any,
        datasets: Dict[str, Any],
        writer: Optional[Any] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError("AlgoLoss uses iterators, not embeddings.")

    def evaluate_iterators(
        self,
        algo: Any,
        global_step: int,
        cfg: Any,
        iterators: Dict[str, Any],
        writer: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Compute average loss on each split.

        Parameters
        ----------
        algo:
            Algorithm module.
        global_step:
            Current training step.
        cfg:
            TCCConfig.
        iterators:
            Dict mapping split name to DataLoader.
        writer:
            Optional TensorBoard SummaryWriter.

        Returns
        -------
        dict:
            Mapping of ``"algo_loss/{split}"`` to average loss.
        """
        metrics: Dict[str, float] = {}

        for split, iterator in iterators.items():
            avg_loss = get_loss(algo, iterator, global_step, split, cfg)
            key = f"algo_loss/{split}"
            metrics[key] = avg_loss
            logger.info(
                "Step %d | %s = %.6f", global_step, key, avg_loss
            )
            if writer is not None:
                writer.add_scalar(key, avg_loss, global_step)

        return metrics
