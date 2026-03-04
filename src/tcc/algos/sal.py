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

"""Shuffle and Learn (SAL) algorithm — PyTorch port."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcc.algos.algorithm import Algorithm
from tcc.config import SALConfig, TCCConfig
from tcc.models import Classifier


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_shuffled_indices_and_labels(
    batch_size: int,
    num_samples: int,
    shuffle_fraction: float,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate ordered vs shuffled triplet indices and one-hot labels.

    For each sample we pick three consecutive frame indices (i, i+1, i+2).
    With probability ``shuffle_fraction`` we swap i and i+2 to create
    a shuffled (negative) triplet; otherwise the triplet stays ordered
    (positive).

    Args:
        batch_size: number of sequences in the batch.
        num_samples: number of triplets to sample per sequence.
        shuffle_fraction: probability of shuffling a triplet.
        num_steps: number of timesteps available per sequence.

    Returns:
        indices: ``[batch_size * num_samples, 3]`` frame indices.
        labels: ``[batch_size * num_samples, 2]`` one-hot labels
            (class 0 = ordered, class 1 = shuffled).
    """
    total = batch_size * num_samples

    # Sample starting indices (must allow room for i+2)
    max_start = max(num_steps - 2, 1)
    starts = torch.randint(0, max_start, (total,))

    indices = torch.stack([starts, starts + 1, starts + 2], dim=1)

    # Decide which triplets to shuffle
    shuffle_mask = torch.rand(total) < shuffle_fraction

    # Swap first and last for shuffled triplets
    shuffled_indices = indices.clone()
    shuffled_indices[shuffle_mask, 0] = indices[shuffle_mask, 2]
    shuffled_indices[shuffle_mask, 2] = indices[shuffle_mask, 0]

    # Labels: 0 = ordered, 1 = shuffled
    labels_idx = shuffle_mask.long()
    labels = F.one_hot(labels_idx, num_classes=2).float()

    return shuffled_indices, labels


def sample_batch(
    embs: torch.Tensor,
    batch_size: int,
    num_steps: int,
    cfg: SALConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample triplet embeddings and labels for SAL classification.

    Args:
        embs: ``[batch_size, num_steps, embedding_dim]``.
        batch_size: batch size.
        num_steps: number of timesteps.
        cfg: SAL config section.

    Returns:
        concat_embs: ``[batch_size * num_samples, 3 * embedding_dim]``
        labels: ``[batch_size * num_samples, 2]`` one-hot labels.
    """
    indices, labels = get_shuffled_indices_and_labels(
        batch_size=batch_size,
        num_samples=cfg.num_samples,
        shuffle_fraction=cfg.shuffle_fraction,
        num_steps=num_steps,
    )

    # indices: [total, 3] — need to gather from the correct sequence
    total = batch_size * cfg.num_samples
    seq_idx = torch.arange(batch_size).repeat_interleave(cfg.num_samples)

    # Gather embeddings for each of the 3 positions in the triplet
    triplet_parts = []
    for col in range(3):
        frame_idx = indices[:, col]
        triplet_parts.append(embs[seq_idx, frame_idx])

    concat_embs = torch.cat(triplet_parts, dim=-1)

    return concat_embs, labels


# ---------------------------------------------------------------------------
# SAL Algorithm
# ---------------------------------------------------------------------------


class SAL(Algorithm):
    """Shuffle and Learn algorithm.

    Creates a classifier that distinguishes ordered from shuffled
    frame triplets.
    """

    def __init__(
        self,
        cfg: Optional[TCCConfig] = None,
        model=None,
    ) -> None:
        super().__init__(cfg=cfg, model=model)

        emb_size = self.cfg.model.conv_embedder.embedding_size
        sal_cfg = self.cfg.sal

        self.classifier = Classifier(
            fc_layers=sal_cfg.fc_layers,
            dropout_rate=sal_cfg.dropout_rate,
            in_features=emb_size * 3,  # triplet concatenation
        )

    def compute_loss(
        self,
        embs: torch.Tensor,
        steps: torch.Tensor,
        seq_lens: torch.Tensor,
        global_step: int,
        training: bool,
        frame_labels: Optional[torch.Tensor] = None,
        seq_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = embs.shape[0]
        num_steps = embs.shape[1]

        concat_embs, labels = sample_batch(
            embs, batch_size, num_steps, self.cfg.sal,
        )

        # Move to same device as embs
        concat_embs = concat_embs.to(embs.device)
        labels = labels.to(embs.device)

        logits = self.classifier(concat_embs)

        # labels is one-hot [N, 2]; cross_entropy wants class indices
        target = labels.argmax(dim=-1)
        loss = F.cross_entropy(
            logits, target,
            label_smoothing=self.cfg.sal.label_smoothing,
        )
        return loss

    def get_algo_params(self) -> List[nn.Parameter]:
        return list(self.classifier.parameters())
