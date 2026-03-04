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

"""Stochastic alignment between sampled cycles in the sequences in a batch."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tcc.losses import classification_loss, regression_loss


def _align_single_cycle(
    cycle: torch.Tensor,
    embs: torch.Tensor,
    cycle_length: int,
    num_steps: int,
    similarity_type: str,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Takes a single cycle and returns logits (similarity scores) and labels."""
    # Choose random frame.
    n_idx = torch.randint(0, num_steps, (), device=embs.device)
    # Create labels.
    onehot_labels = F.one_hot(n_idx, num_steps).float()

    # Choose query feats for first frame.
    query_feats = embs[cycle[0], n_idx : n_idx + 1]  # [1, D]

    num_channels = query_feats.shape[-1]
    for c in range(1, cycle_length + 1):
        candidate_feats = embs[cycle[c]]  # [T, D]

        if similarity_type == "l2":
            # Find L2 distance.
            mean_squared_distance = (
                (query_feats.expand(num_steps, -1) - candidate_feats)
                .square()
                .sum(dim=1)
            )
            # Convert L2 distance to similarity.
            similarity = -mean_squared_distance
        elif similarity_type == "cosine":
            # Dot product of embeddings.
            similarity = (candidate_feats @ query_feats.t()).squeeze(1)
        else:
            raise ValueError("similarity_type can either be l2 or cosine.")

        # Scale the distance by number of channels.
        similarity = similarity / float(num_channels)
        # Scale the distance by temperature.
        similarity = similarity / temperature

        beta = F.softmax(similarity, dim=0)  # [T]
        beta = beta.unsqueeze(1).expand(-1, num_channels)  # [T, D]

        # Find weighted nearest neighbour.
        query_feats = (beta * candidate_feats).sum(dim=0, keepdim=True)  # [1, D]

    return similarity, onehot_labels


def _align(
    cycles: torch.Tensor,
    embs: torch.Tensor,
    num_steps: int,
    num_cycles: int,
    cycle_length: int,
    similarity_type: str,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align by finding cycles in embs."""
    logits_list = []
    labels_list = []
    for i in range(num_cycles):
        logits, labels = _align_single_cycle(
            cycles[i], embs, cycle_length, num_steps,
            similarity_type, temperature,
        )
        logits_list.append(logits)
        labels_list.append(labels)

    logits = torch.stack(logits_list)
    labels = torch.stack(labels_list)

    return logits, labels


def gen_cycles(
    num_cycles: int,
    batch_size: int,
    cycle_length: int = 2,
) -> torch.Tensor:
    """Generates cycles for alignment.

    Generates a batch of indices to cycle over. For example setting
    num_cycles=2, batch_size=5, cycle_length=3 might return something like:
    cycles = [[0, 3, 4, 0], [1, 2, 0, 3]].

    Args:
        num_cycles: Number of cycles that will be matched in one pass.
        batch_size: Number of sequences in one batch.
        cycle_length: Length of the cycles. Default is 2, giving cycles
            like [0, 1, 0].

    Returns:
        Tensor of shape [num_cycles, cycle_length + 1] with batch indices
        denoting cycles.
    """
    # Create a [batch_size, num_cycles] tensor of tiled range indices,
    # shuffle rows, then reshape to [num_cycles, batch_size].
    sorted_idxes = torch.arange(batch_size).unsqueeze(0).expand(num_cycles, -1)
    sorted_idxes = sorted_idxes.t().contiguous()  # [batch_size, num_cycles]

    # Shuffle along dim 0 (rows).
    perm = torch.randperm(batch_size)
    sorted_idxes = sorted_idxes[perm]

    cycles = sorted_idxes.t().contiguous()  # [num_cycles, batch_size]
    cycles = cycles[:, :cycle_length]

    # Append the first index at the end to create cycle.
    cycles = torch.cat([cycles, cycles[:, 0:1]], dim=1)
    return cycles


def compute_stochastic_alignment_loss(
    embs: torch.Tensor,
    steps: torch.Tensor,
    seq_lens: torch.Tensor,
    num_steps: int,
    batch_size: int,
    loss_type: str,
    similarity_type: str,
    num_cycles: int,
    cycle_length: int,
    temperature: float,
    label_smoothing: float,
    variance_lambda: float,
    huber_delta: float,
    normalize_indices: bool,
) -> torch.Tensor:
    """Compute cycle-consistency loss by stochastically sampling cycles.

    Args:
        embs: Sequential embeddings of shape [N, T, D].
        steps: Step indices of shape [N, T].
        seq_lens: Lengths of the sequences, shape [N].
        num_steps: Number of timesteps in the embeddings.
        batch_size: Batch size.
        loss_type: Kind of loss function to use. Supported: 'classification',
            'regression_mse', 'regression_mse_var', 'regression_huber'.
        similarity_type: Similarity metric: 'l2' or 'cosine'.
        num_cycles: Number of cycles to match while aligning stochastically.
        cycle_length: Lengths of the cycle to use for matching.
        temperature: Temperature scaling for softmax similarity distributions.
        label_smoothing: Label smoothing factor for classification loss.
        variance_lambda: Weight of variance term for regression_mse_var.
        huber_delta: Huber delta for regression_huber.
        normalize_indices: If True, normalizes indices by sequence lengths.

    Returns:
        Scalar loss tensor.
    """
    # Generate cycles.
    cycles = gen_cycles(num_cycles, batch_size, cycle_length)

    logits, labels = _align(
        cycles, embs, num_steps, num_cycles, cycle_length,
        similarity_type, temperature,
    )

    if loss_type == "classification":
        loss = classification_loss(logits, labels, label_smoothing)
    elif "regression" in loss_type:
        steps = steps[cycles[:, 0]]
        seq_lens = seq_lens[cycles[:, 0]]
        loss = regression_loss(
            logits, labels, num_steps, steps, seq_lens,
            loss_type, normalize_indices, variance_lambda, huber_delta,
        )
    else:
        raise ValueError(
            "Unidentified loss type %s. Currently supported loss "
            "types are: regression_mse, regression_huber, "
            "classification." % loss_type
        )
    return loss
