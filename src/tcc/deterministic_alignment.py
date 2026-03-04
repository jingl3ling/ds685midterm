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

"""Deterministic alignment between all pairs of sequences in a batch."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tcc.losses import classification_loss, regression_loss


def pairwise_l2_distance(
    embs1: torch.Tensor,
    embs2: torch.Tensor,
) -> torch.Tensor:
    """Computes pairwise distances between all rows of embs1 and embs2.

    Uses the norm expansion trick: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b

    Args:
        embs1: Embeddings of shape [M, D].
        embs2: Embeddings of shape [N, D].

    Returns:
        Pairwise L2 distances of shape [M, N].
    """
    norm1 = (embs1**2).sum(dim=1, keepdim=True)       # [M, 1]
    norm2 = (embs2**2).sum(dim=1, keepdim=True).t()    # [1, N]

    # Clamp to zero to guard against negative values from floating point error.
    dist = torch.clamp(norm1 + norm2 - 2.0 * embs1 @ embs2.t(), min=0.0)
    return dist


def get_scaled_similarity(
    embs1: torch.Tensor,
    embs2: torch.Tensor,
    similarity_type: str,
    temperature: float,
) -> torch.Tensor:
    """Returns scaled similarity between all rows of embs1 and embs2.

    The similarity is scaled by the number of channels/embedding size and
    temperature.

    Args:
        embs1: Embeddings of shape [M, D].
        embs2: Embeddings of shape [N, D].
        similarity_type: Either 'l2' or 'cosine'.
        temperature: Temperature used in scaling logits before softmax.

    Returns:
        Similarity tensor of shape [M, N].
    """
    channels = float(embs1.shape[1])

    if similarity_type == "cosine":
        similarity = embs1 @ embs2.t()
    elif similarity_type == "l2":
        similarity = -1.0 * pairwise_l2_distance(embs1, embs2)
    else:
        raise ValueError("similarity_type can either be l2 or cosine.")

    # Scale by number of channels (helps with optimization).
    similarity = similarity / channels
    # Scale by temperature (controls soft/hard alignment).
    similarity = similarity / temperature

    return similarity


def align_pair_of_sequences(
    embs1: torch.Tensor,
    embs2: torch.Tensor,
    similarity_type: str,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a given pair of embedding sequences.

    Args:
        embs1: Embeddings of shape [M, D].
        embs2: Embeddings of shape [N, D].
        similarity_type: Either 'l2' or 'cosine'.
        temperature: Temperature used in scaling logits before softmax.

    Returns:
        logits: Pre-softmax similarity scores after cycling back, shape [M, M].
        labels: One-hot labels with ground truth identity, shape [M, M].
    """
    max_num_steps = embs1.shape[0]

    # Find distances between embs1 and embs2.
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
    # Softmax the distance.
    softmaxed_sim_12 = F.softmax(sim_12, dim=1)

    # Calculate soft-nearest neighbors.
    nn_embs = softmaxed_sim_12 @ embs2

    # Find distances between nn_embs and embs1.
    sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)

    logits = sim_21
    labels = F.one_hot(
        torch.arange(max_num_steps, device=embs1.device), max_num_steps
    ).float()

    return logits, labels


def compute_deterministic_alignment_loss(
    embs: torch.Tensor,
    steps: torch.Tensor,
    seq_lens: torch.Tensor,
    num_steps: int,
    batch_size: int,
    loss_type: str,
    similarity_type: str,
    temperature: float,
    label_smoothing: float,
    variance_lambda: float,
    huber_delta: float,
    normalize_indices: bool,
) -> torch.Tensor:
    """Compute cycle-consistency loss for all steps in each sequence.

    This aligns each pair of videos in the batch except with itself.
    When aligning it also matters which video is the starting video. So for N
    videos in the batch, we have N * (N-1) alignments happening.

    Args:
        embs: Sequential embeddings of shape [N, T, D].
        steps: Step indices of shape [N, T].
        seq_lens: Lengths of the sequences, shape [N].
        num_steps: Number of timesteps in the embeddings.
        batch_size: Size of the batch.
        loss_type: Kind of loss function to use. Supported: 'classification',
            'regression_mse', 'regression_mse_var', 'regression_huber'.
        similarity_type: Similarity metric: 'l2' or 'cosine'.
        temperature: Temperature scaling for softmax similarity distributions.
        label_smoothing: Label smoothing factor for classification loss.
        variance_lambda: Weight of variance term for regression_mse_var.
        huber_delta: Huber delta for regression_huber.
        normalize_indices: If True, normalizes indices by sequence lengths.

    Returns:
        Scalar loss tensor.
    """
    labels_list = []
    logits_list = []
    steps_list = []
    seq_lens_list = []

    for i in range(batch_size):
        for j in range(batch_size):
            # We do not align the sequence with itself.
            if i != j:
                logits, labels = align_pair_of_sequences(
                    embs[i], embs[j], similarity_type, temperature
                )
                logits_list.append(logits)
                labels_list.append(labels)
                steps_list.append(
                    steps[i : i + 1].expand(num_steps, -1)
                )
                seq_lens_list.append(
                    seq_lens[i : i + 1].expand(num_steps)
                )

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    steps = torch.cat(steps_list, dim=0)
    seq_lens = torch.cat(seq_lens_list, dim=0)

    if loss_type == "classification":
        loss = classification_loss(logits, labels, label_smoothing)
    elif "regression" in loss_type:
        loss = regression_loss(
            logits,
            labels,
            num_steps,
            steps,
            seq_lens,
            loss_type,
            normalize_indices,
            variance_lambda,
            huber_delta,
        )
    else:
        raise ValueError(
            "Unidentified loss_type %s. Currently supported loss "
            "types are: regression_mse, regression_huber, "
            "classification." % loss_type
        )

    return loss
