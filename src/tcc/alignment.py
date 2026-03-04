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

"""Variants of the cycle-consistency loss described in the TCC paper.

The Temporal Cycle-Consistency (TCC) Learning paper
(https://arxiv.org/pdf/1904.07846.pdf) describes a loss that enables learning
of self-supervised representations from sequences of embeddings that are good
at temporally fine-grained tasks like phase classification, video alignment etc.

These losses impose cycle-consistency constraints between sequences of
embeddings. Another interpretation of the cycle-consistency constraints is
that of mutual nearest-neighbors. This means if state A in sequence 1 is the
nearest neighbor of state B in sequence 2 then it must also follow that B is the
nearest neighbor of A. We found that imposing this constraint on a dataset of
related sequences (like videos of people pitching a baseball) allows us to learn
generally useful visual representations.

This code allows the user to apply the loss while giving them the freedom to
choose the right encoder for their dataset/task. One advice for choosing an
encoder is to ensure that the encoder does not solve the mutual neighbor finding
task in a trivial fashion. For example, if one uses an LSTM or Transformer with
positional encodings, the matching between sequences may be done trivially by
counting the frame index with the encoder rather than learning good features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tcc.deterministic_alignment import compute_deterministic_alignment_loss
from tcc.stochastic_alignment import compute_stochastic_alignment_loss


def compute_alignment_loss(
    embs: torch.Tensor,
    batch_size: int,
    steps: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    stochastic_matching: bool = False,
    normalize_embeddings: bool = False,
    loss_type: str = "classification",
    similarity_type: str = "l2",
    num_cycles: int = 20,
    cycle_length: int = 2,
    temperature: float = 0.1,
    label_smoothing: float = 0.1,
    variance_lambda: float = 0.001,
    huber_delta: float = 0.1,
    normalize_indices: bool = True,
) -> torch.Tensor:
    """Computes alignment loss between sequences of embeddings.

    This function is a wrapper around different variants of the alignment loss
    described in deterministic_alignment.py and stochastic_alignment.py.

    There are four major hparams that need to be tuned while applying the loss:
    i) Should the loss be applied with L2 normalization on the embeddings?
    ii) Should we perform stochastic alignment of sequences?
    iii) Should we apply cycle-consistency constraints using a classification
        loss or a regression loss?
    iv) Should the similarity metric be based on L2 distance or cosine
        similarity?

    Args:
        embs: Sequential embeddings of shape [N, T, D] where N is the
            batch size, T is the number of timesteps, D is the embedding size.
        batch_size: Size of the batch.
        steps: Step indices of shape [N, T]. If None, assumes uniform sampling
            and uses torch.arange(num_steps).
        seq_lens: Lengths of the sequences, shape [N]. If None, assumes
            equal to the number of timesteps.
        stochastic_matching: If True, uses stochastic alignment.
        normalize_embeddings: If True, L2-normalizes embeddings.
        loss_type: Kind of loss function to use. Supported: 'classification',
            'regression_mse', 'regression_mse_var', 'regression_huber'.
        similarity_type: Similarity metric: 'l2' or 'cosine'.
        num_cycles: Number of cycles for stochastic matching.
        cycle_length: Length of cycles for stochastic matching.
        temperature: Temperature scaling for softmax similarity distributions.
        label_smoothing: Label smoothing factor for classification loss.
        variance_lambda: Weight of variance term for regression_mse_var.
        huber_delta: Huber delta for regression_huber.
        normalize_indices: If True, normalizes indices by sequence lengths.

    Returns:
        Scalar loss tensor.
    """
    # -------------------------------------------------------------------------
    # Checking inputs and setting defaults.
    # -------------------------------------------------------------------------
    num_steps = embs.shape[1]

    # If steps has not been provided assume sampling has been done uniformly.
    if steps is None:
        steps = torch.arange(num_steps, device=embs.device).unsqueeze(0).expand(
            batch_size, -1
        )

    # If seq_lens has not been provided assume it equals the time axis size.
    if seq_lens is None:
        seq_lens = torch.full(
            (batch_size,), num_steps, device=embs.device, dtype=torch.long
        )

    assert batch_size == embs.shape[0], (
        f"batch_size ({batch_size}) != embs.shape[0] ({embs.shape[0]})"
    )
    assert num_steps == steps.shape[1], (
        f"num_steps ({num_steps}) != steps.shape[1] ({steps.shape[1]})"
    )
    assert batch_size == steps.shape[0], (
        f"batch_size ({batch_size}) != steps.shape[0] ({steps.shape[0]})"
    )

    # -------------------------------------------------------------------------
    # Perform alignment and return loss.
    # -------------------------------------------------------------------------
    if normalize_embeddings:
        embs = F.normalize(embs, dim=-1)

    if stochastic_matching:
        loss = compute_stochastic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            num_steps=num_steps,
            batch_size=batch_size,
            loss_type=loss_type,
            similarity_type=similarity_type,
            num_cycles=num_cycles,
            cycle_length=cycle_length,
            temperature=temperature,
            label_smoothing=label_smoothing,
            variance_lambda=variance_lambda,
            huber_delta=huber_delta,
            normalize_indices=normalize_indices,
        )
    else:
        loss = compute_deterministic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            num_steps=num_steps,
            batch_size=batch_size,
            loss_type=loss_type,
            similarity_type=similarity_type,
            temperature=temperature,
            label_smoothing=label_smoothing,
            variance_lambda=variance_lambda,
            huber_delta=huber_delta,
            normalize_indices=normalize_indices,
        )

    return loss
