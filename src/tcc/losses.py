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

"""Loss functions imposing the cycle-consistency constraints."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    """Loss function based on classifying the correct indices.

    In the paper, this is called Cycle-back Classification.

    Args:
        logits: Pre-softmax scores used for classification loss. These are
            similarity scores after cycling back to the starting sequence.
        labels: One-hot labels containing the ground truth. The index where
            the cycle started is 1.
        label_smoothing: Label smoothing factor which can be used to
            determine how hard the alignment should be.

    Returns:
        A scalar classification loss calculated using standard softmax
        cross-entropy loss.
    """
    # Stop gradients from labels as we are generating labels.
    labels = labels.detach()

    # Apply label smoothing manually:
    # smoothed = (1 - label_smoothing) * labels + label_smoothing / num_classes
    num_classes = labels.shape[-1]
    smoothed = (1.0 - label_smoothing) * labels + label_smoothing / num_classes

    # Compute cross-entropy from logits and smoothed soft targets.
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(smoothed * log_probs).sum(dim=-1)
    return loss.mean()


def regression_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    steps: torch.Tensor,
    seq_lens: torch.Tensor,
    loss_type: str,
    normalize_indices: bool,
    variance_lambda: float,
    huber_delta: float,
) -> torch.Tensor:
    """Loss function based on regressing to the correct indices.

    In the paper, this is called Cycle-back Regression. There are 3 variants
    of this loss:
    i) regression_mse: MSE of the predicted indices and ground truth indices.
    ii) regression_mse_var: MSE of the predicted indices that takes into account
        the variance of the similarities.
    iii) regression_huber: Huber loss between the predicted indices and ground
        truth indices.

    Args:
        logits: Pre-softmax similarity scores after cycling back to the
            starting sequence.
        labels: One-hot labels containing the ground truth.
        num_steps: Number of steps in the sequence embeddings.
        steps: Step indices/frame indices of the embeddings of the shape
            [N, T].
        seq_lens: Lengths of the sequences from which the sampling was done.
        loss_type: Specifies the kind of regression loss function.
            Currently supported: regression_mse, regression_mse_var,
            regression_huber.
        normalize_indices: If True, normalizes indices by sequence lengths.
        variance_lambda: Weight of the variance of the similarity predictions
            while cycling back.
        huber_delta: Huber delta for the Huber loss variant.

    Returns:
        A scalar loss calculated using a variant of regression.
    """
    # Stop gradients from labels and steps as we are generating them.
    labels = labels.detach()
    steps = steps.detach()

    if normalize_indices:
        float_seq_lens = seq_lens.float()
        tile_seq_lens = float_seq_lens.unsqueeze(1).expand(-1, num_steps)
        steps = steps.float() / tile_seq_lens
    else:
        steps = steps.float()

    beta = F.softmax(logits, dim=-1)
    true_time = (steps * labels).sum(dim=1)
    pred_time = (steps * beta).sum(dim=1)

    if loss_type in ("regression_mse", "regression_mse_var"):
        if "var" in loss_type:
            # Variance aware regression.
            pred_time_tiled = pred_time.unsqueeze(1).expand(-1, num_steps)
            pred_time_variance = (
                (steps - pred_time_tiled).square() * beta
            ).sum(dim=1)

            # Using log of variance as it is numerically stabler.
            pred_time_log_var = torch.log(pred_time_variance)
            squared_error = (true_time - pred_time).square()
            return (
                torch.exp(-pred_time_log_var) * squared_error
                + variance_lambda * pred_time_log_var
            ).mean()
        else:
            return F.mse_loss(pred_time, true_time)
    elif loss_type == "regression_huber":
        return F.smooth_l1_loss(
            pred_time, true_time, beta=huber_delta
        )
    else:
        raise ValueError(
            "Unsupported regression loss %s. Supported losses are: "
            "regression_mse, regression_mse_var and regression_huber."
            % loss_type
        )
