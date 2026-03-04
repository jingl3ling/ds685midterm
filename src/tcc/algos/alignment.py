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

"""TCC Alignment algorithm — PyTorch port."""

from __future__ import annotations

from typing import Optional

import torch

from tcc.algos.algorithm import Algorithm
from tcc.alignment import compute_alignment_loss


class Alignment(Algorithm):
    """Temporal Cycle-Consistency alignment algorithm.

    Delegates to :func:`tcc.alignment.compute_alignment_loss` using
    parameters from ``cfg.alignment``.
    """

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
        acfg = self.cfg.alignment

        loss = compute_alignment_loss(
            embs=embs,
            batch_size=batch_size,
            steps=steps,
            seq_lens=seq_lens,
            stochastic_matching=acfg.stochastic_matching,
            normalize_embeddings=False,
            loss_type=acfg.loss_type,
            similarity_type=acfg.similarity_type,
            temperature=acfg.temperature,
            label_smoothing=acfg.label_smoothing,
            variance_lambda=acfg.variance_lambda,
            huber_delta=acfg.huber_delta,
            normalize_indices=acfg.normalize_indices,
            cycle_length=acfg.cycle_length,
        )
        return loss
