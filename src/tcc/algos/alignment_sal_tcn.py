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

"""Combined Alignment + SAL + TCN algorithm — PyTorch port."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcc.algos.tcn import TCN
from tcc.algos.sal import sample_batch
from tcc.alignment import compute_alignment_loss
from tcc.config import TCCConfig
from tcc.models import Classifier


class AlignmentSaLTCN(TCN):
    """Joint Alignment + Shuffle-and-Learn + TCN algorithm.

    Weights are configurable via ``cfg.alignment_sal_tcn``:
    - ``alignment_loss_weight``
    - ``sal_loss_weight``
    - TCN weight = 1 - alignment_weight - sal_weight

    Raises ``ValueError`` if weights do not sum to <= 1 or are negative.
    """

    def __init__(
        self,
        cfg: Optional[TCCConfig] = None,
        model=None,
    ) -> None:
        super().__init__(cfg=cfg, model=model)

        joint_cfg = self.cfg.alignment_sal_tcn
        self.alignment_weight = joint_cfg.alignment_loss_weight
        self.sal_weight = joint_cfg.sal_loss_weight
        self.tcn_weight = 1.0 - self.alignment_weight - self.sal_weight

        # Validate weights
        if self.alignment_weight < 0 or self.sal_weight < 0:
            raise ValueError(
                "Loss weights must be non-negative. Got "
                f"alignment={self.alignment_weight}, sal={self.sal_weight}."
            )
        if self.tcn_weight < -1e-7:
            raise ValueError(
                "alignment_loss_weight + sal_loss_weight must be <= 1.0. Got "
                f"{self.alignment_weight} + {self.sal_weight} = "
                f"{self.alignment_weight + self.sal_weight}."
            )

        # SAL classifier
        emb_size = self.cfg.model.conv_embedder.embedding_size
        sal_cfg = self.cfg.sal
        self.sal_classifier = Classifier(
            fc_layers=sal_cfg.fc_layers,
            dropout_rate=sal_cfg.dropout_rate,
            in_features=emb_size * 3,
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
        total_steps = embs.shape[1]
        num_steps = total_steps // 2

        total_loss = torch.tensor(0.0, device=embs.device)

        # --- TCN loss (uses full doubled embeddings) ----------------------
        if self.tcn_weight > 0:
            tcn_loss = super().compute_loss(
                embs, steps, seq_lens, global_step, training,
            )
            total_loss = total_loss + self.tcn_weight * tcn_loss

        # Subsample: use only first half (anchors) for alignment and SAL
        sub_embs = embs[:, :num_steps, :]
        sub_steps = steps[:, :num_steps] if steps.shape[1] >= num_steps else steps

        # --- Alignment loss -----------------------------------------------
        if self.alignment_weight > 0:
            acfg = self.cfg.alignment
            align_loss = compute_alignment_loss(
                embs=sub_embs,
                batch_size=batch_size,
                steps=sub_steps,
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
            total_loss = total_loss + self.alignment_weight * align_loss

        # --- SAL loss -----------------------------------------------------
        if self.sal_weight > 0:
            concat_embs, labels = sample_batch(
                sub_embs, batch_size, num_steps, self.cfg.sal,
            )
            concat_embs = concat_embs.to(embs.device)
            labels = labels.to(embs.device)

            logits = self.sal_classifier(concat_embs)
            target = labels.argmax(dim=-1)
            sal_loss = F.cross_entropy(
                logits, target,
                label_smoothing=self.cfg.sal.label_smoothing,
            )
            total_loss = total_loss + self.sal_weight * sal_loss

        return total_loss

    def get_algo_params(self) -> List[nn.Parameter]:
        return list(self.sal_classifier.parameters())
