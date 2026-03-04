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

"""Supervised per-frame Classification algorithm — PyTorch port."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcc.algos.algorithm import Algorithm
from tcc.config import TCCConfig
from tcc.models import Classifier


class Classification(Algorithm):
    """Supervised per-frame classification algorithm.

    Uses a ``Classifier`` head to predict per-frame labels.
    """

    def __init__(
        self,
        cfg: Optional[TCCConfig] = None,
        model=None,
        num_classes: int = 10,
    ) -> None:
        super().__init__(cfg=cfg, model=model)

        emb_size = self.cfg.model.conv_embedder.embedding_size
        cls_cfg = self.cfg.classification

        self.num_classes = num_classes
        self.classifier = Classifier(
            fc_layers=[(num_classes, False)],
            dropout_rate=cls_cfg.dropout_rate,
            in_features=emb_size,
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
        if frame_labels is None:
            raise ValueError(
                "Classification algorithm requires frame_labels."
            )

        batch_size, num_steps, emb_dim = embs.shape

        # Flatten to [B*T, D]
        flat_embs = embs.reshape(batch_size * num_steps, emb_dim)
        logits = self.classifier(flat_embs)

        # Flatten labels to [B*T]
        flat_labels = frame_labels.reshape(batch_size * num_steps)

        loss = F.cross_entropy(
            logits,
            flat_labels,
            label_smoothing=self.cfg.classification.label_smoothing,
        )
        return loss

    def get_algo_params(self) -> List[nn.Parameter]:
        return list(self.classifier.parameters())
