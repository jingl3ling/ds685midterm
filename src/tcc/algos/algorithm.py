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

"""Base Algorithm class — PyTorch port of TF2 Algorithm(tf.keras.Model)."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from tcc.config import TCCConfig, get_default_config
from tcc.models import get_model


class Algorithm(nn.Module):
    """Base class for all TCC training algorithms.

    Subclasses must implement :meth:`compute_loss`.
    """

    def __init__(
        self,
        cfg: Optional[TCCConfig] = None,
        model: Optional[Dict[str, nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else get_default_config()

        if model is not None:
            self.model = model
        else:
            self.model = get_model()

        # Register sub-modules so that .parameters() picks them up.
        self.cnn = self.model["cnn"]
        self.emb = self.model["emb"]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        data: Dict[str, Any],
        steps: torch.Tensor,
        seq_lens: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Extract frames, pass through CNN + embedder, reshape.

        Args:
            data: dict containing ``"frames"`` key with tensor
                ``[B, T_total, C, H, W]``.
            steps: step indices ``[B, num_steps]``.
            seq_lens: sequence lengths ``[B]``.
            training: whether we are in training mode.

        Returns:
            Embeddings ``[batch_size, num_steps, embedding_dim]``.
        """
        frames = data["frames"]
        batch_size = frames.shape[0]

        num_steps = (
            self.cfg.train.num_frames if training else self.cfg.eval.num_frames
        )

        # CNN feature extraction
        features = self.cnn(frames)

        # Embedding
        embs = self.emb(features, num_frames=num_steps)

        # Reshape to [B, num_steps, D]
        embedding_dim = embs.shape[-1]
        embs = embs.reshape(batch_size, num_steps, embedding_dim)

        return embs

    # ------------------------------------------------------------------
    # Loss (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
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
        """Compute the training loss. Must return a scalar tensor."""
        ...

    # ------------------------------------------------------------------
    # Trainable parameters
    # ------------------------------------------------------------------

    def get_algo_params(self) -> List[torch.nn.Parameter]:
        """Return algorithm-specific parameters (e.g. classifier head).

        Subclasses override this to include extra learnable parameters.
        """
        return []

    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """Return the list of parameters that should be optimized."""
        params: List[torch.nn.Parameter] = []

        # CNN parameters based on train_base mode
        train_base = self.cfg.model.base_model.train_base
        if train_base == "train_all":
            params.extend(self.cnn.parameters())
        elif train_base == "only_bn":
            for m in self.cnn.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    params.extend(m.parameters())
        # 'frozen' — no CNN params

        # Embedder parameters
        if self.cfg.model.train_embedding:
            params.extend(self.emb.parameters())

        # Algorithm-specific parameters
        params.extend(self.get_algo_params())

        return params

    # ------------------------------------------------------------------
    # Training iteration
    # ------------------------------------------------------------------

    def train_one_iter(
        self,
        data: Dict[str, Any],
        steps: torch.Tensor,
        seq_lens: torch.Tensor,
        global_step: int,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training iteration: forward, loss, backward, step.

        Returns:
            Loss value as a Python float.
        """
        optimizer.zero_grad()

        embs = self.forward(data, steps, seq_lens, training=True)

        loss = self.compute_loss(
            embs, steps, seq_lens, global_step, training=True,
            frame_labels=data.get("frame_labels"),
            seq_labels=data.get("seq_labels"),
        )

        # L2 regularization
        l2_reg = self._l2_reg_loss()
        total_loss = loss + l2_reg

        total_loss.backward()
        optimizer.step()

        return total_loss.item()

    def _l2_reg_loss(self) -> torch.Tensor:
        """Compute L2 regularization over all trainable parameters."""
        weight = self.cfg.model.l2_reg_weight
        if weight <= 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        l2 = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.get_trainable_params():
            l2 = l2 + torch.sum(p ** 2)
        return weight * l2
