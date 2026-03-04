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

"""Time-Contrastive Network (TCN) algorithm — PyTorch port."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from tcc.algos.algorithm import Algorithm


class TCN(Algorithm):
    """Time-Contrastive Network using N-pairs loss.

    Overrides ``forward`` to handle doubled frames (anchor + positive pairs
    sampled from a temporal window).
    """

    def forward(
        self,
        data: Dict[str, Any],
        steps: torch.Tensor,
        seq_lens: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Forward pass that handles doubled frames for anchor/positive.

        The input ``data["frames"]`` has shape ``[B, 2*T, C, H, W]`` where
        the first T frames are anchors and the last T frames are positives.
        """
        frames = data["frames"]
        batch_size = frames.shape[0]

        num_steps = (
            self.cfg.train.num_frames if training else self.cfg.eval.num_frames
        )

        # Features
        features = self.cnn(frames)

        # The embedder sees 2*num_steps frames
        embs = self.emb(features, num_frames=2 * num_steps)

        embedding_dim = embs.shape[-1]
        embs = embs.reshape(batch_size, 2 * num_steps, embedding_dim)

        return embs

    # ------------------------------------------------------------------
    # N-pairs loss
    # ------------------------------------------------------------------

    @staticmethod
    def _npairs_loss(
        labels: torch.Tensor,
        embeddings_anchor: torch.Tensor,
        embeddings_positive: torch.Tensor,
        reg_lambda: float,
    ) -> torch.Tensor:
        """N-pairs metric loss with L2 regularization.

        Args:
            labels: integer labels ``[N]`` (used to form the identity target).
            embeddings_anchor: ``[N, D]`` anchor embeddings.
            embeddings_positive: ``[N, D]`` positive embeddings.
            reg_lambda: L2 regularization weight.

        Returns:
            Scalar loss.
        """
        # Similarity matrix: [N, N]
        similarity = torch.matmul(
            embeddings_anchor, embeddings_positive.t()
        )

        # Target: identity (each anchor matches its own positive)
        n = similarity.shape[0]
        target = torch.arange(n, device=similarity.device)

        # Softmax cross-entropy on the similarity matrix
        ce_loss = F.cross_entropy(similarity, target)

        # L2 regularization on embeddings
        l2_loss = reg_lambda * (
            torch.mean(torch.sum(embeddings_anchor ** 2, dim=1))
            + torch.mean(torch.sum(embeddings_positive ** 2, dim=1))
        )

        return ce_loss + l2_loss

    def single_sequence_loss(
        self,
        embs: torch.Tensor,
        num_steps: int,
        reg_lambda: float,
    ) -> torch.Tensor:
        """Compute N-pairs loss for a single sequence.

        Args:
            embs: ``[2 * num_steps, D]`` — first half are anchors,
                second half are positives.
            num_steps: number of anchor/positive pairs.
            reg_lambda: L2 regularization weight.

        Returns:
            Scalar loss.
        """
        anchors = embs[:num_steps]
        positives = embs[num_steps:]
        labels = torch.arange(num_steps, device=embs.device)

        return self._npairs_loss(labels, anchors, positives, reg_lambda)

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
        num_steps = embs.shape[1] // 2
        reg_lambda = self.cfg.tcn.reg_lambda

        losses = []
        for i in range(batch_size):
            losses.append(
                self.single_sequence_loss(embs[i], num_steps, reg_lambda)
            )

        return torch.mean(torch.stack(losses))
