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

"""Base Task class for TCC evaluation — PyTorch port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Task(ABC):
    """Base class for all evaluation tasks.

    Parameters
    ----------
    downstream_task:
        If ``True``, the task operates on pre-extracted embeddings
        (via :meth:`evaluate_embeddings`).  If ``False``, it operates
        on raw data iterators (via :meth:`evaluate_iterators`).
    """

    def __init__(self, downstream_task: bool = True) -> None:
        self.downstream_task = downstream_task

    def evaluate(
        self,
        algo: Any,
        global_step: int,
        cfg: Any,
        iterators: Optional[Dict[str, Any]] = None,
        embeddings_dataset: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Dispatch to the appropriate evaluation method.

        Parameters
        ----------
        algo:
            The algorithm (nn.Module) being evaluated.
        global_step:
            Current training step (plain int).
        cfg:
            TCCConfig instance.
        iterators:
            Dict of split name -> DataLoader, used by iterator-based tasks.
        embeddings_dataset:
            Dict of split name -> dict with 'embeddings', 'labels', etc.,
            used by embedding-based (downstream) tasks.
        writer:
            Optional TensorBoard SummaryWriter for logging.

        Returns
        -------
        dict:
            Mapping of metric name to scalar value.
        """
        if self.downstream_task:
            if embeddings_dataset is None:
                raise ValueError(
                    f"Task {self.__class__.__name__} requires "
                    "embeddings_dataset but got None."
                )
            return self.evaluate_embeddings(
                algo, global_step, cfg, embeddings_dataset, writer=writer
            )
        else:
            if iterators is None:
                raise ValueError(
                    f"Task {self.__class__.__name__} requires "
                    "iterators but got None."
                )
            return self.evaluate_iterators(
                algo, global_step, cfg, iterators, writer=writer
            )

    @abstractmethod
    def evaluate_embeddings(
        self,
        algo: Any,
        global_step: int,
        cfg: Any,
        datasets: Dict[str, Any],
        writer: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Evaluate using pre-extracted embeddings."""
        ...

    @abstractmethod
    def evaluate_iterators(
        self,
        algo: Any,
        global_step: int,
        cfg: Any,
        iterators: Dict[str, Any],
        writer: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Evaluate using raw data iterators."""
        ...
