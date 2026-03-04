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

"""Task registry and factory for the TCC evaluation system."""

from __future__ import annotations

from typing import Dict, List, Tuple

from tcc.evaluation.algo_loss import AlgoLoss
from tcc.evaluation.classification import Classification
from tcc.evaluation.event_completion import EventCompletion
from tcc.evaluation.few_shot_classification import FewShotClassification
from tcc.evaluation.kendalls_tau import KendallsTau
from tcc.evaluation.task import Task

TASK_NAME_TO_CLASS: Dict[str, type] = {
    "algo_loss": AlgoLoss,
    "classification": Classification,
    "kendalls_tau": KendallsTau,
    "event_completion": EventCompletion,
    "few_shot_classification": FewShotClassification,
}


def get_tasks(
    task_names: List[str],
) -> Tuple[Dict[str, Task], Dict[str, Task]]:
    """Instantiate evaluation tasks and split them by type.

    Parameters
    ----------
    task_names:
        List of task name strings (must be keys in
        :data:`TASK_NAME_TO_CLASS`).

    Returns
    -------
    tuple of (iterator_tasks, embedding_tasks):
        Two dicts mapping task name to :class:`Task` instance.
        ``iterator_tasks`` contains tasks with ``downstream_task=False``
        (they need data iterators).  ``embedding_tasks`` contains tasks
        with ``downstream_task=True`` (they need pre-extracted embeddings).

    Raises
    ------
    ValueError:
        If a task name is not recognized.
    """
    iterator_tasks: Dict[str, Task] = {}
    embedding_tasks: Dict[str, Task] = {}

    for name in task_names:
        if name not in TASK_NAME_TO_CLASS:
            raise ValueError(
                f"Unknown task '{name}'. "
                f"Available: {sorted(TASK_NAME_TO_CLASS.keys())}."
            )
        task = TASK_NAME_TO_CLASS[name]()
        if task.downstream_task:
            embedding_tasks[name] = task
        else:
            iterator_tasks[name] = task

    return iterator_tasks, embedding_tasks
