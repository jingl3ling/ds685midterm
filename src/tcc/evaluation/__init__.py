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

"""TCC Evaluation — PyTorch port of the TF2 evaluation system."""

from tcc.evaluation.algo_loss import AlgoLoss
from tcc.evaluation.classification import Classification
from tcc.evaluation.event_completion import EventCompletion
from tcc.evaluation.few_shot_classification import FewShotClassification
from tcc.evaluation.kendalls_tau import KendallsTau
from tcc.evaluation.registry import get_tasks
from tcc.evaluation.task import Task

__all__ = [
    "AlgoLoss",
    "Classification",
    "EventCompletion",
    "FewShotClassification",
    "KendallsTau",
    "Task",
    "get_tasks",
]
