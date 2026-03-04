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

"""TCC algorithm implementations — PyTorch port."""

from tcc.algos.algorithm import Algorithm
from tcc.algos.alignment import Alignment
from tcc.algos.alignment_sal_tcn import AlignmentSaLTCN
from tcc.algos.classification import Classification
from tcc.algos.registry import get_algo
from tcc.algos.sal import SAL
from tcc.algos.tcn import TCN

__all__ = [
    "Algorithm",
    "Alignment",
    "AlignmentSaLTCN",
    "Classification",
    "SAL",
    "TCN",
    "get_algo",
]
