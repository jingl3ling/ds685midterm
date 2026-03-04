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

"""Algorithm registry and factory function."""

from __future__ import annotations

from typing import Optional

from tcc.algos.algorithm import Algorithm
from tcc.algos.alignment import Alignment
from tcc.algos.alignment_sal_tcn import AlignmentSaLTCN
from tcc.algos.classification import Classification
from tcc.algos.sal import SAL
from tcc.algos.tcn import TCN
from tcc.config import TCCConfig

_ALGO_REGISTRY: dict[str, type] = {
    "alignment": Alignment,
    "sal": SAL,
    "tcn": TCN,
    "classification": Classification,
    "alignment_sal_tcn": AlignmentSaLTCN,
}


def get_algo(
    algo_name: str,
    cfg: Optional[TCCConfig] = None,
    **kwargs,
) -> Algorithm:
    """Factory that instantiates an algorithm by name.

    Args:
        algo_name: one of ``"alignment"``, ``"sal"``, ``"tcn"``,
            ``"classification"``, ``"alignment_sal_tcn"``.
        cfg: TCC configuration. Uses defaults if ``None``.
        **kwargs: extra keyword arguments forwarded to the algorithm
            constructor (e.g. ``num_classes`` for classification).

    Returns:
        An :class:`Algorithm` instance.

    Raises:
        ValueError: if *algo_name* is not in the registry.
    """
    if algo_name not in _ALGO_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Available: {sorted(_ALGO_REGISTRY.keys())}."
        )

    cls = _ALGO_REGISTRY[algo_name]
    return cls(cfg=cfg, **kwargs)
