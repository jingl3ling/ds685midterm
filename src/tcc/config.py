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

"""TCC configuration using Python dataclasses.

Ported from the TF2 EasyDict-based config.py to typed dataclasses with
YAML serialization support.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Dataclass definitions
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training loop parameters."""

    max_iters: int = 150000
    batch_size: int = 2
    num_frames: int = 20
    visualize_interval: int = 200


@dataclass
class EvalConfig:
    """Evaluation parameters."""

    batch_size: int = 2
    num_frames: int = 20
    val_iters: int = 20
    tasks: list[str] = field(
        default_factory=lambda: [
            "algo_loss",
            "classification",
            "kendalls_tau",
            "event_completion",
            "few_shot_classification",
        ]
    )
    frames_per_batch: int = 25
    kendalls_tau_stride: int = 5
    kendalls_tau_distance: str = "sqeuclidean"
    classification_fractions: list[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0]
    )
    few_shot_num_labeled: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    few_shot_num_episodes: int = 50


@dataclass
class BaseModelConfig:
    """Base feature-extractor network."""

    network: str = "resnet50"
    layer: str = "conv4_block3_out"
    train_base: str = "only_bn"


@dataclass
class ConvEmbedderConfig:
    """Convolutional embedder head."""

    embedding_size: int = 128
    num_context_steps: int = 2
    conv_layers: list[tuple[int, int, bool]] = field(
        default_factory=lambda: [(256, 3, True), (256, 3, True)]
    )
    fc_layers: list[tuple[int, bool]] = field(
        default_factory=lambda: [(256, True), (256, True)]
    )
    capacity_scalar: int = 2
    flatten_method: str = "max_pool"
    base_dropout_rate: float = 0.0
    base_dropout_spatial: bool = False
    fc_dropout_rate: float = 0.1
    dropout_rate: float = 0.1
    pooling: str = "max"
    l2_normalize: bool = False
    use_bn: bool = True


@dataclass
class ConvGRUEmbedderConfig:
    """Conv + GRU embedder head."""

    conv_layers: list[tuple[int, int, bool]] = field(
        default_factory=lambda: [(512, 3, True), (512, 3, True)]
    )
    gru_layers: list[int] = field(default_factory=lambda: [128])
    dropout_rate: float = 0.0
    use_bn: bool = True


@dataclass
class VGGMConfig:
    """VGG-M model options."""

    use_bn: bool = True


@dataclass
class ModelConfig:
    """Top-level model configuration."""

    embedder_type: str = "conv"
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)
    conv_embedder: ConvEmbedderConfig = field(default_factory=ConvEmbedderConfig)
    convgru_embedder: ConvGRUEmbedderConfig = field(
        default_factory=ConvGRUEmbedderConfig
    )
    vggm: VGGMConfig = field(default_factory=VGGMConfig)
    train_embedding: bool = True
    l2_reg_weight: float = 0.00001
    resnet_pretrained_weights: str = (
        "/tmp/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )


@dataclass
class AlignmentConfig:
    """Temporal cycle-consistency alignment loss parameters."""

    loss_type: str = "regression_mse_var"
    similarity_type: str = "l2"
    temperature: float = 0.1
    label_smoothing: float = 0.1
    cycle_length: int = 2
    stochastic_matching: bool = False
    variance_lambda: float = 0.001
    huber_delta: float = 0.1
    normalize_indices: bool = True
    fraction: float = 1.0


@dataclass
class SALConfig:
    """Shuffle and Learn parameters."""

    dropout_rate: float = 0.0
    fc_layers: list[tuple[int, bool]] = field(
        default_factory=lambda: [(128, True), (64, True), (2, False)]
    )
    shuffle_fraction: float = 0.75
    num_samples: int = 8
    label_smoothing: float = 0.0


@dataclass
class AlignmentSalTcnConfig:
    """Joint alignment + SAL + TCN loss weights."""

    alignment_loss_weight: float = 0.33
    sal_loss_weight: float = 0.33


@dataclass
class ClassificationConfig:
    """Supervised per-frame classification parameters."""

    label_smoothing: float = 0.0
    dropout_rate: float = 0.0


@dataclass
class TCNConfig:
    """Time Contrastive Network parameters."""

    positive_window: int = 5
    reg_lambda: float = 0.002


@dataclass
class LRConfig:
    """Learning-rate schedule."""

    initial_lr: float = 0.0001
    decay_type: str = "fixed"
    exp_decay_rate: float = 0.97
    exp_decay_steps: int = 1000
    manual_lr_step_boundaries: list[int] = field(
        default_factory=lambda: [5000, 10000]
    )
    manual_lr_decay_rate: float = 0.1
    num_warmup_steps: int = 0


@dataclass
class OptimizerConfig:
    """Optimizer parameters."""

    type: str = "adam"
    lr: LRConfig = field(default_factory=LRConfig)
    weight_decay: float = 0.0


@dataclass
class DataConfig:
    """Data loading and sampling parameters."""

    sampling_strategy: str = "offset_uniform"
    stride: int = 16
    num_steps: int = 2
    frame_stride: int = 15
    image_size: int = 224
    augment: bool = True
    shuffle_queue_size: int = 0
    num_prefetch_batches: int = 1
    random_offset: int = 1
    frame_labels: bool = True
    per_dataset_fraction: float = 1.0
    per_class: bool = False
    sample_all_stride: int = 1


@dataclass
class AugmentationConfig:
    """Data augmentation parameters."""

    random_flip: bool = True
    random_crop: bool = False
    brightness: bool = True
    brightness_max_delta: float = 32.0 / 255
    contrast: bool = True
    contrast_lower: float = 0.5
    contrast_upper: float = 1.5
    hue: bool = False
    hue_max_delta: float = 0.2
    saturation: bool = False
    saturation_lower: float = 0.5
    saturation_upper: float = 1.5


@dataclass
class LoggingConfig:
    """Logging parameters."""

    report_interval: int = 100


@dataclass
class CheckpointConfig:
    """Checkpointing parameters."""

    save_interval: int = 1000


@dataclass
class TCCConfig:
    """Root configuration for a TCC experiment."""

    logdir: str = "/tmp/alignment_logs/"
    datasets: list[str] = field(default_factory=lambda: ["pouring"])
    path_to_tfrecords: str = "/tmp/%s_tfrecords/"
    training_algo: str = "alignment"

    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    sal: SALConfig = field(default_factory=SALConfig)
    alignment_sal_tcn: AlignmentSalTcnConfig = field(
        default_factory=AlignmentSalTcnConfig
    )
    classification: ClassificationConfig = field(
        default_factory=ClassificationConfig
    )
    tcn: TCNConfig = field(default_factory=TCNConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


# ---------------------------------------------------------------------------
# Helpers: dataclass <-> dict conversion
# ---------------------------------------------------------------------------

# Registry mapping section names to their dataclass types, used when
# converting a plain dict back into the typed hierarchy.
_SECTION_TYPES: dict[str, type] = {
    "train": TrainConfig,
    "eval": EvalConfig,
    "model": ModelConfig,
    "alignment": AlignmentConfig,
    "sal": SALConfig,
    "alignment_sal_tcn": AlignmentSalTcnConfig,
    "classification": ClassificationConfig,
    "tcn": TCNConfig,
    "optimizer": OptimizerConfig,
    "data": DataConfig,
    "augmentation": AugmentationConfig,
    "logging": LoggingConfig,
    "checkpoint": CheckpointConfig,
}

_MODEL_SECTION_TYPES: dict[str, type] = {
    "base_model": BaseModelConfig,
    "conv_embedder": ConvEmbedderConfig,
    "convgru_embedder": ConvGRUEmbedderConfig,
    "vggm": VGGMConfig,
}

_OPTIMIZER_SECTION_TYPES: dict[str, type] = {
    "lr": LRConfig,
}


def config_to_dict(cfg: TCCConfig) -> dict[str, Any]:
    """Convert a ``TCCConfig`` instance to a plain nested dictionary."""
    return asdict(cfg)


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively instantiate *cls* from *data*, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def dict_to_config(data: dict[str, Any]) -> TCCConfig:
    """Convert a plain nested dictionary to a ``TCCConfig`` instance.

    Unknown keys at any level are silently dropped so that partial YAML
    overrides work seamlessly.
    """
    data = copy.deepcopy(data)

    # Recursively reconstruct nested model sub-sections.
    if "model" in data and isinstance(data["model"], dict):
        for sub_key, sub_cls in _MODEL_SECTION_TYPES.items():
            if sub_key in data["model"] and isinstance(data["model"][sub_key], dict):
                data["model"][sub_key] = _dict_to_dataclass(
                    sub_cls, data["model"][sub_key]
                )

    # Recursively reconstruct nested optimizer sub-sections.
    if "optimizer" in data and isinstance(data["optimizer"], dict):
        for sub_key, sub_cls in _OPTIMIZER_SECTION_TYPES.items():
            if sub_key in data["optimizer"] and isinstance(
                data["optimizer"][sub_key], dict
            ):
                data["optimizer"][sub_key] = _dict_to_dataclass(
                    sub_cls, data["optimizer"][sub_key]
                )

    # Reconstruct top-level sections.
    for section, cls in _SECTION_TYPES.items():
        if section in data and isinstance(data[section], dict):
            data[section] = _dict_to_dataclass(cls, data[section])

    return _dict_to_dataclass(TCCConfig, data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_default_config() -> TCCConfig:
    """Return a ``TCCConfig`` populated with all default values."""
    return TCCConfig()


def load_config(yaml_path: str) -> TCCConfig:
    """Load a YAML file and merge its values with the defaults.

    Parameters
    ----------
    yaml_path:
        Path to a YAML configuration file.  Only the keys present in the
        file will override the corresponding defaults; everything else
        keeps the default value.

    Returns
    -------
    TCCConfig
        Fully-populated configuration.
    """
    with open(yaml_path, "r") as fh:
        overrides = yaml.safe_load(fh) or {}

    base = config_to_dict(get_default_config())
    merged = _deep_merge(base, overrides)
    return dict_to_config(merged)


def save_config(config: TCCConfig, path: str) -> None:
    """Serialize *config* to a YAML file at *path*.

    Parent directories are created automatically if they do not exist.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        yaml.dump(
            config_to_dict(config),
            fh,
            default_flow_style=False,
            sort_keys=True,
        )
