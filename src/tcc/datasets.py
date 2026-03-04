# coding=utf-8
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

"""Dataset loading for TCC — PyTorch port.

Supports loading video sequences stored as directories of extracted frame
images (JPEG/PNG). Provides configurable frame sampling strategies and
data augmentation pipelines matching the original TF2 implementation.

Supported datasets: Pouring, Penn Action (and any dataset following the
same directory layout).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Default configuration (mirrors CONFIG.DATA / CONFIG.AUGMENTATION / etc.)
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """Data-augmentation knobs corresponding to CONFIG.AUGMENTATION."""

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
class DataConfig:
    """Dataset-level configuration matching CONFIG.DATA."""

    # Paths -----------------------------------------------------------------
    # Root directory containing per-video sub-directories of frame images.
    root_dir: str = ""
    # Optional list of dataset names (sub-folders under root_dir).
    datasets: List[str] = field(default_factory=lambda: ["pouring"])

    # Frame sampling --------------------------------------------------------
    sampling_strategy: Literal["stride", "offset_uniform"] = "offset_uniform"
    num_frames: int = 20
    stride: int = 16
    random_offset: int = 1
    num_steps: int = 2
    frame_stride: int = 15
    sample_all_stride: int = 1
    frame_labels: bool = True

    # Image -----------------------------------------------------------------
    image_size: int = 224

    # Training algorithm ----------------------------------------------------
    training_algo: str = "alignment"
    tcn_positive_window: int = 5

    # Train / eval batch sizes ---------------------------------------------
    train_batch_size: int = 2
    eval_batch_size: int = 2

    # Augmentation ----------------------------------------------------------
    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig
    )


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 224,
    aug: Optional[AugmentationConfig] = None,
) -> transforms.Compose:
    """Build the training transform pipeline.

    Mirrors the TF2 ``preprocess_input(..., augment=True)`` behaviour:
    random crop, random flip, colour jitter, resize, ImageNet normalisation.
    """
    if aug is None:
        aug = AugmentationConfig()

    t: list = []

    # Random crop
    if aug.random_crop:
        t.append(
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0), ratio=(1.0, 1.0)
            )
        )
    else:
        t.append(transforms.Resize((image_size, image_size)))

    # Random horizontal flip
    if aug.random_flip:
        t.append(transforms.RandomHorizontalFlip(p=0.5))

    # Colour jitter (brightness, contrast, saturation, hue)
    brightness = aug.brightness_max_delta if aug.brightness else 0.0
    contrast = (
        (aug.contrast_lower, aug.contrast_upper) if aug.contrast else None
    )
    saturation = (
        (aug.saturation_lower, aug.saturation_upper) if aug.saturation else None
    )
    hue = aug.hue_max_delta if aug.hue else 0.0

    # Only add ColorJitter if at least one param is non-trivial
    if brightness or contrast or saturation or hue:
        t.append(
            transforms.ColorJitter(
                brightness=brightness if brightness else 0,
                contrast=list(contrast) if contrast else 0,
                saturation=list(saturation) if saturation else 0,
                hue=hue if hue else 0,
            )
        )

    t.append(transforms.ToTensor())  # [0,255] -> [0,1], HWC -> CHW
    t.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(t)


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Build the eval transform pipeline (deterministic)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Frame sampling strategies
# ---------------------------------------------------------------------------

def _sample_stride(
    seq_len: int,
    num_frames: int,
    stride: int,
) -> np.ndarray:
    """Fixed-stride sampling with a random offset.

    A random offset is drawn from ``[0, max(1, seq_len - stride * num_frames))``.
    Indices that exceed ``seq_len - 1`` are clamped.
    """
    max_offset = max(1, seq_len - stride * num_frames)
    offset = random.randint(0, max_offset - 1)
    steps = np.arange(offset, offset + num_frames * stride + 1, stride)
    steps = steps[:num_frames]
    steps = np.minimum(steps, seq_len - 1)
    return steps


def _sample_offset_uniform(
    seq_len: int,
    num_frames: int,
    random_offset: int = 1,
) -> np.ndarray:
    """Random-offset then uniform sampling.

    If there are enough frames after the offset, ``num_frames`` are sampled
    uniformly (without replacement) from ``[offset, seq_len)``.  Otherwise
    fall back to ``range(num_frames)``.
    """
    if num_frames <= seq_len - random_offset:
        pool = np.arange(random_offset, seq_len)
        chosen = np.sort(
            np.random.choice(pool, size=num_frames, replace=False)
        )
        return chosen
    else:
        return np.arange(num_frames, dtype=np.int64)


def _apply_tcn_pairing(
    steps: np.ndarray,
    positive_window: int,
) -> np.ndarray:
    """Double the sampled steps for TCN anchor/positive pairs.

    For each step, a positive is sampled from
    ``[step - positive_window, step)``.  The returned array interleaves
    ``[pos_0, anchor_0, pos_1, anchor_1, ...]``.
    """
    pos_steps = np.array(
        [
            np.random.randint(max(0, s - positive_window), max(1, s))
            for s in steps
        ]
    )
    paired = np.stack([pos_steps, steps], axis=1).reshape(-1)
    return paired


def sample_frames(
    seq_len: int,
    num_frames: int,
    strategy: str = "offset_uniform",
    stride: int = 16,
    random_offset: int = 1,
    sample_all: bool = False,
    sample_all_stride: int = 1,
    training_algo: str = "alignment",
    tcn_positive_window: int = 5,
) -> np.ndarray:
    """High-level frame sampling dispatcher.

    Returns an array of integer frame indices to load.
    """
    if sample_all:
        return np.arange(0, seq_len, sample_all_stride)

    if strategy == "stride":
        steps = _sample_stride(seq_len, num_frames, stride)
    elif strategy == "offset_uniform":
        steps = _sample_offset_uniform(seq_len, num_frames, random_offset)
    else:
        raise ValueError(
            f"Unknown sampling strategy '{strategy}'. "
            "Supported: 'stride', 'offset_uniform'."
        )

    if "tcn" in training_algo:
        steps = _apply_tcn_pairing(steps, tcn_positive_window)

    return steps


# ---------------------------------------------------------------------------
# VideoDataset
# ---------------------------------------------------------------------------

def _natural_sort_key(s: str):
    """Sort strings with embedded integers in natural order."""
    import re
    return [
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r"(\d+)", s)
    ]


class VideoDataset(Dataset):
    """PyTorch Dataset that loads video sequences stored as frame images.

    Each video is a directory containing frame images (JPEG or PNG) in sorted
    order.  The directory layout is::

        root_dir/
            video_001/
                frame_0000.jpg
                frame_0001.jpg
                ...
            video_002/
                ...

    Optionally, per-frame labels can be supplied via a ``labels.npy`` or
    ``frame_labels.npy`` file inside each video directory.  Sequence-level
    labels can be supplied via a ``seq_label.npy`` file.

    Parameters
    ----------
    root_dir:
        Path to the root directory containing video sub-directories.
    video_dirs:
        Explicit list of video directory paths. If ``None``, all
        sub-directories under *root_dir* are used.
    num_frames:
        Number of frames to sample per video.
    sampling_strategy:
        One of ``'stride'``, ``'offset_uniform'``.
    stride:
        Stride used when ``sampling_strategy='stride'``.
    random_offset:
        Minimum offset for ``'offset_uniform'`` strategy.
    sample_all:
        If ``True``, return all frames (for evaluation).
    sample_all_stride:
        Sub-sampling stride when ``sample_all=True``.
    training_algo:
        Algorithm name; if it contains ``'tcn'``, TCN double-sampling
        is applied.
    tcn_positive_window:
        Window size for TCN positive pair sampling.
    transform:
        A ``torchvision.transforms`` pipeline applied to each frame.
    frame_labels:
        Whether to load per-frame labels.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root_dir: str,
        video_dirs: Optional[List[str]] = None,
        num_frames: int = 20,
        sampling_strategy: str = "offset_uniform",
        stride: int = 16,
        random_offset: int = 1,
        sample_all: bool = False,
        sample_all_stride: int = 1,
        training_algo: str = "alignment",
        tcn_positive_window: int = 5,
        transform: Optional[transforms.Compose] = None,
        frame_labels: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy
        self.stride = stride
        self.random_offset = random_offset
        self.sample_all = sample_all
        self.sample_all_stride = sample_all_stride
        self.training_algo = training_algo
        self.tcn_positive_window = tcn_positive_window
        self.transform = transform
        self.frame_labels_enabled = frame_labels

        # Discover video directories -----------------------------------------
        if video_dirs is not None:
            self.video_dirs = sorted(video_dirs)
        else:
            self.video_dirs = sorted(
                [
                    os.path.join(root_dir, d)
                    for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))
                ]
            )

        # Pre-scan frame paths per video for faster __getitem__ ---------------
        self.frame_paths: List[List[str]] = []
        self.seq_labels: List[int] = []
        self._frame_labels_cache: List[Optional[np.ndarray]] = []

        for vdir in self.video_dirs:
            frames = sorted(
                [
                    os.path.join(vdir, f)
                    for f in os.listdir(vdir)
                    if os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS
                ],
                key=lambda p: _natural_sort_key(os.path.basename(p)),
            )
            self.frame_paths.append(frames)

            # Sequence-level label
            seq_label_path = os.path.join(vdir, "seq_label.npy")
            if os.path.isfile(seq_label_path):
                self.seq_labels.append(int(np.load(seq_label_path)))
            else:
                self.seq_labels.append(0)

            # Per-frame labels
            if frame_labels:
                for candidate in ("frame_labels.npy", "labels.npy"):
                    p = os.path.join(vdir, candidate)
                    if os.path.isfile(p):
                        self._frame_labels_cache.append(np.load(p))
                        break
                else:
                    self._frame_labels_cache.append(None)
            else:
                self._frame_labels_cache.append(None)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.video_dirs)

    # ------------------------------------------------------------------
    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, str]:
        """Return a sampled clip from video *idx*.

        Returns
        -------
        frames : torch.Tensor
            Shape ``[T, 3, H, W]`` (float32, normalised).
        frame_labels : torch.Tensor
            Shape ``[T]`` (int64). Zeros if labels are unavailable.
        seq_label : int
            Sequence-level class label.
        seq_len : int
            Total number of frames in the original video.
        name : str
            Basename of the video directory.
        """
        all_frame_paths = self.frame_paths[idx]
        seq_len = len(all_frame_paths)
        name = os.path.basename(self.video_dirs[idx])

        # --- Sample frame indices ---
        indices = sample_frames(
            seq_len=seq_len,
            num_frames=self.num_frames,
            strategy=self.sampling_strategy,
            stride=self.stride,
            random_offset=self.random_offset,
            sample_all=self.sample_all,
            sample_all_stride=self.sample_all_stride,
            training_algo=self.training_algo,
            tcn_positive_window=self.tcn_positive_window,
        )
        # Clamp into valid range
        indices = np.clip(indices, 0, seq_len - 1)

        # --- Load and transform frames ---
        frame_tensors = []
        for i in indices:
            img = Image.open(all_frame_paths[int(i)]).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            frame_tensors.append(img)
        frames = torch.stack(frame_tensors, dim=0)  # [T, 3, H, W]

        # --- Frame labels ---
        cached = self._frame_labels_cache[idx]
        if cached is not None:
            fl = cached[indices.astype(np.int64)]
            frame_labels = torch.from_numpy(fl.copy()).long()
        else:
            frame_labels = torch.zeros(len(indices), dtype=torch.long)

        seq_label = self.seq_labels[idx]

        return frames, frame_labels, seq_label, seq_len, name


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, int, str]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Custom collate that handles variable-length names.

    Returns
    -------
    frames : [B, T, 3, H, W]
    frame_labels : [B, T]
    seq_labels : [B]
    seq_lens : [B]
    names : list[str] of length B
    """
    frames = torch.stack([b[0] for b in batch])
    frame_labels = torch.stack([b[1] for b in batch])
    seq_labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    seq_lens = torch.tensor([b[3] for b in batch], dtype=torch.long)
    names = [b[4] for b in batch]
    return frames, frame_labels, seq_labels, seq_lens, names


def create_dataset(
    split: str,
    mode: str,
    batch_size: Optional[int] = None,
    config: Optional[DataConfig] = None,
) -> DataLoader:
    """Create a :class:`DataLoader` for training or evaluation.

    Parameters
    ----------
    split:
        Dataset split identifier, e.g. ``'train'``, ``'val'``, ``'test'``.
    mode:
        ``'train'`` or ``'eval'``.
    batch_size:
        Override batch size. If ``None``, uses the config default.
    config:
        A :class:`DataConfig` instance. If ``None``, default values are used.

    Returns
    -------
    DataLoader
    """
    if config is None:
        config = DataConfig()

    if mode not in ("train", "eval"):
        raise ValueError(
            f"Unidentified mode: '{mode}'. Use either 'train' or 'eval'."
        )

    is_train = mode == "train"
    if batch_size is None:
        batch_size = (
            config.train_batch_size if is_train else config.eval_batch_size
        )

    # Build transforms
    if is_train:
        transform = get_train_transforms(
            config.image_size, config.augmentation
        )
    else:
        transform = get_eval_transforms(config.image_size)

    # Discover video directories for the given split.
    # Convention: root_dir/split/ or root_dir/ with a split-specific manifest.
    split_dir = os.path.join(config.root_dir, split)
    if os.path.isdir(split_dir):
        root = split_dir
    else:
        root = config.root_dir

    dataset = VideoDataset(
        root_dir=root,
        num_frames=config.num_frames,
        sampling_strategy=config.sampling_strategy,
        stride=config.stride,
        random_offset=config.random_offset,
        sample_all=False,
        training_algo=config.training_algo,
        tcn_positive_window=config.tcn_positive_window,
        transform=transform,
        frame_labels=config.frame_labels,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=4,
        pin_memory=True,
        collate_fn=_collate_fn,
    )


def create_one_epoch_dataset(
    root_dir: str,
    mode: str = "eval",
    batch_size: int = 1,
    config: Optional[DataConfig] = None,
) -> DataLoader:
    """Create a :class:`DataLoader` that iterates once over every video.

    All frames are returned (``sample_all=True``) at the configured stride.
    Useful for evaluation / embedding extraction.
    """
    if config is None:
        config = DataConfig()

    is_train = mode == "train"
    if is_train:
        transform = get_train_transforms(
            config.image_size, config.augmentation
        )
    else:
        transform = get_eval_transforms(config.image_size)

    dataset = VideoDataset(
        root_dir=root_dir,
        num_frames=config.num_frames,
        sampling_strategy=config.sampling_strategy,
        stride=config.stride,
        random_offset=config.random_offset,
        sample_all=True,
        sample_all_stride=config.sample_all_stride,
        training_algo=config.training_algo,
        tcn_positive_window=config.tcn_positive_window,
        transform=transform,
        frame_labels=config.frame_labels,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=_collate_fn,
    )
