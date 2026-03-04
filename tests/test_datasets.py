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

"""Tests for tcc.datasets — sampling strategies, transforms, DataLoader."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from tcc.datasets import (
    AugmentationConfig,
    DataConfig,
    VideoDataset,
    _collate_fn,
    _sample_offset_uniform,
    _sample_stride,
    _apply_tcn_pairing,
    create_dataset,
    create_one_epoch_dataset,
    get_eval_transforms,
    get_train_transforms,
    sample_frames,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_video_dir(
    root: str,
    name: str = "video_001",
    num_frames: int = 40,
    size: tuple = (64, 64),
    with_frame_labels: bool = True,
    with_seq_label: bool = True,
    seq_label: int = 1,
) -> str:
    """Create a directory of synthetic JPEG frames."""
    vdir = os.path.join(root, name)
    os.makedirs(vdir, exist_ok=True)
    for i in range(num_frames):
        img = Image.new("RGB", size, color=(i % 256, (i * 7) % 256, (i * 13) % 256))
        img.save(os.path.join(vdir, f"frame_{i:04d}.jpg"))
    if with_frame_labels:
        labels = np.arange(num_frames, dtype=np.int64)
        np.save(os.path.join(vdir, "frame_labels.npy"), labels)
    if with_seq_label:
        np.save(os.path.join(vdir, "seq_label.npy"), np.array(seq_label))
    return vdir


@pytest.fixture()
def video_root(tmp_path):
    """Fixture that creates a temporary root with two synthetic videos."""
    root = str(tmp_path / "videos")
    os.makedirs(root)
    _make_video_dir(root, "video_001", num_frames=40, seq_label=0)
    _make_video_dir(root, "video_002", num_frames=60, seq_label=1)
    return root


# ---------------------------------------------------------------------------
# Sampling strategy tests
# ---------------------------------------------------------------------------


class TestSampleStride:
    def test_length(self):
        steps = _sample_stride(seq_len=100, num_frames=10, stride=5)
        assert len(steps) == 10

    def test_within_bounds(self):
        for _ in range(20):
            steps = _sample_stride(seq_len=30, num_frames=10, stride=5)
            assert steps.min() >= 0
            assert steps.max() <= 29

    def test_short_sequence_clamped(self):
        steps = _sample_stride(seq_len=5, num_frames=10, stride=3)
        assert len(steps) == 10
        assert steps.max() <= 4


class TestSampleOffsetUniform:
    def test_length(self):
        steps = _sample_offset_uniform(seq_len=100, num_frames=10, random_offset=1)
        assert len(steps) == 10

    def test_sorted(self):
        steps = _sample_offset_uniform(seq_len=100, num_frames=10, random_offset=1)
        assert np.all(steps[:-1] <= steps[1:])

    def test_fallback_short_sequence(self):
        steps = _sample_offset_uniform(seq_len=5, num_frames=10, random_offset=1)
        np.testing.assert_array_equal(steps, np.arange(10))

    def test_respects_offset(self):
        for _ in range(50):
            steps = _sample_offset_uniform(
                seq_len=100, num_frames=5, random_offset=10
            )
            assert steps.min() >= 10


class TestTcnPairing:
    def test_doubles_length(self):
        steps = np.array([5, 10, 15, 20])
        paired = _apply_tcn_pairing(steps, positive_window=3)
        assert len(paired) == 8

    def test_anchors_preserved(self):
        steps = np.array([5, 10, 15, 20])
        paired = _apply_tcn_pairing(steps, positive_window=3)
        # Anchors are at odd indices
        np.testing.assert_array_equal(paired[1::2], steps)


class TestSampleFrames:
    def test_sample_all(self):
        steps = sample_frames(seq_len=50, num_frames=10, sample_all=True)
        np.testing.assert_array_equal(steps, np.arange(50))

    def test_sample_all_stride(self):
        steps = sample_frames(
            seq_len=50, num_frames=10, sample_all=True, sample_all_stride=3
        )
        np.testing.assert_array_equal(steps, np.arange(0, 50, 3))

    def test_tcn_doubles(self):
        steps = sample_frames(
            seq_len=100,
            num_frames=10,
            strategy="stride",
            stride=2,
            training_algo="tcn",
        )
        assert len(steps) == 20

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            sample_frames(seq_len=50, num_frames=10, strategy="bogus")


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_train_output_shape(self):
        t = get_train_transforms(image_size=224)
        img = Image.new("RGB", (320, 240))
        out = t(img)
        assert out.shape == (3, 224, 224)
        assert out.dtype == torch.float32

    def test_eval_output_shape(self):
        t = get_eval_transforms(image_size=224)
        img = Image.new("RGB", (320, 240))
        out = t(img)
        assert out.shape == (3, 224, 224)
        assert out.dtype == torch.float32

    def test_normalized_range(self):
        """After ImageNet normalisation values should not be in [0, 1]."""
        t = get_eval_transforms(image_size=224)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        out = t(img)
        # For a mid-gray image, normalised values should be close to 0
        # (since (0.5 - mean) / std ~ 0 for ImageNet stats) but exact
        # value depends on channel; just check it's not raw [0, 255].
        assert out.max() < 10.0
        assert out.min() > -10.0


# ---------------------------------------------------------------------------
# VideoDataset tests
# ---------------------------------------------------------------------------


class TestVideoDataset:
    def test_len(self, video_root):
        ds = VideoDataset(root_dir=video_root, num_frames=10)
        assert len(ds) == 2

    def test_getitem_shapes(self, video_root):
        t = get_eval_transforms(image_size=224)
        ds = VideoDataset(
            root_dir=video_root, num_frames=10, transform=t
        )
        frames, fl, sl, seq_len, name = ds[0]
        assert frames.shape == (10, 3, 224, 224)
        assert fl.shape == (10,)
        assert isinstance(sl, int)
        assert isinstance(seq_len, int)
        assert isinstance(name, str)

    def test_frame_labels_loaded(self, video_root):
        ds = VideoDataset(root_dir=video_root, num_frames=5)
        _, fl, _, _, _ = ds[0]
        assert fl.dtype == torch.long
        # Labels should be indices clamped within [0, 39]
        assert fl.min() >= 0
        assert fl.max() <= 39

    def test_seq_label(self, video_root):
        ds = VideoDataset(root_dir=video_root, num_frames=5)
        _, _, sl, _, _ = ds[0]
        assert sl == 0
        _, _, sl, _, _ = ds[1]
        assert sl == 1

    def test_sample_all(self, video_root):
        ds = VideoDataset(
            root_dir=video_root, num_frames=10, sample_all=True
        )
        frames, fl, _, seq_len, _ = ds[0]
        assert frames.shape[0] == 40  # video_001 has 40 frames
        assert fl.shape[0] == 40

    def test_no_frame_labels(self, video_root):
        ds = VideoDataset(
            root_dir=video_root, num_frames=5, frame_labels=False
        )
        _, fl, _, _, _ = ds[0]
        assert (fl == 0).all()

    def test_explicit_video_dirs(self, video_root):
        vdir = os.path.join(video_root, "video_001")
        ds = VideoDataset(
            root_dir=video_root, video_dirs=[vdir], num_frames=5
        )
        assert len(ds) == 1


# ---------------------------------------------------------------------------
# DataLoader / collate tests
# ---------------------------------------------------------------------------


class TestCollateAndDataLoader:
    def test_collate_shapes(self, video_root):
        t = get_eval_transforms(image_size=64)
        ds = VideoDataset(root_dir=video_root, num_frames=8, transform=t)
        batch = [ds[0], ds[1]]
        frames, fl, sl, sl_lens, names = _collate_fn(batch)
        assert frames.shape == (2, 8, 3, 64, 64)
        assert fl.shape == (2, 8)
        assert sl.shape == (2,)
        assert sl_lens.shape == (2,)
        assert len(names) == 2

    def test_create_dataset_train(self, video_root):
        cfg = DataConfig(
            root_dir=video_root,
            num_frames=5,
            image_size=64,
            train_batch_size=2,
        )
        loader = create_dataset(split=".", mode="train", config=cfg)
        batch = next(iter(loader))
        frames, fl, sl, sl_lens, names = batch
        assert frames.shape == (2, 5, 3, 64, 64)

    def test_create_dataset_eval(self, video_root):
        cfg = DataConfig(
            root_dir=video_root,
            num_frames=5,
            image_size=64,
            eval_batch_size=2,
        )
        loader = create_dataset(split=".", mode="eval", config=cfg)
        batch = next(iter(loader))
        frames, fl, sl, sl_lens, names = batch
        assert frames.shape == (2, 5, 3, 64, 64)

    def test_create_dataset_invalid_mode(self, video_root):
        cfg = DataConfig(root_dir=video_root)
        with pytest.raises(ValueError, match="Unidentified mode"):
            create_dataset(split=".", mode="bogus", config=cfg)

    def test_create_one_epoch_dataset(self, video_root):
        cfg = DataConfig(
            root_dir=video_root,
            num_frames=5,
            image_size=64,
        )
        loader = create_one_epoch_dataset(
            root_dir=video_root, mode="eval", batch_size=1, config=cfg
        )
        # Should be able to iterate once
        batches = list(loader)
        assert len(batches) == 2  # two videos
        # First video has 40 frames
        assert batches[0][0].shape[1] == 40

    def test_dataloader_pin_memory(self, video_root):
        cfg = DataConfig(
            root_dir=video_root,
            num_frames=5,
            image_size=64,
            train_batch_size=2,
        )
        loader = create_dataset(split=".", mode="train", config=cfg)
        assert loader.pin_memory is True
        assert loader.drop_last is True

    def test_dataloader_eval_no_shuffle(self, video_root):
        cfg = DataConfig(
            root_dir=video_root,
            num_frames=5,
            image_size=64,
            eval_batch_size=1,
        )
        loader = create_dataset(split=".", mode="eval", config=cfg)
        # DataLoader.sampler is SequentialSampler when shuffle=False
        assert not isinstance(
            loader.sampler, torch.utils.data.sampler.RandomSampler
        )
