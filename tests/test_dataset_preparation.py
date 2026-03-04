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

"""Tests for tcc.dataset_preparation utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from tcc.dataset_preparation.dataset_utils import (
    label_timestamps,
    merge_annotations,
    save_frames,
    video_to_frames,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_video(
    path: str,
    num_frames: int = 30,
    width: int = 64,
    height: int = 48,
    fps: float = 15.0,
) -> str:
    """Write a synthetic video file using cv2.VideoWriter."""
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        # Create a frame with a unique colour per frame
        frame = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------------------
# Tests: video_to_frames
# ---------------------------------------------------------------------------

class TestVideoToFrames:
    def test_basic_extraction(self, tmp_path):
        video_path = str(tmp_path / "test.avi")
        _create_synthetic_video(video_path, num_frames=30, fps=15.0)

        frames, timestamps, native_fps = video_to_frames(
            video_path, resize=False
        )

        assert len(frames) == 30
        assert len(timestamps) == 30
        assert native_fps == pytest.approx(15.0, abs=1.0)
        # Frames should be RGB uint8
        assert frames[0].dtype == np.uint8
        assert frames[0].shape == (48, 64, 3)

    def test_resize(self, tmp_path):
        video_path = str(tmp_path / "test.avi")
        _create_synthetic_video(video_path, num_frames=10, width=100, height=80)

        frames, _, _ = video_to_frames(
            video_path, resize=True, width=224, height=224
        )

        assert frames[0].shape == (224, 224, 3)

    def test_fps_subsampling(self, tmp_path):
        video_path = str(tmp_path / "test.avi")
        _create_synthetic_video(video_path, num_frames=30, fps=30.0)

        frames, timestamps, native_fps = video_to_frames(
            video_path, fps=15, resize=False
        )

        # Should extract roughly half the frames
        assert len(frames) == 15
        assert len(timestamps) == 15

    def test_rotate(self, tmp_path):
        video_path = str(tmp_path / "test.avi")
        _create_synthetic_video(
            video_path, num_frames=5, width=64, height=48, fps=15.0
        )

        frames, _, _ = video_to_frames(
            video_path, rotate=True, resize=False
        )

        # After 90-degree clockwise rotation: (48, 64) -> (64, 48)
        assert frames[0].shape == (64, 48, 3)

    def test_invalid_path_raises(self):
        with pytest.raises(IOError, match="Cannot open video"):
            video_to_frames("/nonexistent/video.mp4")


# ---------------------------------------------------------------------------
# Tests: merge_annotations
# ---------------------------------------------------------------------------

class TestMergeAnnotations:
    def test_single_annotator(self):
        labels = np.array([[0, 1, 1, 2, 2]])
        merged = merge_annotations(labels, expected_n=5)
        np.testing.assert_array_equal(merged, [0, 1, 1, 2, 2])

    def test_multiple_annotators_majority_vote(self):
        labels = np.array([
            [0, 1, 1, 2, 2],
            [0, 1, 2, 2, 2],
            [1, 1, 1, 2, 0],
        ])
        merged = merge_annotations(labels, expected_n=5)
        # Frame 0: votes [0,0,1] -> 0
        # Frame 1: votes [1,1,1] -> 1
        # Frame 2: votes [1,2,1] -> 1
        # Frame 3: votes [2,2,2] -> 2
        # Frame 4: votes [2,2,0] -> 2
        np.testing.assert_array_equal(merged, [0, 1, 1, 2, 2])

    def test_padding_when_shorter(self):
        labels = np.array([[0, 1, 2]])
        merged = merge_annotations(labels, expected_n=5)
        np.testing.assert_array_equal(merged, [0, 1, 2, 0, 0])

    def test_truncation_when_longer(self):
        labels = np.array([[0, 1, 2, 3, 4, 5]])
        merged = merge_annotations(labels, expected_n=3)
        np.testing.assert_array_equal(merged, [0, 1, 2])

    def test_1d_input(self):
        labels = np.array([0, 1, 2, 3])
        merged = merge_annotations(labels, expected_n=4)
        np.testing.assert_array_equal(merged, [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Tests: label_timestamps
# ---------------------------------------------------------------------------

class TestLabelTimestamps:
    def test_exact_match(self):
        timestamps = [0.0, 0.1, 0.2, 0.3]
        annotations = np.array([0, 1, 1, 2])
        labels = label_timestamps(timestamps, annotations)
        np.testing.assert_array_equal(labels, [0, 1, 1, 2])

    def test_annotations_shorter(self):
        timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
        annotations = np.array([0, 1, 2])
        labels = label_timestamps(timestamps, annotations)
        np.testing.assert_array_equal(labels, [0, 1, 2, 0, 0])

    def test_annotations_longer(self):
        timestamps = [0.0, 0.1]
        annotations = np.array([0, 1, 2, 3, 4])
        labels = label_timestamps(timestamps, annotations)
        np.testing.assert_array_equal(labels, [0, 1])


# ---------------------------------------------------------------------------
# Tests: save_frames
# ---------------------------------------------------------------------------

class TestSaveFrames:
    def test_creates_directory_and_files(self, tmp_path):
        frames = [
            np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        video_dir = save_frames(frames, str(tmp_path), "test_video")

        assert video_dir.is_dir()
        assert video_dir.name == "test_video"

        # Check frame files
        jpg_files = sorted(video_dir.glob("frame_*.jpg"))
        assert len(jpg_files) == 5
        assert jpg_files[0].name == "frame_000000.jpg"
        assert jpg_files[4].name == "frame_000004.jpg"

        # Verify images can be opened
        img = Image.open(jpg_files[0])
        assert img.size == (64, 48)  # PIL uses (width, height)

    def test_saves_frame_labels(self, tmp_path):
        frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        labels = np.array([0, 1, 2])

        video_dir = save_frames(
            frames, str(tmp_path), "labeled_video", frame_labels=labels
        )

        label_path = video_dir / "frame_labels.npy"
        assert label_path.is_file()
        loaded = np.load(label_path)
        np.testing.assert_array_equal(loaded, [0, 1, 2])

    def test_saves_seq_label(self, tmp_path):
        frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for _ in range(2)
        ]

        video_dir = save_frames(
            frames, str(tmp_path), "seq_video", seq_label=5
        )

        seq_path = video_dir / "seq_label.npy"
        assert seq_path.is_file()
        assert int(np.load(seq_path)) == 5

    def test_no_labels_by_default(self, tmp_path):
        frames = [
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            for _ in range(2)
        ]

        video_dir = save_frames(frames, str(tmp_path), "no_labels")

        assert not (video_dir / "frame_labels.npy").exists()
        assert not (video_dir / "seq_label.npy").exists()


# ---------------------------------------------------------------------------
# Integration: save_frames -> VideoDataset
# ---------------------------------------------------------------------------

class TestIntegrationWithVideoDataset:
    """Test that saved frames can be loaded by VideoDataset."""

    def test_save_then_load(self, tmp_path):
        # Create a small dataset with 2 videos
        for vid_idx in range(2):
            num_frames = 10 + vid_idx * 5
            frames = [
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                for _ in range(num_frames)
            ]
            labels = np.arange(num_frames, dtype=np.int64)
            save_frames(
                frames,
                str(tmp_path),
                f"video_{vid_idx:03d}",
                frame_labels=labels,
                seq_label=vid_idx,
            )

        # Verify directory structure
        video_dirs = sorted(
            [d for d in tmp_path.iterdir() if d.is_dir()]
        )
        assert len(video_dirs) == 2

        # Try loading with VideoDataset
        from tcc.datasets import VideoDataset

        dataset = VideoDataset(
            root_dir=str(tmp_path),
            num_frames=5,
            sampling_strategy="offset_uniform",
            sample_all=False,
            frame_labels=True,
        )

        assert len(dataset) == 2

        frames_t, frame_labels_t, seq_label, seq_len, name = dataset[0]
        assert frames_t.shape[0] == 5  # num_frames
        assert frames_t.shape[1] == 3  # channels
        assert frame_labels_t.shape[0] == 5
        assert name == "video_000"
        assert seq_label == 0
        assert seq_len == 10

    def test_save_then_load_without_labels(self, tmp_path):
        frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(8)
        ]
        save_frames(frames, str(tmp_path), "unlabeled")

        from tcc.datasets import VideoDataset

        dataset = VideoDataset(
            root_dir=str(tmp_path),
            num_frames=4,
            frame_labels=True,
        )

        assert len(dataset) == 1
        frames_t, frame_labels_t, seq_label, seq_len, name = dataset[0]
        assert frames_t.shape[0] == 4
        # Without labels, should get zeros
        assert (frame_labels_t == 0).all()
        assert seq_label == 0
