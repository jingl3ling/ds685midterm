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

"""Visualize an image-folder dataset using matplotlib.

Displays sampled frames from the dataset for quick inspection.

Usage::

    python -m tcc.dataset_preparation.visualize_dataset \\
        --dataset-dir /data/processed/pouring \\
        --num-vids 4 \\
        --num-skip-frames 5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _natural_sort_key(s: str):
    """Sort strings with embedded integers in natural order."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def _discover_video_dirs(dataset_dir: Path) -> List[Path]:
    """Find all sub-directories containing image files."""
    dirs = []
    for d in sorted(dataset_dir.iterdir(), key=lambda p: _natural_sort_key(p.name)):
        if d.is_dir():
            has_images = any(
                f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir()
            )
            if has_images:
                dirs.append(d)
    return dirs


def visualize(
    dataset_dir: str,
    num_vids: int = 4,
    num_skip_frames: int = 5,
) -> None:
    """Display frames from video directories in a grid.

    Parameters
    ----------
    dataset_dir:
        Path to the dataset directory (containing per-video sub-dirs).
    num_vids:
        Number of videos to display.
    num_skip_frames:
        Show every N-th frame (1 = show all frames).
    """
    dataset_path = Path(dataset_dir)
    video_dirs = _discover_video_dirs(dataset_path)

    if not video_dirs:
        print(f"No video directories found in {dataset_dir}")
        return

    num_vids = min(num_vids, len(video_dirs))
    video_dirs = video_dirs[:num_vids]

    for vdir in video_dirs:
        image_files = sorted(
            [
                f
                for f in vdir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda p: _natural_sort_key(p.name),
        )

        # Sub-sample frames
        sampled = image_files[::num_skip_frames]
        if not sampled:
            continue

        # Load labels if available
        labels = None
        for candidate in ("frame_labels.npy", "labels.npy"):
            label_path = vdir / candidate
            if label_path.is_file():
                labels = np.load(label_path)
                break

        num_cols = min(len(sampled), 10)
        num_rows = (len(sampled) + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(2 * num_cols, 2.5 * num_rows)
        )
        fig.suptitle(vdir.name, fontsize=14)

        if num_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes[np.newaxis, :]
        elif num_cols == 1:
            axes = axes[:, np.newaxis]

        for i, img_path in enumerate(sampled):
            row, col = divmod(i, num_cols)
            ax = axes[row, col]
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            frame_idx = i * num_skip_frames
            title = f"#{frame_idx}"
            if labels is not None and frame_idx < len(labels):
                title += f" L={labels[frame_idx]}"
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        # Hide unused axes
        for i in range(len(sampled), num_rows * num_cols):
            row, col = divmod(i, num_cols)
            axes[row, col].axis("off")

        plt.tight_layout()

    plt.show()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize an image-folder dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--num-vids",
        type=int,
        default=4,
        help="Number of videos to display (default: 4).",
    )
    parser.add_argument(
        "--num-skip-frames",
        type=int,
        default=5,
        help="Show every N-th frame (default: 5).",
    )

    args = parser.parse_args()
    visualize(
        dataset_dir=args.dataset_dir,
        num_vids=args.num_vids,
        num_skip_frames=args.num_skip_frames,
    )


if __name__ == "__main__":
    main()
