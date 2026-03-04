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

"""Convert video files into the image-folder dataset format.

Reads video files from an input directory, extracts frames using OpenCV,
and writes them into the layout expected by
:class:`tcc.datasets.VideoDataset`.

Usage::

    python -m tcc.dataset_preparation.videos_to_dataset \\
        --input-dir /data/raw_videos \\
        --output-dir /data/processed \\
        --name pouring \\
        --fps 15 --width 224 --height 224
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from tcc.dataset_preparation.dataset_utils import (
    label_timestamps,
    merge_annotations,
    save_frames,
    video_to_frames,
)

logger = logging.getLogger(__name__)


def create_dataset(
    name: str,
    output_dir: str,
    input_dir: str,
    file_pattern: str = "*.mp4",
    label_file: Optional[str] = None,
    fps: float = 0,
    width: int = 224,
    height: int = 224,
    rotate: bool = False,
    resize: bool = True,
) -> Path:
    """Read videos from *input_dir*, extract frames, and save as image folders.

    Parameters
    ----------
    name:
        Dataset name (used as a sub-directory under *output_dir*).
    output_dir:
        Root output directory.
    input_dir:
        Directory containing source video files.
    file_pattern:
        Glob pattern to match video files inside *input_dir*.
    label_file:
        Optional path to a ``.npy`` file containing per-video annotations.
        Expected shape ``(num_videos, num_annotators, num_frames)`` or
        ``(num_videos, num_frames)``.
    fps:
        Target FPS for frame extraction (0 = native FPS).
    width, height:
        Target frame dimensions.
    rotate:
        Rotate frames 90 degrees clockwise.
    resize:
        Resize frames to *width* x *height*.

    Returns
    -------
    dataset_dir:
        Path to the created dataset directory.
    """
    dataset_dir = Path(output_dir) / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(glob.glob(str(Path(input_dir) / file_pattern)))
    if not video_paths:
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' found in {input_dir}"
        )

    # Load labels if provided
    all_labels = None
    if label_file is not None:
        all_labels = np.load(label_file, allow_pickle=True)
        logger.info("Loaded labels from %s with shape %s", label_file, all_labels.shape)

    manifest = {
        "name": name,
        "num_videos": len(video_paths),
        "fps": fps,
        "width": width,
        "height": height,
        "resize": resize,
        "rotate": rotate,
        "videos": [],
    }

    for idx, video_path in enumerate(video_paths):
        video_name = Path(video_path).stem
        logger.info(
            "Processing video %d/%d: %s", idx + 1, len(video_paths), video_name
        )

        frames, timestamps, native_fps = video_to_frames(
            video_path,
            rotate=rotate,
            fps=fps,
            resize=resize,
            width=width,
            height=height,
        )

        if len(frames) == 0:
            logger.warning("No frames extracted from %s, skipping.", video_path)
            continue

        # Process labels
        frame_labels = None
        if all_labels is not None and idx < len(all_labels):
            video_labels = np.asarray(all_labels[idx])
            if video_labels.ndim >= 2:
                # Multiple annotators -- merge via voting
                frame_labels = merge_annotations(video_labels, len(frames))
            else:
                frame_labels = label_timestamps(timestamps, video_labels)

        save_frames(
            frames,
            output_dir=str(dataset_dir),
            name=video_name,
            frame_labels=frame_labels,
        )

        manifest["videos"].append(
            {
                "name": video_name,
                "source": video_path,
                "num_frames": len(frames),
                "native_fps": native_fps,
                "has_labels": frame_labels is not None,
            }
        )

    # Write manifest
    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Dataset '%s' created at %s with %d videos.",
        name,
        dataset_dir,
        len(manifest["videos"]),
    )
    return dataset_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert video files to image-folder dataset format."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing source video files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory for the dataset.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name (creates a sub-directory).",
    )
    parser.add_argument(
        "--file-pattern",
        default="*.mp4",
        help="Glob pattern for video files (default: '*.mp4').",
    )
    parser.add_argument(
        "--label-file",
        default=None,
        help="Path to .npy file with per-video annotations.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0,
        help="Target FPS (0 = native FPS).",
    )
    parser.add_argument("--width", type=int, default=224, help="Frame width.")
    parser.add_argument("--height", type=int, default=224, help="Frame height.")
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Rotate frames 90 degrees clockwise.",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        default=True,
        help="Resize frames (default: True).",
    )
    parser.add_argument(
        "--no-resize",
        action="store_false",
        dest="resize",
        help="Do not resize frames.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    create_dataset(
        name=args.name,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        file_pattern=args.file_pattern,
        label_file=args.label_file,
        fps=args.fps,
        width=args.width,
        height=args.height,
        rotate=args.rotate,
        resize=args.resize,
    )


if __name__ == "__main__":
    main()
