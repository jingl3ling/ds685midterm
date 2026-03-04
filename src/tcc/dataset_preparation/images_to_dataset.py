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

"""Convert existing image directories into the standard dataset layout.

Reads image directories (one directory per video), optionally resizes or
rotates the images, and writes them in the layout expected by
:class:`tcc.datasets.VideoDataset`.

Usage::

    python -m tcc.dataset_preparation.images_to_dataset \\
        --input-dir /data/raw_images \\
        --output-dir /data/processed \\
        --name penn_action \\
        --width 224 --height 224
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from tcc.dataset_preparation.dataset_utils import save_frames

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _natural_sort_key(s: str):
    """Sort strings with embedded integers in natural order."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def create_dataset(
    name: str,
    output_dir: str,
    input_dir: str,
    width: int = 224,
    height: int = 224,
    rotate: bool = False,
    resize: bool = True,
    label_file: Optional[str] = None,
) -> Path:
    """Read image directories and save in standard format.

    Parameters
    ----------
    name:
        Dataset name (sub-directory under *output_dir*).
    output_dir:
        Root output directory.
    input_dir:
        Directory containing per-video sub-directories of images.
    width, height:
        Target frame dimensions when *resize* is True.
    rotate:
        Rotate images 90 degrees clockwise.
    resize:
        Resize images to *width* x *height*.
    label_file:
        Optional path to a ``.npy`` file with per-video label arrays.

    Returns
    -------
    dataset_dir:
        Path to the created dataset directory.
    """
    dataset_dir = Path(output_dir) / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_dir)
    video_dirs = sorted(
        [d for d in input_path.iterdir() if d.is_dir()],
        key=lambda p: _natural_sort_key(p.name),
    )

    if not video_dirs:
        raise FileNotFoundError(f"No sub-directories found in {input_dir}")

    # Load labels if provided
    all_labels = None
    if label_file is not None:
        all_labels = np.load(label_file, allow_pickle=True)
        logger.info("Loaded labels from %s", label_file)

    manifest = {
        "name": name,
        "num_videos": len(video_dirs),
        "width": width,
        "height": height,
        "resize": resize,
        "rotate": rotate,
        "videos": [],
    }

    for idx, vdir in enumerate(video_dirs):
        video_name = vdir.name
        logger.info(
            "Processing %d/%d: %s", idx + 1, len(video_dirs), video_name
        )

        # Collect image files
        image_files = sorted(
            [f for f in vdir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: _natural_sort_key(p.name),
        )

        if not image_files:
            logger.warning("No images found in %s, skipping.", vdir)
            continue

        frames = []
        for img_path in image_files:
            img = np.array(Image.open(img_path).convert("RGB"))
            if rotate:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            if resize:
                img = cv2.resize(
                    img, (width, height), interpolation=cv2.INTER_AREA
                )
            frames.append(img)

        # Labels
        frame_labels = None
        if all_labels is not None and idx < len(all_labels):
            fl = np.asarray(all_labels[idx]).ravel()
            padded = np.zeros(len(frames), dtype=np.int64)
            copy_len = min(len(frames), len(fl))
            padded[:copy_len] = fl[:copy_len]
            frame_labels = padded

        # Check for existing label files in the source directory
        if frame_labels is None:
            for candidate in ("frame_labels.npy", "labels.npy"):
                p = vdir / candidate
                if p.is_file():
                    frame_labels = np.load(p)
                    break

        # Check for seq_label
        seq_label = None
        seq_label_path = vdir / "seq_label.npy"
        if seq_label_path.is_file():
            seq_label = int(np.load(seq_label_path))

        save_frames(
            frames,
            output_dir=str(dataset_dir),
            name=video_name,
            frame_labels=frame_labels,
            seq_label=seq_label,
        )

        manifest["videos"].append(
            {
                "name": video_name,
                "source": str(vdir),
                "num_frames": len(frames),
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
        description="Convert image directories to standard dataset format."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing per-video sub-directories of images.",
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
    parser.add_argument("--width", type=int, default=224, help="Frame width.")
    parser.add_argument("--height", type=int, default=224, help="Frame height.")
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Rotate images 90 degrees clockwise.",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        default=True,
        help="Resize images (default: True).",
    )
    parser.add_argument(
        "--no-resize",
        action="store_false",
        dest="resize",
        help="Do not resize images.",
    )
    parser.add_argument(
        "--label-file",
        default=None,
        help="Path to .npy file with per-video label arrays.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    create_dataset(
        name=args.name,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        width=args.width,
        height=args.height,
        rotate=args.rotate,
        resize=args.resize,
        label_file=args.label_file,
    )


if __name__ == "__main__":
    main()
