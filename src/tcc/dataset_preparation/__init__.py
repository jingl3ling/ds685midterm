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

"""Dataset preparation utilities for TCC (PyTorch port).

This package provides tools for converting raw video files and image
directories into the image-folder format expected by
:class:`tcc.datasets.VideoDataset`.

Output layout::

    output_dir/
        video_name/
            frame_000000.jpg
            frame_000001.jpg
            ...
            frame_labels.npy   (optional)
            seq_label.npy      (optional)
        manifest.json
"""

from tcc.dataset_preparation.dataset_utils import (
    label_timestamps,
    merge_annotations,
    save_frames,
    video_to_frames,
)

__all__ = [
    "video_to_frames",
    "merge_annotations",
    "label_timestamps",
    "save_frames",
]
