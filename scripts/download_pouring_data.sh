#!/bin/bash
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

# Download the pouring dataset used for TCC training/evaluation.
#
# NOTE: The original dataset was distributed as TFRecords. After
# downloading, you will need to convert them to the image-folder format
# expected by the PyTorch pipeline. Use:
#
#   python -m tcc.dataset_preparation.videos_to_dataset \
#       --input-dir <downloaded_videos_dir> \
#       --output-dir <output_dir> \
#       --name pouring \
#       --fps 15 --width 224 --height 224
#
# Alternatively, if you already have raw video files, use the
# videos_to_dataset script directly.

set -e

DATA_DIR="${1:-./data/pouring}"
mkdir -p "${DATA_DIR}"

echo "Downloading pouring dataset..."
echo "Target directory: ${DATA_DIR}"

# Original TFRecord URLs from the TCC paper.
# These may need to be converted to image-folder format after download.
BASE_URL="https://storage.googleapis.com/brain-tcc-data/pouring"

for SPLIT in train val; do
    echo "Downloading ${SPLIT} split..."
    wget -c "${BASE_URL}/${SPLIT}.tfrecord" -O "${DATA_DIR}/${SPLIT}.tfrecord" || {
        echo "Warning: Could not download ${SPLIT}.tfrecord"
        echo "The dataset may no longer be available at the original URL."
        echo "Check the TCC repository for updated download instructions."
    }
done

echo ""
echo "Download complete. Files saved to: ${DATA_DIR}"
echo ""
echo "IMPORTANT: The downloaded files are in TFRecord format."
echo "To convert to the image-folder format required by the PyTorch pipeline,"
echo "you will need to write a conversion script or obtain the raw videos"
echo "and use:"
echo ""
echo "  python -m tcc.dataset_preparation.videos_to_dataset \\"
echo "      --input-dir <raw_videos_dir> \\"
echo "      --output-dir ${DATA_DIR}/processed \\"
echo "      --name pouring"
