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

echo "Downloading pouring dataset from HuggingFace..."
echo "  Source: https://huggingface.co/datasets/sermanet/multiview-pouring"
echo "  Target: ${DATA_DIR}"
echo ""

# Primary method: HuggingFace CLI (recommended)
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download sermanet/multiview-pouring \
        --repo-type dataset \
        --local-dir "${DATA_DIR}"
else
    echo "huggingface-cli not found. Install with: pip install huggingface_hub"
    echo ""
    echo "Alternatively, download from Python:"
    echo ""
    echo "  from huggingface_hub import snapshot_download"
    echo "  snapshot_download("
    echo "      repo_id='sermanet/multiview-pouring',"
    echo "      repo_type='dataset',"
    echo "      local_dir='${DATA_DIR}',"
    echo "  )"
    exit 1
fi

echo ""
echo "Download complete. Files saved to: ${DATA_DIR}"
echo ""
echo "IMPORTANT: The downloaded files are in TFRecord format."
echo "To convert to the image-folder format required by the PyTorch pipeline:"
echo ""
echo "  python -m tcc.dataset_preparation.videos_to_dataset \\"
echo "      --input-dir ${DATA_DIR} \\"
echo "      --output-dir ${DATA_DIR}_processed/pouring \\"
echo "      --name pouring --fps 15 --width 224 --height 224"
