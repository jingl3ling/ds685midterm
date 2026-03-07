# Temporal Cycle-Consistency Learning

PyTorch implementation of [Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846) (Dwibedi et al., CVPR 2019).

Self-supervised representation learning on videos by exploiting temporal cycle-consistency constraints. Useful for fine-grained sequential/temporal understanding tasks.

## Branch Structure

- **`main`** — PyTorch implementation (in progress)
- **`tf2`** — Original Keras/TF2 codebase (archived reference)

## Dataset

The **multiview pouring dataset** used for training and evaluation is hosted on HuggingFace:

> [`sermanet/multiview-pouring`](https://huggingface.co/datasets/sermanet/multiview-pouring)

Download with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sermanet/multiview-pouring",
    repo_type="dataset",
    local_dir="data/pouring",
)
```

Or via the CLI:

```bash
huggingface-cli download sermanet/multiview-pouring --repo-type dataset --local-dir data/pouring
```

After downloading, convert the TFRecords to the image-folder format required by the PyTorch pipeline:

```bash
python -m tcc.dataset_preparation.videos_to_dataset \
    --input-dir data/pouring \
    --output-dir data/pouring_processed/pouring \
    --name pouring --fps 15 --width 224 --height 224
```

## Self-supervised Algorithms

- Temporal Cycle-Consistency (TCC)
- Shuffle and Learn
- Time-Contrastive Networks (TCN)
- Combined methods
- Supervised baseline (per-frame classification)

## Evaluation Tasks

- Phase classification
- Few-shot phase classification
- Phase progression
- Kendall's Tau

## Citation

If you found this paper/code useful in your research, please consider citing:

```bibtex
@InProceedings{Dwibedi_2019_CVPR,
  author = {Dwibedi, Debidatta and Aytar, Yusuf and Tompson, Jonathan and Sermanet, Pierre and Zisserman, Andrew},
  title = {Temporal Cycle-Consistency Learning},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019},
}
```
