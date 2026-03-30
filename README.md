# Representation Difference Distillation (RDD)

This repo contains the implementation of 
[Representational Difference Distillation](https://drive.google.com/file/d/1Is9-a2ZMwBdPYS7piE1mn8NuHTZYFg_p/view), a knowledge distillation method that exploits representational differences between teacher and student models. Read the blog [here](https://divinrkz.com/blog/research/rdd-distillation).



## Overview

RDD is a knowledge distillation framework that uses representational difference analysis to focus the distillation loss where the teacher and student structurally disagree.

**Core components:**

- **Asymmetric affinity matrices** (A01 / A10) that capture directional representational relationships between teacher and student
- **Momentum memory bank** for efficient, stable contrastive estimation across the dataset
- **constrastive loss** that smoothly encourages the student's representations to align with the teacher's in the disagreement region
- **Per-sample confusion weighting** that up-weights samples in the student's zone of proximal development — where learning signal is richest

## Installation

This repo was tested with Python 3.8+, PyTorch ≥ 1.10, and CUDA 11.x on Ubuntu.

```bash
pip install -r requirements.txt
```

## Running

### 1. Fetch pretrained teacher models

```bash
sh scripts/fetch_pretrained_teachers.sh
```

This downloads pretrained teachers to `save/models/`.

### 2. Run RDD distillation

```bash
python train_student.py \
  --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
  --distill rdx_contrast \
  --model_s resnet8x4 \
  -a 0 -b 0.8 \
  --trial 1
```

**Flags:**


| Flag        | Description                                     |
| ----------- | ----------------------------------------------- |
| `--path_t`  | Path to the pretrained teacher model            |
| `--model_s` | Student architecture (see `models/__init__.py`) |
| `--distill` | Distillation method (`rdx_contrast` for RDD)    |
| `-r`        | Weight of cross-entropy loss (default: `1`)     |
| `-a`        | Weight of KD loss (default: `None`)             |
| `-b`        | Weight of the RDD loss (default: `None`)        |
| `--trial`   | Experiment ID for multiple runs                 |


#### RDD sampling examples

Training uses **RDX scoring** to mine positives, negatives, and optional per-sample weights. The examples below show curriculum scheduling on the training set, periodic re-mining, and cheaper scoring with a subset of anchors.

**RDX curriculum** (easiest-first scheduling; works with `rdx_contrast`):

```bash
python train_student.py \
  --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
  --distill rdx_contrast \
  --model_s resnet8x4 \
  -a 0 -b 0.8 \
  --sampling curriculum \
  --curriculum_start_frac 0.3 \
  --curriculum_pace_epochs 100 \
  --trial 1
```

**Periodic re-scoring** (refresh contrast pairs every 20 epochs after `--rdx_start_epoch`; triplet table refreshes the same way for `rdx_triplet`):

```bash
python train_student.py \
  --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
  --distill rdx_contrast \
  --model_s resnet8x4 \
  -a 0 -b 0.8 \
  --rdx_start_epoch 1 \
  --rdx_refresh_epochs 20 \
  --trial 1
```

**Faster RDX scoring** (use 2048 anchor images instead of the full training set when mining):

```bash
python train_student.py \
  --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
  --distill rdx_contrast \
  --model_s resnet8x4 \
  -a 0 -b 0.8 \
  --rdx_anchor_n 2048 \
  --trial 1
```


| Flag                                       | Description                                                                    |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| `--sampling`                               | `standard` (default) or `curriculum` for RDX difficulty-based curriculum       |
| `--curriculum_start_frac`                  | Initial fraction of easiest (RDX-scored) samples (default: `0.3`)              |
| `--curriculum_pace_epochs`                 | Epochs to ramp from that fraction to the full set (default: `100`)             |
| `--rdx_start_epoch`                        | First epoch at which RDX tables / curriculum use scored subsets (default: `1`) |
| `--rdx_refresh_epochs`                     | Re-mine every N epochs (`0` = once; default: `0`)                              |
| `--rdx_anchor_n`                           | Number of anchor images for scoring (`0` = all; default: `0`)                  |
| `--rdx_gamma`, `--rdx_beta`, `--rdx_kna_k` | RDX affinity / curriculum hyperparameters (see `train_student.py`)             |


### 3. Combine RDD with KD

```bash
python train_student.py \
  --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth \
  --distill rdx_contrast \
  --model_s resnet8x4 \
  -a 1 -b 0.8 \
  --trial 1
```

### 4. (Optional) Train teacher networks from scratch

```bash
sh scripts/run_cifar_vanilla.sh
```

## Benchmark Results on CIFAR-100

Performance is measured by top-1 classification accuracy (%). Results are shown in the paper.

## Project Structure

```
├── train_student.py          # Main training script
├── distiller_zoo/
│   ├── RDXContrast.py                # RDDLoss: affinity matrices, memory bank,
│   └── ...                   # Other distillation methods (KD, CRD, etc.)
├── models/                   # Teacher and student architectures
├── dataset/                  # CIFAR-100 data loading
├── scripts/                  # Shell scripts for training
└── save/                     # Checkpoints and logs
```

## Maintainers

- **AbdulKarim Mugisha**
- **Divin Irakiza** — [divinrkz](https://divinrkz.com)

## Acknowledgement

The starter code for this project was adapted from the [RepDistiller](https://github.com/HobbitLong/RepDistiller) repository by Yonglong Tian, which accompanies the ICLR 2020 paper *Contrastive Representation Distillation* (CRD). We thank the original authors for making their codebase and pretrained models publicly available.

## Citation

```bibtex
@article{keza2026rdd,
  title={Representation Difference Distillation},
  author={Irakiza, Divin and Mugisha, AbdulKarim},
  year={2026}
}
```

