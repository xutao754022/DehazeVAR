# DehazeVAR: Bootstrapping Visual Autoregressive Modeling for Unsupervised Image Dehazing

Official PyTorch implementation of **DehazeVAR: Bootstrapping Visual Autoregressive Modeling for Unsupervised Image Dehazing** (Tao Xu, Ping Lang, Jianhong Lian, and Chang Liu).

DehazeVAR adapts pre-trained Visual Autoregressive (`VAR-d16`) models for unpaired image dehazing via two components:
- **Parallax-Resilient Tokenizer (PRT)**: Combines Dark Channel Prior (DCP) physical encoding, shared-index Softmax Relaxed Quantization (SRQ) codebook calibration, and disentangled clean/haze decoder branches pre-trained with online atmospheric scattering synthesis.
- **Conditional Residual Adapter (CRA) & Multi-Domain Objectives**: Injects domain-conditioned bottleneck residual adapters across decoder stages and optimizes bidirectional cycle consistency ($\ell_1 + \text{LPIPS}$), 2D Fourier amplitude/phase alignment, and frozen DINO ViT-B/8 spatial/spectral discrimination.

## Installation

```bash
git clone https://github.com/xutao754022/DehazeVAR.git
cd DehazeVAR

conda create -n dehazevar python=3.10 -y
conda activate dehazevar
pip install -r requirements.txt
```

## Pre-trained Weights & Datasets

### 1. Pre-trained Backbones

Download the official `VAR` checkpoints and `DINO` backbone into `weights/` (or update the corresponding paths in `configs/dehazevar.yaml`):

- [vae_ch160v4096z32.pth](https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth) -> `weights/var/vae_ch160v4096z32.pth`
- [var_d16.pth](https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth) -> `weights/var/var_d16.pth`
- [facebook/dino-vitb8](https://huggingface.co/facebook/dino-vitb8) -> `weights/dino-vitb8/`

### 2. Dataset Layout

Place the unpaired training images and evaluation benchmarks under `datasets/` (or modify the paths in `configs/dehazevar.yaml`):

```text
datasets/
├── train/
│   ├── hazy/
│   └── clean/
└── evaluation/
    ├── rtts/hazy/
    ├── haze2020/hazy/
    ├── urhi/hazy/
    ├── o-haze/{hazy,gt}/
    ├── i-haze/{hazy,gt}/
    ├── nh-haze/{hazy,gt}/
    ├── sots-indoor/{hazy,gt}/
    └── sots-outdoor/{hazy,gt}/
```

## Training

### Stage 1: PRT Pre-training

Pre-train the disentangled clean and physical haze decoder branches for 10,000 steps while keeping the base VQ-VAE encoder and codebook frozen:

```bash
CUDA_VISIBLE_DEVICES=0 python train_prt.py \
    --config configs/prt_pretraining.yaml \
    --output-dir runs/prt_stage1 \
    --max-steps 10000
```

After Stage 1 finishes, copy or link `runs/prt_stage1/checkpoints/step_0010000.pt` to `weights/prt/step_0010000.pt` (or pass `--prt-checkpoint` in Stage 2).

### Stage 2: Unpaired Dehazing Training

Train the full DehazeVAR model for 30,000 steps (including adapter warm-up prior to joint optimization):

```bash
CUDA_VISIBLE_DEVICES=0 python train_dehazevar.py \
    --config configs/dehazevar.yaml \
    --output-dir runs/dehazevar_stage2 \
    --max-steps 30000
```

To train the spectral discriminator variant:

```bash
CUDA_VISIBLE_DEVICES=0 python train_dehazevar.py \
    --config configs/dehazevar_spectral.yaml \
    --output-dir runs/dehazevar_spectral \
    --max-steps 30000
```

## Inference & Evaluation

### Directory Inference

Render dehazed predictions for a folder of hazy images:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config configs/dehazevar.yaml \
    --checkpoint runs/dehazevar_stage2/checkpoints/step_0030000.pt \
    --input-dir /path/to/hazy_images \
    --output-dir outputs/dehazed_results
```

### Benchmark Evaluation

Evaluate a trained checkpoint on the configured benchmarks (or specify a subset via `--dataset-names`):

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --config configs/dehazevar.yaml \
    --checkpoint runs/dehazevar_stage2/checkpoints/step_0030000.pt \
    --output-dir runs/dehazevar_stage2/evaluations
```

## Citation

```bibtex
@article{xu2026dehazevar,
  title   = {Bootstrapping Visual Autoregressive Modeling for Unsupervised Image Dehazing},
  author  = {Xu, Tao and Lang, Ping and Lian, Jianhong and Liu, Chang},
  journal = {IEEE Trans. Image Process.},
  year    = {2026}
}
```
