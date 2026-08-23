# DDPM on CIFAR-10

Course project for Machine Learning, Faculty of Mathematics, University of Belgrade.

## Project

This repository implements an unconditional Denoising Diffusion Probabilistic Model (DDPM) following Ho, Jain, and Abbeel (2020):

- linear noise schedule and closed-form forward process,
- time-conditioned U-Net noise predictor,
- simplified noise-prediction MSE objective,
- Adam training with dropout, EMA, AMP on CUDA, and resumable checkpoints,
- Algorithm 2 reverse sampler,
- held-out noise-prediction MSE and FID evaluation.

The final training budget is 100,000 optimizer steps, below the paper's 800,000. FID-5k is therefore reported only as this project's result and is not compared with the paper's 50,000-sample FID.

## Data

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) has 50,000 training and 10,000 test RGB images at 32×32. Every class is balanced. The unconditional model ignores labels, trains only on the official training split, and uses the test split only for final loss and FID evaluation.

Training applies random horizontal flips and maps pixels to `[-1, 1]`. [`01_dataset.ipynb`](01_dataset.ipynb) contains the dataset analysis.

## Files

1. [`01_dataset.ipynb`](01_dataset.ipynb) — CIFAR-10 analysis.
2. [`02_diffusion.py`](02_diffusion.py) — forward and reverse diffusion.
3. [`03_forward_noising.ipynb`](03_forward_noising.ipynb) — schedule and noising checks.
4. [`04_unet.py`](04_unet.py) — configurable time-conditioned U-Net.
5. [`05_train.ipynb`](05_train.ipynb) — model checks, overfit test, timed training, EMA, and checkpoints.
6. [`06_sample_eval.ipynb`](06_sample_eval.ipynb) — samples, trajectory, held-out loss, and FID.

## Environment

Python 3.12.6, PyTorch 2.13.0, and CUDA 13.0 with the pinned packages in [`requirements.txt`](requirements.txt). Final hardware: NVIDIA RTX 4070 SUPER 12 GB.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

## Running

Training is opt-in. First run the CUDA/AMP preflight and inspect throughput and peak VRAM:

```bash
jupyter execute 05_train.ipynb --inplace
```

Then train to 100,000 optimizer steps and run the final evaluation:

```bash
DDPM_RUN_TRAINING=1 jupyter execute 05_train.ipynb --inplace
jupyter execute 06_sample_eval.ipynb --inplace
```

If physical batch 32 does not fit, add `DDPM_BATCH_SIZE=16`; accumulation changes automatically from four to eight steps, preserving effective batch 128.

## Results

- Hardware: NVIDIA RTX 4070 SUPER 12 GB.
- Environment: Python 3.12.6, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130.
- U-Net parameters: 35,746,307 (35.75M).
- Training wall time and throughput: **PLEEASE UPD THIS**.
- Training `L_simple`: **PLEEASE UPD THIS**.
- Held-out `L_simple`, overall and by timestep quarter: **PLEEASE UPD THIS**.
- FID-5k: **PLEEASE UPD THIS**.
- Sample quality and limitations: **PLEEASE UPD THIS**.

Large checkpoints are gitignored. Training writes a resumable `*_latest.pt` file and a smaller inference-only `*_ema.pt` file. After the production run, record its size and checksum with:

```bash
ls -lh checkpoints/ddpm_cifar10_production_ema.pt
shasum -a 256 checkpoints/ddpm_cifar10_production_ema.pt
```

To evaluate a downloaded artifact, set `DDPM_CHECKPOINT=/path/to/model.pt` when running `06_sample_eval.ipynb`; the architecture config is read from the checkpoint.

Model download: **PLEEASE UPD THIS**.

## Team

- Branko Grbić — final implementation and current repository history.
- Bogdan Stojadinović — dropped off the course.

## Literature

- Ho, J., Jain, A., and Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- Krizhevsky, A. (2009). [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).
