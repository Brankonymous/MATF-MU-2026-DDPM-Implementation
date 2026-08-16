# MATF-MU-2026-DDPM-Implementation
Implementation of Denoising Diffusion Probabilistic Models

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

## Notebooks

1. [`01_dataset.ipynb`](01_dataset.ipynb) downloads CIFAR-10 into `dataset/` and analyzes its
   splits, class balance, pixel statistics, visual structure, and model-ready normalization.
