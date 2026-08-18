# DDPM Implementation on CIFAR-10

Course project for Machine Learning, Faculty of Mathematics, University of Belgrade.

## Project description

The goal is to implement *Denoising Diffusion Probabilistic Models* (Ho, Jain, and Abbeel, 2020) for image generation on CIFAR-10.

What is implemented:

- linear noise schedule `β_t` from `1e-4` to `0.02` with `T = 1000` steps,
- closed-form noising: `x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε`,
- paper-width time-conditioned U-Net (about 35.7M parameters) as the noise predictor `ε_θ(x_t, t)`,
- simplified objective `L_simple` (MSE between true and predicted noise),
- training with Adam, dropout, EMA weights, and checkpoints,
- reverse process (Algorithm 2 from the paper) for sampling images.

Training is unconditional: images are used and labels are ignored. The training budget is smaller than in the original paper (800k steps there), so we do not claim the paper FID. The long GPU run is in `06_train.ipynb`, behind the `RUN_PRODUCTION_TRAINING` flag.

## Dataset

We use [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009): 60,000 RGB images of size 32×32, in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

- Official train split: 50,000 images, 5,000 per class.
- Official test split: 10,000 images, 1,000 per class.

This is generation, not classification, so the model trains on the full train split. The test set is not a classification hold-out; it is a visual / optional FID reference.

For training, images are mapped to `[-1, 1]` (`Normalize(0.5)`) and a random horizontal flip is applied, as in the paper. Data is downloaded into `dataset/` (gitignored). Split, class-balance, and pixel analysis is in `01_dataset.ipynb`.

## Layout

1. [`01_dataset.ipynb`](01_dataset.ipynb) — download and analyze CIFAR-10.
2. [`02_diffusion.py`](02_diffusion.py) — `β_t` schedule, `q_sample`, `L_simple`, reverse sampler.
3. [`03_forward_noising.ipynb`](03_forward_noising.ipynb) — check the schedule and the noising visualization.
4. [`04_unet.py`](04_unet.py) — time-conditioned U-Net.
5. [`05_check_unet.ipynb`](05_check_unet.ipynb) — check shapes, gradients, and parameter count.
6. [`06_train.ipynb`](06_train.ipynb) — `L_simple`, one-batch overfit, AMP / EMA / checkpoints. The long run is behind `RUN_PRODUCTION_TRAINING`.
7. [`07_sample_eval.ipynb`](07_sample_eval.ipynb) — demo: sampler, loss curve, trajectory, sample grid, raw vs EMA.

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

## Team

- Branko Grbić
- Bogdan Stojadinović (dropped off the course)

## Literature

- Ho, J., Jain, A., and Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- Krizhevsky, A. (2009). [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf). Technical report, University of Toronto. Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
