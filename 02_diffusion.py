"""Forward and reverse diffusion math.

This is `02_diffusion.py`, used by `03_forward_noising.ipynb` and later notebooks.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor

    def to(self, device: torch.device) -> "DiffusionSchedule":
        moved = {name: value.to(device) for name, value in self.__dict__.items()}
        return DiffusionSchedule(**moved)


# Build all coefficients for the linear diffusion schedule.
def build_schedule(num_steps: int, beta_start: float, beta_end: float) -> DiffusionSchedule:
    # Gradually increase the noise added at each forward step.
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
    alphas = 1.0 - betas
    # The cumulative product measures how much original signal remains by step t.
    alpha_bars = torch.cumprod(alphas, dim=0)
    # Paper ᾱ_{t-1} is 1 at the first step because no previous product exists.
    alpha_bars_prev = torch.cat([torch.ones(1, dtype=alphas.dtype), alpha_bars[:-1]])
    # Controls how much random noise is added while moving from xₜ to the slightly cleaner xₜ₋₁.
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_prev=alpha_bars_prev,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
        sqrt_recip_alphas=torch.sqrt(1.0 / alphas),
        posterior_variance=posterior_variance,
    )


# Select and reshape one timestep coefficient per image.
def extract(coefficients: torch.Tensor, timesteps: torch.Tensor, shape: tuple) -> torch.Tensor:
    values = coefficients.to(timesteps.device).gather(0, timesteps.long())
    return values.reshape(timesteps.shape[0], *([1] * (len(shape) - 1)))


# Add noise to clean images at chosen timesteps.
def q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x0)
    # The closed form reaches any timestep without simulating earlier steps.
    signal = extract(schedule.sqrt_alpha_bars, timesteps, x0.shape)
    noise_scale = extract(schedule.sqrt_one_minus_alpha_bars, timesteps, x0.shape)
    return signal * x0 + noise_scale * noise


# Convert normalized model images to the display range.
def to_display(images: torch.Tensor) -> torch.Tensor:
    # Inverse of Normalize(0.5): map model space [-1, 1] to matplotlib [0, 1].
    return (images.clamp(-1, 1) + 1) / 2


# Measure how accurately the model predicts the added noise.
def simple_loss(
    model: nn.Module,
    images: torch.Tensor,
    schedule: DiffusionSchedule,
    timesteps: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    # Ho et al. 2020, Algorithm 1 / L_simple: predict the noise that formed x_t.
    batch = images.size(0)
    num_steps = int(schedule.betas.shape[0])
    # Random timesteps teach the model to handle every noise level.
    if timesteps is None:
        timesteps = torch.randint(0, num_steps, (batch,), device=images.device)
    if noise is None:
        noise = torch.randn_like(images)
    xt = q_sample(images, timesteps, schedule, noise)
    return F.mse_loss(model(xt, timesteps), noise)


# Perform one reverse-diffusion step without tracking gradients.
@torch.inference_mode()
def p_sample(
    model: nn.Module,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # Ho et al. 2020, Algorithm 2. Index 0 is the last reverse step and adds no noise.
    # The model estimates which noise component should be removed from x_t.
    predicted_noise = model(x_t, timesteps)
    beta_t = extract(schedule.betas, timesteps, x_t.shape)
    alpha_t = extract(schedule.alphas, timesteps, x_t.shape)
    alpha_bar_t = extract(schedule.alpha_bars, timesteps, x_t.shape)
    # Compute the center of the predicted distribution for x_{t-1}.
    mean = (x_t - beta_t * predicted_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
    variance = extract(schedule.posterior_variance, timesteps, x_t.shape)
    noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
    # Do not corrupt the final image by adding noise after the last step.
    nonzero_mask = (timesteps != 0).float().view(-1, 1, 1, 1)
    return mean + nonzero_mask * torch.sqrt(variance) * noise


# Generate images by repeatedly applying reverse diffusion.
@torch.inference_mode()
def sample_loop(
    model: nn.Module,
    schedule: DiffusionSchedule,
    batch_size: int,
    channels: int,
    image_size: int,
    device: torch.device,
    snapshot_steps: tuple[int, ...] = (),
    generator: torch.Generator | None = None,
    show_progress: bool | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    # Generation starts from pure Gaussian noise.
    images = torch.randn(
        batch_size,
        channels,
        image_size,
        image_size,
        device=device,
        generator=generator,
    )
    num_steps = int(schedule.betas.shape[0])
    # Key num_steps is the starting Gaussian, before any reverse update.
    snapshots: dict[int, torch.Tensor] = {num_steps: images.detach().cpu()}
    if show_progress is None:
        show_progress = sys.stdout.isatty()
    steps = reversed(range(num_steps))
    if show_progress:
        steps = tqdm(steps, total=num_steps, desc="reverse")
    # Reverse diffusion is sequential, so every timestep depends on the previous one.
    for step in steps:
        timesteps = torch.full((batch_size,), step, device=device, dtype=torch.long)
        images = p_sample(model, images, timesteps, schedule, generator)
        if step in snapshot_steps:
            snapshots[step] = images.detach().cpu()
    return images, snapshots
