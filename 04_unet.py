"""Time-conditioned U-Net for DDPM.

This is `04_unet.py`, used by `05_check_unet.ipynb` and later notebooks.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetConfig(Protocol):
    image_size: int
    in_channels: int
    base_channels: int
    channel_mults: tuple[int, ...]
    num_res_blocks: int
    attention_resolutions: tuple[int, ...]
    dropout: float


def group_norm(channels: int) -> nn.GroupNorm:
    # Prefer 32 groups on the paper-width model; fall back until the count divides.
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise ValueError(f"no GroupNorm groups divide {channels}")


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = timesteps.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x))
        query, key, value = qkv.reshape(batch, self.num_heads, -1, height * width).chunk(3, dim=2)
        scale = (channels // self.num_heads) ** -0.5
        attention = torch.softmax(torch.einsum("bhci,bhcj->bhij", query * scale, key), dim=-1)
        mixed = torch.einsum("bhij,bhcj->bhci", attention, value).reshape(batch, channels, height, width)
        return x + self.proj(mixed)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class TimeResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float,
        use_attention: bool,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim, dropout)
        self.attn = AttentionBlock(out_channels) if use_attention else None

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, time_emb)
        if self.attn is not None:
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        time_dim = config.base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(config.base_channels),
            nn.Linear(config.base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(config.in_channels, config.base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        skip_channels = [config.base_channels]
        channels = config.base_channels
        resolution = config.image_size
        for level, multiplier in enumerate(config.channel_mults):
            out_channels = config.base_channels * multiplier
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(
                    TimeResBlock(
                        channels,
                        out_channels,
                        time_dim,
                        config.dropout,
                        resolution in config.attention_resolutions,
                    )
                )
                channels = out_channels
                skip_channels.append(channels)
            if level != len(config.channel_mults) - 1:
                self.down_blocks.append(Downsample(channels))
                skip_channels.append(channels)
                resolution //= 2

        self.mid_block1 = ResidualBlock(channels, channels, time_dim, config.dropout)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ResidualBlock(channels, channels, time_dim, config.dropout)

        self.up_blocks = nn.ModuleList()
        for level, multiplier in reversed(list(enumerate(config.channel_mults))):
            out_channels = config.base_channels * multiplier
            for _ in range(config.num_res_blocks + 1):
                skip_channels_here = skip_channels.pop()
                self.up_blocks.append(
                    TimeResBlock(
                        channels + skip_channels_here,
                        out_channels,
                        time_dim,
                        config.dropout,
                        resolution in config.attention_resolutions,
                    )
                )
                channels = out_channels
            if level != 0:
                self.up_blocks.append(Upsample(channels))
                resolution *= 2

        if skip_channels:
            raise RuntimeError(f"unused skip channels: {skip_channels}")

        self.out_norm = group_norm(channels)
        self.out_conv = nn.Conv2d(channels, config.in_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(timesteps)
        hidden = self.in_conv(x)
        skips = [hidden]
        for block in self.down_blocks:
            hidden = block(hidden, time_emb) if isinstance(block, TimeResBlock) else block(hidden)
            skips.append(hidden)

        hidden = self.mid_block1(hidden, time_emb)
        hidden = self.mid_attn(hidden)
        hidden = self.mid_block2(hidden, time_emb)

        for block in self.up_blocks:
            if isinstance(block, TimeResBlock):
                hidden = torch.cat([hidden, skips.pop()], dim=1)
                hidden = block(hidden, time_emb)
            else:
                hidden = block(hidden)

        if skips:
            raise RuntimeError(f"{len(skips)} skip tensors were never consumed")
        return self.out_conv(F.silu(self.out_norm(hidden)))
