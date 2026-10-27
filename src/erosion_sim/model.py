"""U-Net that predicts the change in a heightmap, conditioned on elapsed time and erosion parameters."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalFeatures(nn.Module):
    """Map each scalar in [0, 1] to sines and cosines at ``frequencies`` geometrically spaced frequencies.

    Small changes in a scalar then change many features, which a small MLP can use more easily than
    the raw value.
    """

    def __init__(self, frequencies: int = 16, max_frequency: float = 100.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(0.0, math.log(max_frequency), frequencies))
        self.register_buffer("freqs", freqs * math.pi)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        angles = cond[..., None] * self.freqs  # (batch, scalars, frequencies)
        return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1)


class ResBlock(nn.Module):
    """Two 3×3 convolutions with a residual connection. The conditioning vector scales and shifts the
    features between them (FiLM, feature-wise linear modulation)."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over all positions of a feature map, with a residual connection."""

    def __init__(self, channels: int, heads: int = 4, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)  # (batch, positions, channels)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    """Predict the normalized change ``y`` from the normalized start heightmap ``x`` and conditioning.

    - ``x``: (batch, 1, H, W). H and W must be divisible by ``2 ** (len(channels) - 1)``.
    - ``cond``: (batch, cond_dim), each value about in [0, 1].
    - Output: (batch, 1, H, W). Edge nodes are set to zero, since the simulator holds edges fixed.

    The output layer starts at zero, so an untrained model predicts no change.
    """

    def __init__(
        self,
        cond_dim: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        blocks: int = 1,
        emb_dim: int = 256,
        frequencies: int = 16,
        attention: bool = True,
    ):
        super().__init__()
        self.levels = len(channels)
        self.features = SinusoidalFeatures(frequencies)
        self.embed = nn.Sequential(
            nn.Linear(cond_dim * 2 * frequencies, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim), nn.SiLU()
        )
        self.stem = nn.Conv2d(1, channels[0], 3, padding=1)

        # Encoder: `blocks` ResBlocks per level, then a stride-2 convolution down to the next level.
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skip_channels = []
        ch = channels[0]
        for level, out_ch in enumerate(channels):
            for _ in range(blocks):
                self.down.append(ResBlock(ch, out_ch, emb_dim))
                ch = out_ch
                skip_channels.append(ch)
            if level < self.levels - 1:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))

        self.mid1 = ResBlock(ch, ch, emb_dim)
        self.mid_attn = SelfAttention(ch) if attention else nn.Identity()
        self.mid2 = ResBlock(ch, ch, emb_dim)

        # Decoder: mirror of the encoder. Each ResBlock also takes the matching encoder output (skip).
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for level, out_ch in reversed(list(enumerate(channels))):
            for _ in range(blocks):
                self.up.append(ResBlock(ch + skip_channels.pop(), out_ch, emb_dim))
                ch = out_ch
            if level > 0:
                self.upsample.append(nn.Conv2d(ch, ch, 3, padding=1))

        self.out_norm = nn.GroupNorm(8, ch)
        self.out = nn.Conv2d(ch, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.blocks = blocks

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.embed(self.features(cond))
        h = self.stem(x)
        skips = []
        blocks = iter(self.down)
        for level in range(self.levels):
            for _ in range(self.blocks):
                h = next(blocks)(h, emb)
                skips.append(h)
            if level < self.levels - 1:
                h = self.downsample[level](h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        blocks = iter(self.up)
        for i, level in enumerate(reversed(range(self.levels))):
            for _ in range(self.blocks):
                h = next(blocks)(torch.cat([h, skips.pop()], dim=1), emb)
            if level > 0:
                h = self.upsample[i](F.interpolate(h, scale_factor=2, mode="nearest"))

        y = self.out(F.silu(self.out_norm(h)))
        edges = torch.ones_like(y)
        edges[..., 1:-1, 1:-1] = 0
        return y * (1 - edges)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
