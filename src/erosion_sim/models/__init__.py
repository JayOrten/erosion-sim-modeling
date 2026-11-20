"""The networks. Each backbone maps (x, cond) to a one-channel map.

- ``unet``: the U-Net.
- ``dit``: DiT, a patch transformer with 2D RoPE (rotary position embedding).
- ``fno``: FNO, the Fourier Neural Operator.
- ``layers``: pieces the backbones share.

``build_backbone`` picks one by name.
"""

from torch import nn

from ..config import BACKBONES
from .dit import DiT
from .fno import FNO
from .unet import UNet


def build_backbone(backbone: str, cond_dim: int, in_channels: int, fixed_edges: bool, **kwargs) -> nn.Module:
    """A ``UNet``, a ``DiT`` (``dit``), or an ``FNO`` (``fno``). Each maps (x, cond) to a one-channel
    map. ``kwargs`` are the chosen backbone's own settings."""
    if backbone == "unet":
        return UNet(cond_dim, in_channels=in_channels, fixed_edges=fixed_edges, **kwargs)
    if backbone == "dit":
        return DiT(cond_dim, in_channels=in_channels, fixed_edges=fixed_edges, **kwargs)
    if backbone == "fno":
        return FNO(cond_dim, in_channels=in_channels, fixed_edges=fixed_edges, **kwargs)
    raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
