"""Building a model from its config, and loading a trained one from its folder."""

from pathlib import Path

import torch

from ..config import Config, config_from_dict
from ..data.normalization import Scaling
from ..models import build_backbone
from ..objectives import consistency, flow
from ..objectives.edm import Denoiser
from ..objectives.residual import ResidualModel
from ..paths import model_path


def build_model(cfg: Config, sigma_data: float = 1.0) -> torch.nn.Module:
    """The backbone alone for regression, or the backbone inside an EDM denoiser, a consistency student, or a
    flow matching model. A residual model's regression part is added by ``load_checkpoint``."""
    net = {
        "unet": dict(channels=cfg.channels, blocks=cfg.blocks, emb_dim=cfg.emb_dim, attention=cfg.attention),
        "dit": dict(patch=cfg.patch, depth=cfg.dit_depth, width=cfg.dit_width, heads=cfg.dit_heads),
        "fno": dict(modes=cfg.fno_modes, width=cfg.fno_width, depth=cfg.fno_depth),
    }[cfg.backbone]
    common = dict(sigma_data=sigma_data, fixed_edges=cfg.fixed_edges, in_channels=cfg.in_channels,
                  backbone=cfg.backbone, **net)
    if cfg.objective == "edm":
        return Denoiser(cfg.cond_dim, **common)
    if cfg.objective == "consistency":
        return consistency.ConsistencyModel(cfg.cond_dim, **common)
    if cfg.objective == "flow_matching":
        return flow.FlowModel(cfg.cond_dim, **common)
    return build_backbone(cfg.backbone, cfg.cond_dim, cfg.in_channels, cfg.fixed_edges, **net)


def load_checkpoint(path: Path, device: str | None = None, models_root: Path | None = None
                    ) -> tuple[torch.nn.Module, Scaling | None, Config, dict]:
    """Return (model in eval mode, scaling, config, full checkpoint) from a model folder (its ``best.pt``)
    or a checkpoint file. ``scaling`` is None for DLA models. A residual model comes back as a
    ``ResidualModel``, with its regression model loaded from ``cfg.residual_of`` (under ``models_root``)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(path)
    ckpt = torch.load(path / "best.pt" if path.is_dir() else path, map_location=device, weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    scaling = Scaling.from_dict(ckpt["scaling"]) if ckpt.get("scaling") is not None else None
    model = build_model(cfg, ckpt["sigma_data"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if cfg.residual_of:
        base = load_checkpoint(model_path(cfg.residual_of, models_root), device)[0]
        model = ResidualModel(base, model).eval()
    return model, scaling, cfg, ckpt
