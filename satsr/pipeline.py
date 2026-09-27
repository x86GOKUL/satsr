"""High-level orchestration: resolve weights, run the super-resolution engine."""
from __future__ import annotations

import os
from typing import Optional

from .config import SatSRConfig
from .inference import SuperResolver
from .logging_utils import get_logger

log = get_logger("pipeline")

# weights directory (override with env SATSR_WEIGHTS)
_DEFAULT_WEIGHTS_DIR = os.environ.get(
    "SATSR_WEIGHTS", os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights"))

_DEFAULT_WEIGHTS = {
    4: "RealESRGAN_ms_s2.5.pth",       # 4-band R,G,B,NIR fine-tuned (10m->4m)
    3: "RealESRGAN_x4plus_s2ft_s2_best.pth",   # RGB fine-tuned; falls back to base
}
_FALLBACK_3 = "RealESRGAN_x4plus.pth"


def resolve_weights(cfg: SatSRConfig) -> str:
    if cfg.weights:
        if not os.path.exists(cfg.weights):
            raise FileNotFoundError(f"weights not found: {cfg.weights}")
        return cfg.weights
    # per_band uses the 3-channel (satellite-adapted) backbone regardless of #bands
    model_ch = 3 if cfg.mode == "per_band" else cfg.in_channels
    cand = os.path.join(_DEFAULT_WEIGHTS_DIR, _DEFAULT_WEIGHTS[model_ch])
    if not os.path.exists(cand) and model_ch == 3:
        cand = os.path.join(_DEFAULT_WEIGHTS_DIR, _FALLBACK_3)
    if not os.path.exists(cand):
        raise FileNotFoundError(
            f"No weights for {cfg.in_channels}-channel model in {_DEFAULT_WEIGHTS_DIR}. "
            f"Set config.weights or env SATSR_WEIGHTS.")
    return cand


def enhance_scene(input_path: str, output_path: str,
                  config: Optional[SatSRConfig] = None,
                  uncertainty_path: Optional[str] = None) -> dict:
    """
    Super-resolve a georeferenced Sentinel-2 scene to <4 m, preserving CRS and
    spectral consistency. Returns a summary dict.
    """
    cfg = (config or SatSRConfig()).validate()
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    cfg.weights = resolve_weights(cfg)
    log.info("weights: %s", cfg.weights)
    if uncertainty_path is None and cfg.uncertainty and cfg.tta > 1:
        base, _ = os.path.splitext(output_path)
        uncertainty_path = base + "_uncertainty.tif"
    engine = SuperResolver(cfg)
    return engine.enhance(input_path, output_path, uncertainty_path)
