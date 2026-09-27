"""Configuration for the satsr framework (dataclass + optional YAML/JSON file)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional


@dataclass
class SatSRConfig:
    # --- model ---
    weights: str = ""                      # path to .pth; resolved by pipeline if empty
    mode: str = "rgbn"                     # "rgbn" (4-band joint) | "per_band" (any N bands)
    in_channels: int = 4                   # #input bands. rgbn: 3 or 4. per_band: any N>=1
    native_scale: int = 4                  # the RRDBNet backbone upscales x4
    target_scale: float = 2.5              # 10 m -> 4 m

    # --- input semantics ---
    band_order: List[str] = field(default_factory=lambda: ["red", "green", "blue", "nir"])
    input_is_dn: bool = True               # Sentinel-2 DN (uint16); divide by dn_scale
    dn_scale: float = 10000.0              # reflectance = DN / dn_scale
    nodata: Optional[float] = 0.0          # input nodata value (None to disable)

    # --- tiled inference (memory-bounded) ---
    tile: int = 384                        # input tile size (px)
    tile_pad: int = 32                     # halo to avoid seams
    device: str = "auto"                   # auto | cuda | cpu

    # --- uncertainty ---
    uncertainty: bool = True               # emit TTA-ensemble uncertainty map
    tta: int = 8                           # 1 (off), 4, or 8 augmentations

    # --- radiometric correction ---
    # match output per-band mean/std to the input radiometry (kills colour cast);
    # GLOBAL coefficients from a preview pass -> identical for every tile (seam-safe)
    radiometric_correction: bool = True

    # --- output ---
    out_dtype: str = "uint16"              # uint16 (DN) or float32 (reflectance)
    compress: str = "deflate"
    log_level: str = "INFO"

    def validate(self) -> "SatSRConfig":
        if self.mode not in ("rgbn", "per_band"):
            raise ValueError("mode must be 'rgbn' or 'per_band'")
        if self.mode == "rgbn" and self.in_channels not in (3, 4):
            raise ValueError("rgbn mode requires in_channels 3 or 4 "
                             "(use mode='per_band' for arbitrary band counts)")
        if self.mode == "per_band" and self.in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if self.target_scale <= 1.0:
            raise ValueError("target_scale must be > 1.0")
        if len(self.band_order) != self.in_channels:
            raise ValueError(f"band_order has {len(self.band_order)} entries but "
                             f"in_channels={self.in_channels}")
        if self.tta not in (1, 4, 8):
            raise ValueError("tta must be 1, 4 or 8")
        if self.out_dtype not in ("uint16", "float32"):
            raise ValueError("out_dtype must be uint16 or float32")
        return self

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_file(cls, path: str) -> "SatSRConfig":
        with open(path) as f:
            text = f.read()
        if path.endswith((".yaml", ".yml")):
            import yaml
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls(**(data or {})).validate()

    @classmethod
    def from_overrides(cls, base: Optional["SatSRConfig"] = None, **overrides) -> "SatSRConfig":
        cfg = base or cls()
        data = asdict(cfg)
        for k, v in overrides.items():
            if v is not None and k in data:
                data[k] = v
        return cls(**data).validate()
