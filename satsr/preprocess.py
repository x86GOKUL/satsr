"""Pre-processing: DN<->reflectance scaling, band reordering, validity masks."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

MODEL_ORDER_4 = ["red", "green", "blue", "nir"]
MODEL_ORDER_3 = ["red", "green", "blue"]


def model_order(in_channels: int) -> List[str]:
    return MODEL_ORDER_4 if in_channels == 4 else MODEL_ORDER_3


def permutation(band_order: List[str], in_channels: int) -> List[int]:
    """Indices that reorder input bands into the model's canonical order."""
    target = model_order(in_channels)
    lo = [b.lower() for b in band_order]
    try:
        return [lo.index(b) for b in target]
    except ValueError as e:
        raise ValueError(f"band_order {band_order} missing a required band "
                         f"for a {in_channels}-channel model {target}") from e


def to_reflectance(dn: np.ndarray, dn_scale: float, input_is_dn: bool) -> np.ndarray:
    x = dn.astype(np.float32)
    if input_is_dn:
        x = x / dn_scale
    return np.clip(x, 0.0, 1.0)


def from_reflectance(refl: np.ndarray, dn_scale: float, dtype: str) -> np.ndarray:
    refl = np.clip(refl, 0.0, 1.0)
    if dtype == "uint16":
        return np.clip(np.round(refl * dn_scale), 0, 65535).astype(np.uint16)
    return refl.astype(np.float32)


def validity_mask(arr_hwc: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """True where the pixel is valid across all bands."""
    if nodata is None:
        return np.ones(arr_hwc.shape[:2], dtype=bool)
    return ~np.all(arr_hwc == nodata, axis=-1)
