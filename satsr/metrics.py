"""Quality + spectral-consistency metrics (numpy/cv2, no skimage)."""
from __future__ import annotations

import cv2
import numpy as np


def psnr(ref, test, data_range=1.0):
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(data_range ** 2 / mse)


def _ssim_1(a, b, dr):
    C1, C2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    a, b = a.astype(np.float64), b.astype(np.float64)
    k, s = (11, 11), 1.5
    ma, mb = cv2.GaussianBlur(a, k, s), cv2.GaussianBlur(b, k, s)
    ma2, mb2, mab = ma * ma, mb * mb, ma * mb
    sa = cv2.GaussianBlur(a * a, k, s) - ma2
    sb = cv2.GaussianBlur(b * b, k, s) - mb2
    sab = cv2.GaussianBlur(a * b, k, s) - mab
    num = (2 * mab + C1) * (2 * sab + C2)
    den = (ma2 + mb2 + C1) * (sa + sb + C2)
    return float(np.mean(num / den))


def ssim_rgb(ref, test, data_range=1.0):
    if ref.ndim == 2:
        return _ssim_1(ref, test, data_range)
    n = min(3, ref.shape[-1])
    return float(np.mean([_ssim_1(ref[..., c], test[..., c], data_range) for c in range(n)]))


def sam(ref, test, eps=1e-8):
    r = ref.reshape(-1, ref.shape[-1]).astype(np.float64)
    t = test.reshape(-1, test.shape[-1]).astype(np.float64)
    dot = np.sum(r * t, 1)
    denom = np.sqrt(np.sum(r * r, 1)) * np.sqrt(np.sum(t * t, 1))
    v = denom > eps
    ang = np.arccos(np.clip(dot[v] / denom[v], -1, 1))
    return float(np.degrees(np.mean(ang))) if v.any() else 0.0


def ergas(ref, test, scale, eps=1e-8):
    B = ref.shape[-1] if ref.ndim == 3 else 1
    ref = ref.reshape(ref.shape[0], ref.shape[1], B)
    test = test.reshape(test.shape[0], test.shape[1], B)
    terms = []
    for b in range(B):
        rb, tb = ref[..., b].astype(np.float64), test[..., b].astype(np.float64)
        rmse = np.sqrt(np.mean((rb - tb) ** 2))
        terms.append((rmse / (np.mean(rb) + eps)) ** 2)
    return float(100.0 * (1.0 / scale) * np.sqrt(np.mean(terms)))
