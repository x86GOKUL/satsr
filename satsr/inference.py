"""
Streaming tiled super-resolution engine.

Processes an arbitrarily large scene tile-by-tile with an overlap halo (no
seams), running the model at the backbone's native x4 then resampling to the
requested target scale, and streams each output tile straight into the
georeferenced GeoTIFF. Peak memory depends on tile size, not scene size.

Optionally emits a per-pixel uncertainty map from an 8x test-time-augmentation
self-ensemble (std across augmentations) — flagging where detail is inferred.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch

from . import io_geo, preprocess
from .config import SatSRConfig
from .logging_utils import get_logger
from .model import build_model, resolve_device

log = get_logger("inference")

# (flip, rot90) — 8 dihedral augmentations
_AUGS8 = [(None, 0), (None, 1), (None, 2), (None, 3),
          (1, 0), (1, 1), (1, 2), (1, 3)]


def _augs(n):
    return _AUGS8[:1] if n == 1 else (_AUGS8[:4] if n == 4 else _AUGS8)


def _apply(a, flip, rot):
    if flip is not None:
        a = np.flip(a, axis=1)
    if rot:
        a = np.rot90(a, rot)
    return np.ascontiguousarray(a)


def _invert(a, flip, rot):
    if rot:
        a = np.rot90(a, -rot)
    if flip is not None:
        a = np.flip(a, axis=1)
    return np.ascontiguousarray(a)


class SuperResolver:
    def __init__(self, cfg: SatSRConfig):
        self.cfg = cfg.validate()
        self.device = resolve_device(cfg.device)
        log.info("device: %s (mode=%s, %d bands)", self.device, cfg.mode, cfg.in_channels)
        # per_band uses a 3-channel backbone applied to each band independently
        model_channels = 3 if cfg.mode == "per_band" else cfg.in_channels
        self.model = build_model(model_channels, cfg.weights or None, self.device)
        self.perm = None  # set per scene
        self.rc = None    # (scale[C], shift[C]) global radiometric-correction coeffs

    def _infer(self, tile_hwc: np.ndarray):
        """Dispatch to joint (rgbn) or per-band super-resolution. Returns (mean4x, std4x)."""
        if self.cfg.mode != "per_band":
            return self._net4x(tile_hwc)
        # band-agnostic: run each band through the 3-ch backbone independently
        means, stds = [], []
        for b in range(tile_hwc.shape[2]):
            rep = np.repeat(tile_hwc[..., b:b + 1], 3, axis=2)   # replicate to 3ch
            m, s = self._net4x(rep)
            means.append(m[..., 0])
            stds.append(s)
        return np.stack(means, -1).astype(np.float32), np.mean(np.stack(stds, -1), -1).astype(np.float32)

    def _compute_rc(self, in_path):
        """Global per-band affine matching output radiometry to the input.
        Uses a cheap decimated preview so one set of coefficients serves every
        tile -> no per-tile drift, no seams."""
        cfg = self.cfg
        ov = io_geo.read_overview(in_path, 384)[..., self.perm]        # model band order
        ref = preprocess.to_reflectance(ov, cfg.dn_scale, cfg.input_is_dn)
        # RC only needs the mean SR statistics — a single pass is enough, so we
        # skip the TTA ensemble here (saves tta-1 forward passes per scene).
        _tta = cfg.tta
        cfg.tta = 1
        try:
            sr_prev, _ = self._infer(ref)                             # SR preview, model order
        finally:
            cfg.tta = _tta
        C = ref.shape[2]
        scale = np.ones(C, np.float32); shift = np.zeros(C, np.float32)
        for c in range(C):
            rstd, rmean = ref[..., c].std(), ref[..., c].mean()
            sstd, smean = sr_prev[..., c].std(), sr_prev[..., c].mean()
            scale[c] = rstd / (sstd + 1e-6)
            shift[c] = rmean - scale[c] * smean
        log.info("radiometric correction: per-band gain %s", np.round(scale, 3).tolist())
        return scale, shift

    @torch.no_grad()
    def _net4x(self, tile_hwc: np.ndarray):
        """Return (mean4x HWC float32, std4x HW float32) over TTA ensemble."""
        outs = []
        for flip, rot in _augs(self.cfg.tta):
            aug = _apply(tile_hwc, flip, rot)
            t = torch.from_numpy(aug).permute(2, 0, 1).unsqueeze(0).to(self.device)
            sr = self.model(t).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            outs.append(_invert(sr, flip, rot))
        stack = np.stack(outs, 0)
        mean = stack.mean(0)
        std = stack.std(0).mean(-1) if len(outs) > 1 else np.zeros(mean.shape[:2], np.float32)
        return mean.astype(np.float32), std.astype(np.float32)

    def enhance(self, in_path: str, out_path: str,
                unc_path: Optional[str] = None) -> dict:
        cfg = self.cfg
        info = io_geo.read_info(in_path)
        if info.count != cfg.in_channels:
            raise ValueError(f"input has {info.count} bands but config expects "
                             f"{cfg.in_channels} ({cfg.band_order})")
        # per_band keeps input order; rgbn reorders to the model's canonical R,G,B,NIR
        self.perm = (list(range(cfg.in_channels)) if cfg.mode == "per_band"
                     else preprocess.permutation(cfg.band_order, cfg.in_channels))
        self.rc = self._compute_rc(in_path) if cfg.radiometric_correction else None
        scale = cfg.target_scale
        s4 = cfg.native_scale

        writer, (out_h, out_w) = io_geo.create_writer(
            out_path, info, scale, count=cfg.in_channels,
            dtype=cfg.out_dtype, compress=cfg.compress,
            nodata=(0 if cfg.out_dtype == "uint16" else None))
        unc_writer = None
        if unc_path and cfg.uncertainty and cfg.tta > 1:
            unc_writer, _ = io_geo.create_writer(
                unc_path, info, scale, count=1, dtype="float32",
                compress=cfg.compress)

        tile, pad = cfg.tile, cfg.tile_pad
        n_tiles = ((info.height + tile - 1) // tile) * ((info.width + tile - 1) // tile)
        log.info("scene %dx%d -> %dx%d (%.2fx), %d tiles, res %.1fm -> %.1fm",
                 info.width, info.height, out_w, out_h, scale, n_tiles,
                 info.res[0], info.res[0] / scale)

        done = 0
        for ry in range(0, info.height, tile):
            for rx in range(0, info.width, tile):
                th = min(tile, info.height - ry)
                tw = min(tile, info.width - rx)
                # padded read region (clipped to bounds)
                pr0, pc0 = max(ry - pad, 0), max(rx - pad, 0)
                pr1 = min(ry + th + pad, info.height)
                pc1 = min(rx + tw + pad, info.width)
                raw = io_geo.read_window(in_path, pr0, pc0, pr1 - pr0, pc1 - pc0)
                raw = raw[..., self.perm]                                  # -> model band order
                valid = preprocess.validity_mask(raw, cfg.nodata)
                refl = preprocess.to_reflectance(raw, cfg.dn_scale, cfg.input_is_dn)

                mean4, std4 = self._infer(refl)                           # 4x (rgbn or per_band)

                # crop core out of padded 4x output
                top4 = (ry - pr0) * s4
                left4 = (rx - pc0) * s4
                core4 = mean4[top4:top4 + th * s4, left4:left4 + tw * s4]
                stdc4 = std4[top4:top4 + th * s4, left4:left4 + tw * s4]

                # output block boundaries in scale-space (seamless integer grid)
                oy0, oy1 = int(round(ry * scale)), int(round((ry + th) * scale))
                ox0, ox1 = int(round(rx * scale)), int(round((rx + tw) * scale))
                oy1, ox1 = min(oy1, out_h), min(ox1, out_w)
                oh, ow = oy1 - oy0, ox1 - ox0
                if oh <= 0 or ow <= 0:
                    continue

                block = cv2.resize(core4, (ow, oh), interpolation=cv2.INTER_CUBIC)
                if self.rc is not None:                        # seam-safe global correction
                    block = np.clip(block * self.rc[0] + self.rc[1], 0, 1)
                out_block = preprocess.from_reflectance(block, cfg.dn_scale, cfg.out_dtype)

                # apply nodata from resized validity mask
                if cfg.nodata is not None and not valid.all():
                    vcore = valid[ry - pr0:ry - pr0 + th, rx - pc0:rx - pc0 + tw].astype(np.uint8)
                    vres = cv2.resize(vcore, (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)
                    fill = 0 if cfg.out_dtype == "uint16" else 0.0
                    out_block[~vres] = fill

                io_geo.write_window(writer, out_block, oy0, ox0)
                if unc_writer is not None:
                    ublock = cv2.resize(stdc4, (ow, oh), interpolation=cv2.INTER_LINEAR)
                    io_geo.write_window(unc_writer, ublock[..., None], oy0, ox0)

                done += 1
                if done % 10 == 0 or done == n_tiles:
                    log.info("  tile %d/%d", done, n_tiles)

        writer.close()
        if unc_writer is not None:
            unc_writer.close()
        peak = (torch.cuda.max_memory_allocated() / 1e9) if self.device == "cuda" else 0.0
        log.info("done -> %s%s (peak VRAM %.2f GB)", out_path,
                 f" + {unc_path}" if unc_writer is not None else "", peak)
        return {"out_path": out_path, "unc_path": unc_path if unc_writer is not None else None,
                "out_size": (out_h, out_w), "tiles": n_tiles, "peak_vram_gb": round(peak, 2)}
