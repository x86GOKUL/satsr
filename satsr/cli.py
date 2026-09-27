"""
satsr command-line interface.

  satsr enhance  INPUT.tif -o OUTPUT.tif [--scale 2.5] [--channels 4] ...
  satsr info     INPUT.tif
  satsr evaluate --sr SR.tif --ref REF.tif
"""
from __future__ import annotations

import argparse
import sys

from .config import SatSRConfig
from .logging_utils import get_logger

log = get_logger("cli")


def _add_common(p):
    p.add_argument("--mode", default=None, choices=("rgbn", "per_band"),
                   help="rgbn=4-band joint model; per_band=any N bands (SWIR/red-edge)")
    p.add_argument("--channels", type=int, default=None,
                   help="number of input bands (rgbn: 3 or 4; per_band: any N)")
    p.add_argument("--scale", type=float, default=None, help="target upscale (default 2.5)")
    p.add_argument("--band-order", default=None,
                   help="comma list matching input band order, e.g. red,green,blue,nir")
    p.add_argument("--device", default=None, choices=("auto", "cuda", "cpu"))
    p.add_argument("--tile", type=int, default=None)
    p.add_argument("--tta", type=int, default=None, choices=(1, 4, 8))
    p.add_argument("--out-dtype", default=None, choices=("uint16", "float32"))
    p.add_argument("--weights", default=None)
    p.add_argument("--config", default=None, help="YAML/JSON config file")
    p.add_argument("--dn-scale", type=float, default=None)


def _cfg_from_args(a) -> SatSRConfig:
    base = SatSRConfig.from_file(a.config) if getattr(a, "config", None) else SatSRConfig()
    overrides = dict(
        mode=a.mode, in_channels=a.channels, target_scale=a.scale, device=a.device,
        tile=a.tile, tta=a.tta, out_dtype=a.out_dtype, weights=a.weights,
        dn_scale=a.dn_scale,
    )
    if a.band_order:
        overrides["band_order"] = [b.strip().lower() for b in a.band_order.split(",")]
    return SatSRConfig.from_overrides(base, **overrides)


def cmd_enhance(a) -> int:
    from .pipeline import enhance_scene
    cfg = _cfg_from_args(a)
    if a.no_uncertainty:
        cfg.uncertainty = False
    if a.no_radiometric_correction:
        cfg.radiometric_correction = False
    res = enhance_scene(a.input, a.output, cfg, a.uncertainty_out)
    log.info("SUMMARY: %s", res)
    return 0


def cmd_info(a) -> int:
    from . import io_geo
    info = io_geo.read_info(a.input)
    print(f"path      : {a.input}")
    print(f"size      : {info.width} x {info.height} px, {info.count} bands ({info.dtype})")
    print(f"CRS       : {info.crs}")
    print(f"resolution: {info.res[0]:.3f} x {info.res[1]:.3f} (map units/px)")
    print(f"nodata    : {info.nodata}")
    print(f"transform : {info.transform}")
    return 0


def cmd_evaluate(a) -> int:
    import numpy as np
    from . import io_geo
    from .metrics import psnr, ssim_rgb, sam, ergas
    sr = io_geo.read_all(a.sr).astype(np.float32)
    ref = io_geo.read_all(a.ref).astype(np.float32)
    if sr.shape != ref.shape:
        import cv2
        sr = cv2.resize(sr, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
    mx = max(sr.max(), ref.max()) or 1.0
    sr_n, ref_n = sr / mx, ref / mx
    print(f"PSNR : {psnr(ref_n, sr_n):.3f} dB")
    print(f"SSIM : {ssim_rgb(ref_n, sr_n):.4f}")
    print(f"SAM  : {sam(ref_n, sr_n):.4f} deg")
    print(f"ERGAS: {ergas(ref_n, sr_n, a.scale):.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("satsr", description="Sentinel-2 super-resolution (10m -> <4m)")
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    e = sub.add_parser("enhance", help="super-resolve a scene")
    e.add_argument("input")
    e.add_argument("-o", "--output", required=True)
    e.add_argument("--uncertainty-out", default=None)
    e.add_argument("--no-uncertainty", action="store_true")
    e.add_argument("--no-radiometric-correction", action="store_true",
                   help="disable the per-band radiometric (colour-cast) correction")
    _add_common(e)
    e.set_defaults(func=cmd_enhance)

    i = sub.add_parser("info", help="print raster geospatial metadata")
    i.add_argument("input")
    i.set_defaults(func=cmd_info)

    v = sub.add_parser("evaluate", help="metrics between an SR product and a reference")
    v.add_argument("--sr", required=True)
    v.add_argument("--ref", required=True)
    v.add_argument("--scale", type=float, default=2.5)
    v.set_defaults(func=cmd_evaluate)
    return p


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if getattr(a, "version", False):
        from . import __version__
        print(f"satsr {__version__}")
        return 0
    if not getattr(a, "cmd", None):
        p.print_help()
        return 1
    try:
        return a.func(a)
    except Exception as e:  # production: fail with a clean message, non-zero code
        log.error("%s: %s", type(e).__name__, e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
