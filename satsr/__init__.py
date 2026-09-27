"""
satsr — Sentinel-2 Super-Resolution framework.

Production-grade 10 m -> <4 m super-resolution for medium-resolution satellite
imagery, preserving geospatial (CRS / geotransform) and spectral consistency,
with per-pixel uncertainty.

Public API:
    from satsr import enhance_scene, SatSRConfig
"""
from .config import SatSRConfig
from .pipeline import enhance_scene

__version__ = "1.0.0"
__all__ = ["SatSRConfig", "enhance_scene", "__version__"]
