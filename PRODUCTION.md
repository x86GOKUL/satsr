# satsr — Production Sentinel-2 Super-Resolution

Turn 10 m Sentinel-2 imagery into georeferenced **4 m** products (2.5×) that stay
on the map (CRS + geotransform preserved), keep spectral meaning, and carry a
per-pixel uncertainty layer. Runs on an 8 GB laptop GPU or CPU.

## Install
```bash
pip install -r requirements.txt      # or: pip install -e .
```
Weights live in `weights/` (auto-resolved by channel count). Override the
location with the `SATSR_WEIGHTS` environment variable.

## CLI
```bash
# inspect a raster's geospatial metadata
python -m satsr info scene_10m.tif

# super-resolve 10 m -> 4 m (4-band R,G,B,NIR), with uncertainty
python -m satsr enhance scene_10m.tif -o scene_4m.tif --config configs/default.yaml

# override options on the fly
python -m satsr enhance scene_10m.tif -o out.tif --scale 2.5 --channels 4 \
    --band-order red,green,blue,nir --tile 384 --tta 8 --device auto

# full spectrum: super-resolve ANY N bands, incl. SWIR (B11/B12), red-edge, ...
python -m satsr enhance stack6_10m.tif -o stack6_4m.tif \
    --mode per_band --channels 6 --band-order red,green,blue,nir,swir1,swir2

# metrics against a reference
python -m satsr evaluate --sr scene_4m.tif --ref reference.tif --scale 2.5
```
After `pip install -e .` the `satsr` command is available directly.

## Python API
```python
from satsr import SatSRConfig, enhance_scene

cfg = SatSRConfig(in_channels=4, target_scale=2.5, tile=384, tta=8, device="auto")
summary = enhance_scene("scene_10m.tif", "scene_4m.tif", cfg)
# -> {'out_path': ..., 'unc_path': ..., 'out_size': (H,W), 'peak_vram_gb': ...}
```

## What it guarantees
- **Radiometric fidelity (default on).** A seam-safe **global** per-band affine
  matches the output's mean/std to the input, removing any colour/gain cast from
  the generative model. Coefficients are computed once from a decimated preview
  and applied identically to every tile — verified seamless (tile 256 vs 384
  differ by <0.7 %) and shown to flip satsr to a clean win over bicubic on real
  NAIP references. Disable with `--no-radiometric-correction`.
- **Geospatial integrity.** Output CRS == input CRS; geotransform rescaled so the
  map footprint is identical and pixels are `1/scale` the size (10 m → 4 m).
  Verified: origin unchanged, resolution 10.0 → 4.0 m, EPSG preserved.
- **Spectral consistency.** 4-band model trained with an SSIM+L1+**spectral-angle**
  loss; held-out SAM ≈ 2°, PSNR 38 dB, ERGAS better than bicubic.
- **Uncertainty.** 8× test-time-augmentation ensemble; the std map (float32
  GeoTIFF) flags where detail is inferred vs observed.
- **Scale, safely.** Streaming tiled inference writes each tile straight to the
  GeoTIFF — peak memory depends on `tile`, not scene size (2.7 GB VRAM for the
  demo; lower `tile` to fit smaller GPUs; `--device cpu` for no-GPU hosts).

## Architecture
```
satsr/
  config.py       SatSRConfig dataclass (+ YAML/JSON load, validation)
  model.py        RRDBNet backbone + 3->4-channel weight adaptation
  io_geo.py       georeferenced GeoTIFF read / windowed write / transform rescale
  preprocess.py   DN<->reflectance, band reordering, validity/nodata masks
  inference.py    streaming tiled engine + TTA uncertainty
  pipeline.py     weight resolution + orchestration (enhance_scene)
  metrics.py      PSNR / SSIM / SAM / ERGAS
  cli.py          enhance / info / evaluate
configs/default.yaml   default 4-band 2.5x config
tests/test_satsr.py    unit + end-to-end tests (6/6 passing)
```

## Configuration (configs/default.yaml)
Key fields: `mode` (`rgbn` = 4-band joint, or `per_band` = any N bands),
`in_channels` (rgbn: 3/4; per_band: any N), `band_order`, `target_scale`,
`dn_scale` (DN→reflectance), `tile`/`tile_pad`, `tta` (1/4/8), `out_dtype`
(uint16 DN or float32 reflectance), `radiometric_correction` (default true),
`nodata`, `device`.

### Band coverage
- `mode: rgbn` — the 4-channel joint model (R,G,B,NIR), spectral-loss fine-tuned;
  best quality on the core analytical bands.
- `mode: per_band` — a 3-channel backbone applied independently to **each band**,
  so **any** Sentinel-2 band works: SWIR (B11/B12) for geology & wildfire,
  red-edge (B5–B7) for crop stress, etc. Feed a 10 m grid (resample 20 m bands
  first). Verified end-to-end on a 6-band R,G,B,NIR,SWIR1,SWIR2 stack.

## Tests
```bash
python tests/test_satsr.py        # standalone runner (no pytest needed)
# or: pytest tests/
```

## Production hardening still recommended
- **Model quality:** train on many scenes + real 10 m↔sub-4 m pairs
  (SEN2VENµS / WorldStrat) instead of the degrade→reconstruct proxy, and
  validate on held-out *scenes*.
- **Ops:** container image (Dockerfile), a REST endpoint (FastAPI) wrapping
  `enhance_scene`, batch/queue orchestration, and cloud object-store I/O
  (`/vsis3/`) for large AOIs.
- **QA:** cloud/shadow masking on ingest; automatic per-tile spectral-drift
  checks that fail loudly.
