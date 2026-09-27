# SATSR Lab — interactive web application

An application layer on top of the **SATSR** super-resolution framework. Three
capabilities, one Streamlit app:

| Feature | What it does |
|---------|--------------|
| 🖥️ **Web GIS** | Before/after **split-slider** (true-colour, NIR false-colour, uncertainty), georeferenced **map overlay** (folium), live **geo-integrity** readout (CRS, pixel size), one-click **GeoTIFF download**, and a sidebar **output-resolution control** — render live at **4 m (2.5×)**, **2.5 m (4×)** or **1 m (10×)** (applies to every tab). |
| 🌾 **Downstream Action** | Full **downstream suite** (6 modes) — **🏢 Real footprints (OSM)** (authoritative surveyed vectors draped on the SR product + GeoJSON), **🏗️ Built-up area (NDBI)** (spectral built-up index — real NDBI on SWIR scenes, proxy otherwise), **🛣️ Roads/linear** (Frangi ridge centrelines), **🌊 Water & coastlines** (NDWI+Otsu), **🌾 Agri boundaries** (NDVI Sobel gradient), **🏠 YOLO footprints (experimental)** (fine-tuned detector — over-triggers at Sentinel resolution, flagged + OOD-gated). |
| ☁️ **STAC Live Ingest** | Choose a preset, paste a **📍 custom point** (decimal *or* DMS, e.g. `27°22'50"N, 33°37'54"E`) + box size, or a **custom bbox** → search **AWS Sentinel-2 L2A** (Element84 Earth Search, *no credentials*) → ingest a R,G,B,NIR window → **super-resolve** it in-app. |

## Run

From the SATSR project root:

```bash
pip install -r app/requirements-app.txt
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501.

- Demo scenes are auto-discovered from `data/` (or `D:/Ai/data`).
- The **Live Ingest** tab needs internet; everything else runs fully offline.
- Weights are resolved from `weights/` (or `$SATSR_WEIGHTS`), same as the CLI.

## Notes

- **Output resolution** — the sidebar renders any scene live at **4 m (2.5×)**,
  **2.5 m (4×)** or **1 m (10×)** via `enhance_scene` on GPU (cached per scene).
  The model's trained/native range is ~2.5–4×; **1 m (10×) is beyond that**, so
  the extra magnification is **interpolated/speculative, not recovered detail** —
  the UI flags this and points to the uncertainty layer. (20× / 0.5 m was removed.)

- **Building targets — honest positioning.** Per-building footprints are *not*
  reliably detectable at 10 m (a 10 m pixel is 100 m², larger than most
  buildings), so the app leads with two trustworthy options and demotes the YOLO
  detector to experimental:
  - **🏢 Real footprints (OSM)** *(recommended)* — pulls authoritative surveyed
    building polygons from OpenStreetMap (Overpass; 3 endpoint fallbacks, no auth)
    for the scene footprint, reprojects lon/lat → scene pixels, drapes them on the
    SR product, and exports **GeoJSON** (EPSG:4326, drag into QGIS). Tested: 49,824
    footprints on the Bengaluru dense core.
  - **🏗️ Built-up area (NDBI)** *(recommended)* — spectral built-up index, the
    analogue of NDWI/NDVI, no training. **Real NDBI = (SWIR B11 − NIR)/(B11 + NIR)**
    when a 6-band SWIR scene is loaded, else a no-SWIR proxy (bright · low-NDVI ·
    low-NDWI). Maps settlement *area* (what 10 m supports), not footprints.
  - **🏠 YOLO footprints (experimental)** — our fine-tuned `yolov8m_building_ft.pt`
    (see below). Kept for transparency but **over-triggers on field parcels at
    Sentinel resolution**; guarded by an **OOD gate** (union footprint density
    >55% ⇒ flagged "non-urban / OOD", vectors + count suppressed — e.g. bare
    desert) and an impervious (NDVI/NDWI) filter. Density is **union-based (≤100%)**.

- **The fine-tuned YOLO model** (`weights/yolov8m_building_ft.pt`, single
  `building` class) was fine-tuned on GPU (RTX 4060) from
  `keremberke/yolov8m-building-segmentation` on the labelled
  `satellite-building-segmentation` dataset (6,764 train / 1,934 val, ~89k
  polygons; COCO→YOLO-seg). Reproducible via `train_buildings.py`
  (`runs/building_ft/`). Validation Box **mAP@50 0.500 → 0.720** over 15 epochs.
  Inference is **tiled + upscaled** (tiles → model native scale → mapped back →
  NMS-deduped). **Resolution order:** `yolov8m_building_ft.pt` → base
  `yolov8m_building_seg.pt` → `yolov8n-obb.pt` → classical fallback. Env overrides:
  `SATSR_YOLO_MODEL`, `SATSR_YOLO_TILE` (256, `0`=off), `SATSR_YOLO_UP` (640),
  `SATSR_YOLO_CONF` (0.12), `SATSR_YOLO_DEVICE` (`cpu` — torchvision's CUDA NMS op
  is unavailable in this torch build).

- **Spectral / morphology downstream** (`downstream_suite.py`), no training:
  - **🛣️ Roads / linear** — Frangi ridge filter + `skeletonize`; connected linear
    length (km). A linear-feature map (also catches strong field edges).
  - **🌊 Water & coastlines** — NDWI + **Otsu** auto threshold; shoreline contour,
    water area (km²), largest-connected-body %.
  - **🌾 Agri field boundaries** — NDVI **Sobel gradient**; cites the validated
    **91%** edge recovery (vs bicubic 70%) from the stress battery.
  Water/field index metrics are resolution-robust, so live LR-vs-HR gains are
  strongest for buildings and roads (what SR actually sharpens).

- **STAC ingest** reads bands directly from cloud-optimised GeoTIFFs over the
  network (windowed), so only the AOI you select is downloaded. Coordinate entry
  accepts decimal or DMS via `satsr_app_utils.parse_latlon` + `point_bbox`.

## Files

```
app/
  streamlit_app.py       Streamlit UI (3 tabs; Web GIS resolution control; Downstream = 6-mode suite)
  satsr_app_utils.py     rendering, indices, YOLO extraction, STAC ingest, coord parsing (pure functions)
  downstream_suite.py    OSM footprints + GeoJSON + NDBI built-up + water/roads/fields
  requirements-app.txt   app-only dependencies (on top of the satsr package)
  README_APP.md          this file
train_buildings.py       GPU fine-tune script for the building model (+CPU-NMS patch)
weights/
  yolov8m_building_ft.pt  our fine-tuned building model (YOLO experimental mode)
  yolov8m_building_seg.pt base building model
runs/building_ft/        training run: metrics (results.csv) + weights/best.pt
```
