"""
Shared helpers for the SATSR web application.

Pure functions (rendering, spectral indices, structure extraction, STAC ingest)
kept out of the Streamlit script so they stay testable and importable.
"""
from __future__ import annotations

import os
import sys
import glob
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

# --- make the satsr package importable regardless of cwd ------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)               # .../SATSR
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from satsr import io_geo                          # noqa: E402

DATA_CANDIDATES = [
    os.path.join(_PKG_ROOT, "data"),
    r"D:/Ai/data",
]
DN_SCALE = 10000.0
BANDS_RGB = (0, 1, 2)      # data is R,G,B,NIR
BAND_NIR = 3


# =========================================================================
# Scene registry — only scenes whose files actually exist are exposed
# =========================================================================
_SCENE_DEFS = [
    ("Demo — Bengaluru farmland (4-band)", "demo"),
    ("Mumbai / Dharavi — ultra-dense (hardest)", "mumbai"),
    ("Dense built-up core (auto-located)", "dense"),
    ("Random unseen crop", "random"),
    ("🔥 Wildfire — East Gippsland (6-band SWIR)", "wildfire"),
]


def _find_data_dir() -> Optional[str]:
    for d in DATA_CANDIDATES:
        if os.path.isdir(d):
            return d
    return None


def available_scenes() -> dict:
    """label -> {'input','sr','unc'} for scenes present on disk."""
    d = _find_data_dir()
    out = {}
    if not d:
        return out
    for label, key in _SCENE_DEFS:
        inp = os.path.join(d, f"{key}_input_10m.tif")
        sr = os.path.join(d, f"{key}_sr_4m.tif")
        unc = os.path.join(d, f"{key}_sr_4m_uncertainty.tif")
        if os.path.exists(inp) and os.path.exists(sr):
            out[label] = {
                "input": inp,
                "sr": sr,
                "unc": unc if os.path.exists(unc) else None,
            }
    return out


# =========================================================================
# Reading / rendering
# =========================================================================
def read_reflectance(path: str, max_dim: int = 900) -> np.ndarray:
    """(H,W,C) float reflectance overview of a scene (cheap, decimated)."""
    arr = io_geo.read_overview(path, max_dim=max_dim).astype(np.float32)
    return arr / DN_SCALE


def _stretch(x: np.ndarray, lo=2, hi=98) -> np.ndarray:
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / (b - a + 1e-6), 0, 1)


def to_rgb(refl: np.ndarray) -> np.ndarray:
    """True-colour uint8 (R,G,B) with a 2–98% per-image stretch."""
    r = np.stack([refl[..., c] for c in BANDS_RGB], -1)
    return (_stretch(r) * 255).astype(np.uint8)


def to_nir_false_color(refl: np.ndarray) -> np.ndarray:
    """Colour-IR (NIR,R,G) — vegetation glows red. uint8."""
    if refl.shape[-1] <= BAND_NIR:
        return to_rgb(refl)
    fc = np.stack([refl[..., BAND_NIR], refl[..., 0], refl[..., 1]], -1)
    return (_stretch(fc) * 255).astype(np.uint8)


def colorize(arr01: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """Map a [0,1] single-band array to an RGB uint8 image via a matplotlib cmap."""
    import matplotlib
    try:
        m = matplotlib.colormaps[cmap]           # matplotlib >= 3.9
    except Exception:
        import matplotlib.cm as cm
        m = cm.get_cmap(cmap)
    rgba = m(np.clip(arr01, 0, 1))
    return (rgba[..., :3] * 255).astype(np.uint8)


def uncertainty_rgb(path: str, max_dim: int = 900) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    u = io_geo.read_overview(path, max_dim=max_dim).astype(np.float32)
    if u.ndim == 3:
        u = u.mean(-1)
    u = _stretch(u, 1, 99)
    return colorize(u, "inferno")


# =========================================================================
# Spectral indices + zonal stats
# =========================================================================
def ndvi(refl: np.ndarray) -> np.ndarray:
    r, nir = refl[..., 0], refl[..., BAND_NIR]
    return (nir - r) / (nir + r + 1e-6)


def ndwi(refl: np.ndarray) -> np.ndarray:
    g, nir = refl[..., 1], refl[..., BAND_NIR]
    return (g - nir) / (g + nir + 1e-6)


def zonal_stats(index: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    v = index if mask is None else index[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }


# =========================================================================
# Structure / building extraction  —  YOLOv8-OBB (ultralytics)
#   Interface: RGB uint8 in -> (overlay, count, boxes) where boxes is a list of
#   oriented quads [(x,y)*4]. The model is configurable:
#       SATSR_YOLO_MODEL   .pt path or name  (default: yolov8n-obb.pt, DOTA-OBB)
#       SATSR_YOLO_DEVICE  cpu | cuda        (default: cpu — see note below)
#       SATSR_YOLO_CONF    confidence        (default: 0.15)
#   NOTE: point SATSR_YOLO_MODEL at a building-trained OBB checkpoint for true
#   building footprints; the default DOTA model detects overhead objects
#   (bridges, courts, tanks, vehicles, pools, ...). YOLO NMS is run on CPU by
#   default because torchvision's CUDA NMS op is unavailable in this torch build.
# =========================================================================
_FT_WEIGHTS = os.path.join(_PKG_ROOT, "weights", "yolov8m_building_ft.pt")       # our fine-tune
_BUILDING_WEIGHTS = os.path.join(_PKG_ROOT, "weights", "yolov8m_building_seg.pt")  # base building model


def _default_model_name() -> str:
    # prefer our fine-tuned building model, then the base building model,
    # then the DOTA-OBB overhead model as a last resort
    if os.path.exists(_FT_WEIGHTS):
        return _FT_WEIGHTS
    if os.path.exists(_BUILDING_WEIGHTS):
        return _BUILDING_WEIGHTS
    return "yolov8n-obb.pt"


_YOLO_MODEL = None
_YOLO_NAME = os.environ.get("SATSR_YOLO_MODEL", _default_model_name())
_YOLO_DEVICE = os.environ.get("SATSR_YOLO_DEVICE", "cpu")
_YOLO_CONF = float(os.environ.get("SATSR_YOLO_CONF", "0.12"))
# tiled inference: medium-res buildings are only a few px at 4 m/px, so we cut the
# scene into tiles and upscale each to the model's native scale. 0 disables tiling.
_YOLO_TILE = int(os.environ.get("SATSR_YOLO_TILE", "256"))
_YOLO_UP = int(os.environ.get("SATSR_YOLO_UP", "640"))


def model_label() -> str:
    n = os.path.basename(_YOLO_NAME).lower()
    if "building_ft" in n:
        return "YOLOv8 building-seg (fine-tuned)"
    if "building" in n:
        return "YOLOv8 building-seg (base)"
    if "obb" in n:
        return "YOLOv8-OBB (DOTA)"
    return os.path.basename(_YOLO_NAME)


def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(_YOLO_NAME)
    return _YOLO_MODEL


def _quads_from_result(res, ox, oy, sx, sy) -> list:
    """Extract detections from a YOLO result, mapped to global coords."""
    quads = []
    if getattr(res, "obb", None) is not None and len(res.obb):
        for poly in res.obb.xyxyxyxy.cpu().numpy():          # (4,2)
            quads.append([[ox + px * sx, oy + py * sy] for px, py in poly])
    elif getattr(res, "boxes", None) is not None and len(res.boxes):
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
            quads.append([[ox + x1 * sx, oy + y1 * sy], [ox + x2 * sx, oy + y1 * sy],
                          [ox + x2 * sx, oy + y2 * sy], [ox + x1 * sx, oy + y2 * sy]])
    return quads


def _predict_tiled(model, rgb, conf, tile, up) -> list:
    import cv2
    H, W, _ = rgb.shape
    if tile <= 0:                       # whole-image inference
        res = model.predict(rgb, imgsz=max(up, 640), conf=conf,
                            device=_YOLO_DEVICE, verbose=False)[0]
        return _quads_from_result(res, 0, 0, 1.0, 1.0)
    quads = []
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            ph, pw = min(tile, H - y), min(tile, W - x)
            if ph < 32 or pw < 32:
                continue
            big = cv2.resize(rgb[y:y + ph, x:x + pw], (up, up),
                             interpolation=cv2.INTER_CUBIC)
            res = model.predict(big, imgsz=up, conf=conf,
                                device=_YOLO_DEVICE, verbose=False)[0]
            quads += _quads_from_result(res, x, y, pw / up, ph / up)
    return quads


def yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def _draw_quads(rgb_uint8, quads, color=(52, 211, 153)) -> np.ndarray:
    import cv2
    overlay = np.ascontiguousarray(rgb_uint8.copy())
    for q in quads:
        pts = np.asarray(q, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
    return overlay


def extract_structures(rgb_uint8: np.ndarray, conf: float = None,
                       imgsz: int = 1024) -> tuple:
    """
    Detect structures with YOLOv8-OBB and return (overlay, count, boxes).
    Falls back to a classical detector only if ultralytics/the model is
    unavailable (e.g. offline first run), so the app never hard-crashes.
    """
    if not yolo_available():
        return _extract_structures_classical(rgb_uint8)
    try:
        model = _get_yolo()
        c = conf if conf is not None else _YOLO_CONF
        quads = _predict_tiled(model, rgb_uint8, c, _YOLO_TILE, _YOLO_UP)
        quads = _dedupe_quads(quads)
        overlay = _draw_quads(rgb_uint8, quads)
        return overlay, len(quads), quads
    except Exception:
        # e.g. CUDA-NMS build issue — degrade gracefully rather than break the UI
        return _extract_structures_classical(rgb_uint8)


def _dedupe_quads(quads, iou_thr: float = 0.45) -> list:
    """Greedy NMS on axis-aligned bounds to drop duplicates across tile seams."""
    if not quads:
        return quads
    boxes = []
    for q in quads:
        a = np.asarray(q)
        boxes.append([a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()])
    boxes = np.asarray(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou < iou_thr]
    return [quads[i] for i in keep]


def _extract_structures_classical(rgb_uint8: np.ndarray,
                                  min_area: int = 12, max_area_frac: float = 0.02) -> tuple:
    """Fallback compact-blob detector (scikit-image) — used only if YOLO can't load."""
    from skimage.color import rgb2gray
    from skimage.filters import threshold_local, sobel
    from skimage.measure import label, regionprops
    from skimage.morphology import remove_small_objects, closing, square

    gray = rgb2gray(rgb_uint8)
    edge = sobel(gray)
    edge = (edge - edge.min()) / (np.ptp(edge) + 1e-6)
    thr = threshold_local(gray, block_size=31, offset=-0.01)
    binf = (gray > thr) | (edge > 0.25)
    binf = closing(binf, square(2))
    binf = remove_small_objects(binf, min_size=min_area)

    lab = label(binf)
    H, W = gray.shape
    max_area = int(max_area_frac * H * W)
    quads = []
    for rp in regionprops(lab):
        if min_area <= rp.area <= max_area:
            minr, minc, maxr, maxc = rp.bbox
            hh, ww = maxr - minr, maxc - minc
            if 0.25 <= (hh / (ww + 1e-6)) <= 4.0:
                quads.append([[minc, minr], [maxc, minr], [maxc, maxr], [minc, maxr]])
    overlay = _draw_quads(rgb_uint8, quads)
    return overlay, len(quads), quads


# =========================================================================
# Geospatial bounds (for map overlays)
# =========================================================================
@dataclass
class SceneGeo:
    west: float
    south: float
    east: float
    north: float
    center: tuple
    crs: str
    res_m: float


def scene_geo(path: str) -> Optional[SceneGeo]:
    from rasterio.warp import transform_bounds
    info = io_geo.read_info(path)
    if info.crs is None:
        return None
    t = info.transform
    left, top = t.c, t.f
    right = left + t.a * info.width
    bottom = top + t.e * info.height
    try:
        w, s, e, n = transform_bounds(info.crs, "EPSG:4326", left, bottom, right, top)
    except Exception:
        return None
    return SceneGeo(w, s, e, n, ((s + n) / 2, (w + e) / 2),
                    str(info.crs), float(abs(t.a)))


# =========================================================================
# STAC live ingest — public AWS Sentinel-2 L2A (Element84 Earth Search).
# No credentials required. Reads a bbox window of R,G,B,NIR at 10 m and writes
# a georeferenced GeoTIFF ready for `satsr enhance`.
# =========================================================================
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"


def stac_search(bbox, date_range="2023-01-01/2024-12-31",
                max_cloud=10, limit=8) -> list:
    """Return a list of dicts: {id,date,cloud,item} sorted by cloud cover."""
    from pystac_client import Client
    cat = Client.open(EARTH_SEARCH)
    search = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=limit,
    )
    items = list(search.items())
    rows = []
    for it in items:
        rows.append({
            "id": it.id,
            "date": str(it.datetime.date()) if it.datetime else "?",
            "cloud": float(it.properties.get("eo:cloud_cover", -1)),
            "item": it,
        })
    rows.sort(key=lambda r: r["cloud"])
    return rows


# GDAL/CURL tuning for fast remote COG range reads (big speedup over defaults)
_COG_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",       # skip listing the bucket dir
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.jp2",
    GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_VERSION="2",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    VSI_CACHE="TRUE", VSI_CACHE_SIZE="67108864",
    GDAL_NUM_THREADS="ALL_CPUS",
)


def _read_band(href, bbox, max_px):
    """Read one bbox window from a remote COG; returns (data_uint16, crs, transform)."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds, Window
    with rasterio.open(href) as ds:
        l, b, r, t = transform_bounds("EPSG:4326", ds.crs, *bbox)
        win = from_bounds(l, b, r, t, ds.transform).round_offsets().round_lengths()
        w = int(min(win.width, max_px)); h = int(min(win.height, max_px))
        win = Window(win.col_off, win.row_off, w, h)
        data = ds.read(1, window=win, out_dtype="uint16")
        return data, ds.crs, ds.window_transform(win)


def stac_ingest(item, bbox, out_path: str, max_px: int = 512) -> dict:
    """
    Read a bbox window of B04,B03,B02,B08 (R,G,B,NIR, 10 m) from the item's COGs
    and write a 4-band GeoTIFF (DN uint16). Bands are fetched concurrently.
    """
    import rasterio
    from concurrent.futures import ThreadPoolExecutor

    hrefs = [item.assets[k].href for k in ("red", "green", "blue", "nir")]
    with rasterio.Env(**_COG_ENV):
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(lambda h: _read_band(h, bbox, max_px), hrefs))
    stacks = [r[0] for r in results]
    crs, transform = results[0][1], results[0][2]
    arr = np.stack(stacks, 0)   # (4,h,w) R,G,B,NIR
    prof = {"driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2],
            "count": 4, "dtype": "uint16", "crs": crs, "transform": transform,
            "compress": "deflate"}
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr)
    return {"path": out_path, "size": (arr.shape[1], arr.shape[2]), "crs": str(crs)}


# =========================================================================
# STAC live ingest — Copernicus Data Space Ecosystem (CDSE).
#   Search is public (no auth). Band pixels live in the CDSE `eodata` S3 bucket,
#   so windowed reads need S3 credentials (generate them at dataspace.copernicus.eu
#   -> User settings -> S3 credentials). The username/password OAuth token is for
#   the catalogue/OData APIs, not efficient per-band windowed reads.
# =========================================================================
CDSE_STAC = "https://stac.dataspace.copernicus.eu/v1"
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"


def _rfc3339_range(date_range: str) -> str:
    """'2023-01-01/2024-12-31' -> full RFC3339 range CDSE requires."""
    def fix(d):
        d = d.strip()
        return d if "T" in d else d + "T00:00:00Z"
    if "/" in date_range:
        a, b = date_range.split("/", 1)
        return f"{fix(a)}/{fix(b)}"
    return fix(date_range)


def cdse_stac_search(bbox, date_range="2023-01-01/2024-12-31",
                     max_cloud=10, limit=8) -> list:
    """Public CDSE STAC search. Returns rows {id,date,cloud,item,source} by cloud."""
    import json, urllib.request, urllib.parse
    q = urllib.parse.urlencode({
        "bbox": ",".join(str(x) for x in bbox),
        "datetime": _rfc3339_range(date_range),
        "limit": str(max(limit * 5, 30)),
    })
    url = f"{CDSE_STAC}/collections/sentinel-2-l2a/items?{q}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        fc = json.load(r)
    rows = []
    for it in fc.get("features", []):
        cc = float(it.get("properties", {}).get("eo:cloud_cover", -1))
        if cc >= 0 and cc > max_cloud:
            continue
        dt = (it.get("properties", {}).get("datetime", "") or "")
        rows.append({"id": it.get("id", ""), "date": dt[:10] or "?",
                     "cloud": cc, "item": it, "source": "cdse"})
    rows.sort(key=lambda r: (r["cloud"] if r["cloud"] >= 0 else 999))
    return rows[:limit]


def _read_kv_file(path: str) -> dict:
    kv = {}
    with open(path, "r", encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            for sep in (":", "="):
                if sep in line:
                    k, v = line.split(sep, 1)
                    kv[k.strip().lower()] = v.strip().strip('"').strip("'")
                    break
    return kv


def _load_cdse_s3_creds(env_file: str | None = None) -> dict:
    """S3 key/secret from env vars, else the credentials file. Never logged."""
    key = os.environ.get("CDSE_S3_ACCESS_KEY") or os.environ.get("AWS_ACCESS_KEY_ID")
    sec = os.environ.get("CDSE_S3_SECRET_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if key and sec:
        return {"key": key, "secret": sec}
    path = env_file or os.environ.get("CDSE_ENV_FILE")
    if not path:
        for c in ("copernicus.env",
                  os.path.join(os.path.dirname(os.path.dirname(__file__)), "copernicus.env"),
                  os.path.expanduser("~/copernicus.env")):
            if os.path.exists(c):
                path = c
                break
    if path and os.path.exists(path):
        kv = _read_kv_file(path)
        key = key or next((kv[k] for k in ("s3_access_key", "access_key",
                          "aws_access_key_id", "s3_key") if k in kv), None)
        sec = sec or next((kv[k] for k in ("s3_secret_key", "secret_key",
                          "aws_secret_access_key", "s3_secret") if k in kv), None)
    if key and sec:
        return {"key": key, "secret": sec}
    raise RuntimeError(
        "CDSE S3 credentials not found. Generate S3 keys at dataspace.copernicus.eu "
        "(User settings -> S3 credentials) and add 'S3_access_key:' + 'S3_secret_key:' "
        "lines to copernicus.env, or set CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY.")


def cdse_stac_ingest(item, bbox, out_path: str, max_px: int = 512,
                     env_file: str | None = None) -> dict:
    """Read a bbox window of B04,B03,B02,B08 (R,G,B,NIR, 10 m) from CDSE S3."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds, Window

    creds = _load_cdse_s3_creds(env_file)
    band_keys = ["B04_10m", "B03_10m", "B02_10m", "B08_10m"]     # R,G,B,NIR
    stacks, prof = [], None
    with rasterio.Env(AWS_S3_ENDPOINT=CDSE_S3_ENDPOINT, AWS_HTTPS="YES",
                      AWS_VIRTUAL_HOSTING="FALSE",
                      AWS_ACCESS_KEY_ID=creds["key"],
                      AWS_SECRET_ACCESS_KEY=creds["secret"],
                      GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for k in band_keys:
            href = item["assets"][k]["href"]                    # s3://eodata/...
            with rasterio.open(href) as ds:
                l, b, r, t = transform_bounds("EPSG:4326", ds.crs, *bbox)
                win = from_bounds(l, b, r, t, ds.transform).round_offsets().round_lengths()
                h = int(min(win.height, max_px))
                w = int(min(win.width, max_px))
                win = Window(win.col_off, win.row_off, w, h)
                data = ds.read(1, window=win)
                if prof is None:
                    prof = {"driver": "GTiff", "height": data.shape[0],
                            "width": data.shape[1], "count": 4, "dtype": "uint16",
                            "crs": ds.crs, "transform": ds.window_transform(win),
                            "compress": "deflate"}
                stacks.append(data.astype(np.uint16))
    arr = np.stack(stacks, 0)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr)
    return {"path": out_path, "size": (arr.shape[1], arr.shape[2]), "crs": str(prof["crs"])}


# ---- source dispatch (AWS Earth Search vs CDSE) -------------------------
SEARCH_SOURCES = {
    "AWS Earth Search (no login)": "aws",
    "Copernicus CDSE (your account)": "cdse",
}


def search(source, bbox, date_range="2023-01-01/2024-12-31", max_cloud=10, limit=8):
    if source == "cdse":
        return cdse_stac_search(bbox, date_range, max_cloud, limit)
    rows = stac_search(bbox, date_range, max_cloud, limit)
    for r in rows:
        r["source"] = "aws"
    return rows


def ingest(row, bbox, out_path, max_px=512):
    if row.get("source") == "cdse":
        # Route Copernicus through Sentinel Hub (OAuth client) - server-side, fast,
        # and no S3 keys required. Ingest the selected scene's day for this bbox.
        date = row.get("date")
        return sh_ingest(bbox, date, date, out_path, size=min(max_px, 512))
    return stac_ingest(row["item"], bbox, out_path, max_px)


# ---- Sentinel Hub Process API (OAuth client) - best S3 alternative ------
SH_PROCESS = "https://sh.dataspace.copernicus.eu/api/v1/process"
_SH_EVALSCRIPT = (
    "//VERSION=3\n"
    "function setup(){return {input:[\"B04\",\"B03\",\"B02\",\"B08\"],"
    "output:{bands:4,sampleType:\"UINT16\"}};}\n"
    "function evaluatePixel(s){return [s.B04*10000,s.B03*10000,s.B02*10000,s.B08*10000];}"
)


# full Sentinel-2 L2A stack (B10 cirrus is not in L2A) = 12 bands
SH_BANDS_12 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
               "B08", "B8A", "B09", "B11", "B12"]


def _sh_evalscript(bands):
    inp = ",".join(f'"{b}"' for b in bands)
    ret = ",".join(f"s.{b}*10000" for b in bands)
    return ("//VERSION=3\n"
            f"function setup(){{return {{input:[{inp}],"
            f"output:{{bands:{len(bands)},sampleType:\"UINT16\"}}}};}}\n"
            f"function evaluatePixel(s){{return [{ret}];}}")


def sh_ingest_bands(bbox, date_from, date_to, out_path, bands=None,
                    size=512, max_cloud=40, env_file=None):
    """Sentinel Hub Process API -> georeferenced N-band GeoTIFF (default: 12-band L2A)."""
    import json, urllib.request
    from cdse_auth import CDSEClientAuth
    bands = bands or SH_BANDS_12
    auth = CDSEClientAuth(env_file)
    body = {
        "input": {"bounds": {"bbox": [float(x) for x in bbox],
                             "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                  "data": [{"type": "sentinel-2-l2a",
                            "dataFilter": {"timeRange": {"from": f"{date_from}T00:00:00Z",
                                                          "to": f"{date_to}T23:59:59Z"},
                                           "maxCloudCoverage": max_cloud},
                            "mosaickingOrder": "leastCC"}]},
        "output": {"width": size, "height": size,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": _sh_evalscript(bands),
    }
    req = urllib.request.Request(SH_PROCESS, data=json.dumps(body).encode(),
                                 headers={**auth.auth_header(),
                                          "Content-Type": "application/json",
                                          "Accept": "image/tiff"})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Sentinel Hub {e.code}: {e.read().decode()[:300]}") from None
    with open(out_path, "wb") as f:
        f.write(data)
    import rasterio
    with rasterio.open(out_path) as ds:
        return {"path": out_path, "size": (ds.height, ds.width),
                "bands": ds.count, "crs": str(ds.crs)}


def sh_ingest(bbox, date_from, date_to, out_path, size=512, max_cloud=40, env_file=None):
    """Sentinel Hub Process API -> georeferenced 4-band (R,G,B,NIR) GeoTIFF for the bbox."""
    import json, urllib.request
    from cdse_auth import CDSEClientAuth
    auth = CDSEClientAuth(env_file)
    body = {
        "input": {"bounds": {"bbox": [float(x) for x in bbox],
                             "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                  "data": [{"type": "sentinel-2-l2a",
                            "dataFilter": {"timeRange": {"from": f"{date_from}T00:00:00Z",
                                                          "to": f"{date_to}T23:59:59Z"},
                                           "maxCloudCoverage": max_cloud},
                            "mosaickingOrder": "leastCC"}]},
        "output": {"width": size, "height": size,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": _SH_EVALSCRIPT,
    }
    req = urllib.request.Request(SH_PROCESS, data=json.dumps(body).encode(),
                                 headers={**auth.auth_header(),
                                          "Content-Type": "application/json",
                                          "Accept": "image/tiff"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Sentinel Hub {e.code}: {e.read().decode()[:300]}") from None
    with open(out_path, "wb") as f:
        f.write(data)
    import rasterio
    with rasterio.open(out_path) as ds:
        return {"path": out_path, "size": (ds.height, ds.width), "crs": str(ds.crs)}


# =========================================================================
# Coordinate parsing — accept decimal OR DMS, single point "lat, lon"
# =========================================================================
def parse_latlon(text: str):
    """
    Parse a 'lat, lon' string in decimal or DMS into (lat, lon) decimal degrees.
    Accepts e.g.  '27.38058, 33.63184'  or  '27°22\'50.10"N, 33°37\'54.62"E'.
    Returns (lat, lon) or None if it can't be parsed.
    """
    import re
    if not text:
        return None
    s = text.strip().replace("''", '"')
    # try plain 'lat, lon' decimal first
    m = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", s)
    hemis = re.findall(r"[NSEWnsew]", s)

    def dms(chunk):
        nums = re.findall(r"\d+\.\d+|\d+", chunk)
        if not nums:
            return None
        val = float(nums[0])
        if len(nums) > 1:
            val += float(nums[1]) / 60.0
        if len(nums) > 2:
            val += float(nums[2]) / 3600.0
        h = re.search(r"[NSEWnsew]", chunk)
        if h and h.group(0).upper() in ("S", "W"):
            val = -val
        return val

    # split into two halves on comma if present
    if "," in s:
        a, b = s.split(",", 1)
        lat, lon = dms(a), dms(b)
        if lat is not None and lon is not None:
            return (lat, lon)
    # fallback: two decimals no comma
    if len(m) >= 2 and not hemis:
        return (float(m[0]), float(m[1]))
    return None


def point_bbox(lat: float, lon: float, size_km: float):
    """Return a (west, south, east, north) bbox of side `size_km` around a point."""
    import math
    dlat = (size_km / 2.0) / 111.0
    dlon = (size_km / 2.0) / (111.0 * max(math.cos(math.radians(lat)), 0.1))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]
