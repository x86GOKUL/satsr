"""
SATSR downstream suite — the full feature triangle beyond building boxes:

  1. Building FOOTPRINT POLYGONS (YOLOv8-seg masks) + area/density + GeoJSON export
  2. Water edges        (NDWI + Otsu threshold, sub-pixel shoreline contour)
  3. Narrow roads       (Frangi ridge filter + skeleton centrelines)
  4. Field boundaries   (NDVI Sobel gradient)

All lightweight — only #1 uses the (already-trained) YOLO model; 2-4 are indices +
morphology, no extra training. Imports the shared helpers from satsr_app_utils.
"""
from __future__ import annotations

import json
import numpy as np

import satsr_app_utils as U
from satsr import io_geo


# ---- geospatial helpers (overview-pixel -> map) -------------------------
def overview_geo(path: str, max_dim: int = 900):
    from affine import Affine
    info = io_geo.read_info(path)
    dec = max(1, max(info.width, info.height) // max_dim)
    return info.transform * Affine.scale(dec, dec), info.crs, dec


def pixel_area_m2(path: str, max_dim: int = 900):
    t, crs, _ = overview_geo(path, max_dim)
    try:
        proj = crs is not None and crs.is_projected
    except Exception:
        proj = False
    return abs(t.a * t.e), bool(proj)


# ========================================================================
# 1. BUILDING FOOTPRINT POLYGONS
# ========================================================================
def _predict_tiled_polys(model, rgb, conf, tile, up):
    import cv2
    H, W, _ = rgb.shape
    polys = []

    def collect(res, ox, oy, sx, sy):
        m = getattr(res, "masks", None)
        if m is not None and m.xy is not None and len(m.xy):
            for p in m.xy:
                if len(p) >= 3:
                    q = np.asarray(p, dtype=np.float32).copy()
                    q[:, 0] = ox + q[:, 0] * sx
                    q[:, 1] = oy + q[:, 1] * sy
                    polys.append(q)
        elif getattr(res, "boxes", None) is not None and len(res.boxes):
            for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
                polys.append(np.array([[ox+x1*sx, oy+y1*sy], [ox+x2*sx, oy+y1*sy],
                                       [ox+x2*sx, oy+y2*sy], [ox+x1*sx, oy+y2*sy]], np.float32))

    if tile <= 0:
        res = model.predict(rgb, imgsz=max(up, 640), conf=conf, device=U._YOLO_DEVICE, verbose=False)[0]
        collect(res, 0, 0, 1.0, 1.0)
        return polys
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            ph, pw = min(tile, H - y), min(tile, W - x)
            if ph < 32 or pw < 32:
                continue
            big = cv2.resize(rgb[y:y+ph, x:x+pw], (up, up), interpolation=cv2.INTER_CUBIC)
            res = model.predict(big, imgsz=up, conf=conf, device=U._YOLO_DEVICE, verbose=False)[0]
            collect(res, x, y, pw/up, ph/up)
    return polys


def _dedupe_polys(polys, iou_thr: float = 0.45):
    if not polys:
        return polys
    boxes = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys])
    areas = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
    order = np.argsort(-areas); keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2-xx1, 0, None) * np.clip(yy2-yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou < iou_thr]
    return [polys[i] for i in keep]


def extract_building_polys(rgb_uint8, conf: float = None):
    if not U.yolo_available():
        _, _, quads = U._extract_structures_classical(rgb_uint8)
        return [np.asarray(q, np.float32) for q in quads]
    try:
        model = U._get_yolo()
        c = conf if conf is not None else U._YOLO_CONF
        return _dedupe_polys(_predict_tiled_polys(model, rgb_uint8, c, U._YOLO_TILE, U._YOLO_UP))
    except Exception:
        _, _, quads = U._extract_structures_classical(rgb_uint8)
        return [np.asarray(q, np.float32) for q in quads]


def draw_polys(rgb_uint8, polys, color=(52, 211, 153), alpha=0.35):
    import cv2
    ov = np.ascontiguousarray(rgb_uint8.copy())
    if polys:
        layer = ov.copy()
        for p in polys:
            cv2.fillPoly(layer, [p.astype(np.int32)], color)
        ov = cv2.addWeighted(layer, alpha, ov, 1 - alpha, 0)
        for p in polys:
            cv2.polylines(ov, [p.astype(np.int32)], True, color, 2)
    return ov


def _poly_area_px(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


OOD_DENSITY_PCT = 55.0   # union footprint density above this = implausible -> OOD/non-urban


def building_stats(polys, path=None, shape=None, max_dim=900):
    """
    Footprint stats. Density uses the UNION of polygons (rasterised), so it can
    never exceed 100%; a union density above OOD_DENSITY_PCT flags a non-urban /
    out-of-distribution scene where the detector is unreliable (e.g. bare desert).
    """
    import cv2
    n = len(polys)
    indiv = [ _poly_area_px(p) for p in polys ] if polys else []
    out = {"count": n, "area_px": float(sum(indiv))}
    if path:
        ppa, proj = pixel_area_m2(path, max_dim)
        info = io_geo.read_info(path)
        dec = max(1, max(info.width, info.height) // max_dim)
        H, W = (shape[:2] if shape is not None else (info.height // dec, info.width // dec))
        # union coverage via rasterisation (fixes >100% double-counting)
        mask = np.zeros((H, W), np.uint8)
        for p in polys:
            cv2.fillPoly(mask, [p.astype(np.int32)], 1)
        union_px = int(mask.sum())
        density = 100.0 * union_px / (H * W) if H * W else 0.0
        out.update(projected=proj, footprint_m2=union_px * ppa, aoi_m2=ppa * H * W,
                   density_pct=density,
                   median_m2=(float(np.median(indiv)) * ppa if indiv else 0.0),
                   ood=(proj and density > OOD_DENSITY_PCT))
    return out


def polys_to_geojson(polys, path, max_dim=900):
    t, crs, _ = overview_geo(path, max_dim)
    warp = None
    if crs is not None:
        from rasterio.warp import transform as warp
    feats = []
    for i, p in enumerate(polys):
        xs, ys = [], []
        for col, row in p:
            X, Y = t * (float(col), float(row)); xs.append(X); ys.append(Y)
        if warp is not None:
            lon, lat = warp(crs, "EPSG:4326", xs, ys)
        else:
            lon, lat = xs, ys
        ring = [[float(a), float(b)] for a, b in zip(lon, lat)]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        feats.append({"type": "Feature",
                      "properties": {"id": i, "area_px": float(_poly_area_px(p))},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return json.dumps({"type": "FeatureCollection",
                       "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                       "features": feats})


# ========================================================================
# 2. WATER EDGES  (NDWI + Otsu)
# ========================================================================
def water_layer(refl):
    from skimage.filters import threshold_otsu
    nd = U.ndwi(refl)
    v = nd[np.isfinite(nd)]
    try:
        thr = max(0.0, float(threshold_otsu(v)))
    except Exception:
        thr = 0.0
    return nd > thr, nd, thr


def water_overlay(rgb_uint8, mask):
    import cv2
    ov = np.ascontiguousarray(rgb_uint8.copy())
    layer = ov.copy(); layer[mask] = (56, 189, 248)
    ov = cv2.addWeighted(layer, 0.4, ov, 0.6, 0)
    cnts, _ = cv2.findContours((mask.astype(np.uint8)) * 255,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (255, 255, 255), 1)
    return ov, cnts


def shoreline_continuity(cnts):
    import cv2
    if not cnts:
        return 0.0
    lens = [cv2.arcLength(c, True) for c in cnts]
    tot = sum(lens)
    return 100.0 * (max(lens) / tot) if tot else 0.0


def water_continuity(mask):
    """% of water pixels in the single largest connected body — HR reconnects
    the thin canals/shorelines that fragment into blobs at 10 m."""
    from skimage.measure import label
    tot = int(mask.sum())
    if tot == 0:
        return 0.0
    lab = label(mask)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return 100.0 * float(counts.max()) / tot


def water_area_m2(mask, path, max_dim=900):
    ppa, proj = pixel_area_m2(path, max_dim)
    return float(mask.sum()) * ppa, proj


# ========================================================================
# 3. NARROW ROADS  (Frangi ridge + skeleton)
# ========================================================================
def road_layer(rgb_uint8):
    from skimage.filters import frangi
    from skimage.morphology import skeletonize, remove_small_objects
    gray = rgb_uint8[..., :3].mean(-1) / 255.0
    ridges = frangi(gray, sigmas=range(1, 4), black_ridges=True)
    ridges = (ridges - ridges.min()) / (np.ptp(ridges) + 1e-6)
    mask = remove_small_objects(ridges > 0.12, 40)
    return skeletonize(mask), mask


def road_overlay(rgb_uint8, skel):
    ov = np.ascontiguousarray(rgb_uint8.copy())
    ov[skel] = (251, 191, 36)
    return ov


def road_length_km(skel, path, max_dim=900):
    ppa, proj = pixel_area_m2(path, max_dim)
    px_m = (abs(ppa) ** 0.5) if proj else 4.0
    return float(skel.sum()) * px_m / 1000.0, proj


# ========================================================================
# 4. FIELD BOUNDARIES  (NDVI Sobel gradient)
# ========================================================================
def field_edge_overlay(refl):
    from skimage.filters import sobel
    g = sobel(U.ndvi(refl))
    mag = float(np.nanmean(np.abs(g)))
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    return U.colorize(g, "magma"), mag


def filter_to_builtup(polys, refl, ndvi_max: float = 0.20, ndwi_max: float = 0.0):
    """
    Keep only footprints over IMPERVIOUS surface — reject polygons whose interior
    is vegetated (mean NDVI >= ndvi_max) or water (mean NDWI >= ndwi_max). Removes
    the agricultural-field false positives the sub-metre-trained model produces at
    Sentinel resolution. `refl` must match the polygon pixel grid.
    """
    import cv2
    nd = U.ndvi(refl); nw = U.ndwi(refl)
    H, W = nd.shape[:2]
    kept = []
    for p in polys:
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [p.astype(np.int32)], 1)
        m = mask.astype(bool)
        if m.sum() == 0:
            continue
        if float(np.nanmean(nd[m])) < ndvi_max and float(np.nanmean(nw[m])) < ndwi_max:
            kept.append(p)
    return kept


# ========================================================================
# 5. BUILT-UP AREA  (NDBI where SWIR present, else a no-SWIR proxy)
# ========================================================================
_SWIR1 = 4   # band order red,green,blue,nir,swir1,swir2


def builtup_index(refl):
    """
    Return (score01, mask, has_swir, coverage_pct).
      - >=6 bands: real NDBI = (SWIR1 - NIR)/(SWIR1 + NIR), built-up = NDBI > 0.
      - else: no-SWIR proxy = bright AND low-NDVI AND low-NDWI (impervious/grey).
    """
    from skimage.filters import threshold_otsu
    nd = U.ndvi(refl); nw = U.ndwi(refl)
    if refl.shape[-1] > _SWIR1:
        swir = refl[..., _SWIR1]; nir = refl[..., U.BAND_NIR]
        idx = (swir - nir) / (swir + nir + 1e-6)
        has = True
        v = idx[np.isfinite(idx)]
        try:
            thr = float(threshold_otsu(v))
        except Exception:
            thr = 0.0
        mask = idx > max(thr, 0.0)
        score = np.clip((idx + 1) / 2, 0, 1)
    else:
        bright = refl[..., :3].mean(-1)
        b = (bright - np.nanmin(bright)) / (np.ptp(bright) + 1e-6)
        # impervious/grey: bright, not vegetation, not water
        score = np.clip(b * (nd < 0.25) * (nw < 0.05), 0, 1)
        mask = (b > np.nanmedian(b)) & (nd < 0.20) & (nw < 0.0)
        has = False
    cov = 100.0 * float(mask.mean())
    return score, mask, has, cov


def builtup_overlay(rgb_uint8, score01, mask):
    import cv2
    heat = U.colorize(score01, "inferno")
    ov = cv2.addWeighted(heat, 0.45, rgb_uint8, 0.55, 0)
    return ov


# ========================================================================
# 6. REAL BUILDING FOOTPRINTS  (OpenStreetMap via Overpass — free, no auth)
# ========================================================================
OVERPASS = "https://overpass-api.de/api/interpreter"


def fetch_osm_buildings(bbox, timeout=60):
    """bbox = [west, south, east, north] WGS84. Returns list of Nx2 (lon,lat) rings."""
    import requests
    w, s, e, n = bbox
    q = (f"[out:json][timeout:25];("
         f"way['building']({s},{w},{n},{e});"
         f"relation['building']({s},{w},{n},{e}););out geom;")
    headers = {"User-Agent": "SATSR-Lab/1.0 (satellite super-resolution demo)",
               "Accept": "application/json"}
    endpoints = [OVERPASS, "https://overpass.kumi.systems/api/interpreter",
                 "https://overpass.openstreetmap.ru/api/interpreter"]
    r = None
    for ep in endpoints:
        try:
            r = requests.post(ep, data={"data": q}, headers=headers, timeout=timeout)
            r.raise_for_status()
            break
        except Exception:
            r = None
    if r is None:
        raise RuntimeError("all Overpass endpoints failed")
    polys = []
    for el in r.json().get("elements", []):
        if el.get("type") == "way" and el.get("geometry"):
            ring = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(ring) >= 3:
                polys.append(np.array(ring, np.float64))
        elif el.get("type") == "relation":
            for mem in el.get("members", []):
                if mem.get("geometry") and mem.get("role") == "outer":
                    ring = [(p["lon"], p["lat"]) for p in mem["geometry"]]
                    if len(ring) >= 3:
                        polys.append(np.array(ring, np.float64))
    return polys


def lonlat_to_pixels(polys_ll, path, max_dim=900):
    """Convert lon/lat rings to overview-pixel polygons for the given scene."""
    t, crs, _ = overview_geo(path, max_dim)
    inv = ~t
    warp = None
    if crs is not None:
        from rasterio.warp import transform as warp
    out = []
    for ring in polys_ll:
        lons = [float(p[0]) for p in ring]; lats = [float(p[1]) for p in ring]
        if warp is not None:
            xs, ys = warp("EPSG:4326", crs, lons, lats)
        else:
            xs, ys = lons, lats
        px = [inv * (float(X), float(Y)) for X, Y in zip(xs, ys)]
        out.append(np.array(px, np.float32))
    return out


def osm_footprints_for_scene(path, max_dim=900):
    """Fetch OSM buildings for a scene's footprint and return pixel polygons + count."""
    geo = U.scene_geo(path)
    if geo is None:
        return [], 0
    ll = fetch_osm_buildings([geo.west, geo.south, geo.east, geo.north])
    return lonlat_to_pixels(ll, path, max_dim), len(ll)


# ========================================================================
# 7. WILDFIRE (SWIR) — active fire + burn severity, with cloud masking
#    Needs >=6 bands (…, SWIR1=B11 idx4, SWIR2=B12 idx5).
#    Fire = thermal SWIR2 spike (B12 >> B11); Cloud = bright but flat-SWIR.
# ========================================================================
_SWIR2 = 5


def wildfire_layers(refl, fire_refl_thr=0.22, cloud_bright=0.40):
    if refl.shape[-1] <= _SWIR2:
        return None
    b4, b3, b2 = refl[..., 0], refl[..., 1], refl[..., 2]
    nir, b11, b12 = refl[..., U.BAND_NIR], refl[..., _SWIR1], refl[..., _SWIR2]
    fire_index = (b12 - b11) / (b12 + b11 + 1e-6)     # >0 => SWIR2 thermal dominance
    brightness = (b2 + b3 + b4) / 3.0
    # clouds: bright across visible AND SWIR1>=SWIR2 (no thermal spike)
    cloud = (brightness > cloud_bright) & (b12 <= b11)
    # active fire: SWIR2 exceeds SWIR1 AND NIR (thermal emission), significant, not cloud
    fire = (b12 > b11 * 1.05) & (b12 > nir * 1.05) & (b12 > fire_refl_thr) & (~cloud)
    nbr = (nir - b12) / (nir + b12 + 1e-6)            # burn severity
    fc = np.stack([b12, nir, b4], -1)                  # SWIR false colour
    lo, hi = np.percentile(fc, 2), np.percentile(fc, 98)
    fc = np.clip((fc - lo) / (hi - lo + 1e-6), 0, 1)
    return {"fc": fc, "fire": fire, "cloud": cloud, "nbr": nbr,
            "n_fire": int(fire.sum()), "n_cloud": int(cloud.sum()),
            "burned_pct": 100.0 * float((nbr < -0.1).mean())}


def wildfire_overlay(layers, show_cloud=True):
    import cv2
    ov = (layers["fc"] * 255).astype(np.uint8).copy()
    ov = np.ascontiguousarray(ov)
    if show_cloud:
        ov[layers["cloud"]] = (120, 200, 220)         # clouds -> pale cyan (masked out)
    ov[layers["fire"]] = (255, 40, 40)                # active fire -> red
    return ov
