"""
Precompute preset results (before/after + uncertainty + GeoTIFF) on this machine
(GPU = fast) and write them into a precache/ folder. Ship that folder inside the
Space so those presets return INSTANTLY on the server, even after a cold restart.

Usage:
    python build_precache.py [OUT_DIR]
    (default OUT_DIR = ../satsr-space/precache)

Needs copernicus.env (Sentinel Hub client id/secret) for live ingest.
"""
import os
import sys
import json
import base64
import shutil
import tempfile

import numpy as np
import cv2
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "app"))
sys.path.insert(0, HERE)

import run_web as R                 # reuse the exact display pipeline + warm engine
import satsr_app_utils as U

# must match the front-end PRESETS in web/index.html
PRESETS = {
    "Dubai":          [55.24, 25.18, 55.32, 25.25],
    "Mumbai":         [72.82, 19.02, 72.90, 19.09],
    "Bengaluru":      [77.55, 12.95, 77.63, 13.02],
    "Cairo":          [31.20, 30.02, 31.28, 30.09],
    "Nepal Trishuli": [85.11, 27.88, 85.19, 27.96],
}
SCALE = 4.0
SIZE = 384
DATE_FROM = "2026-01-01"      # must match the front-end default date window
DATE_TO = "2026-06-30"
MAX_CLOUD = 25                # must match the front-end default cloud %

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "satsr-space", "precache")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)


def slug(name):
    return name.lower().replace(" ", "_").replace(",", "")


def build(name, bbox):
    tmp = tempfile.mkdtemp(prefix="precache_")
    inp = os.path.join(tmp, "in.tif"); sr = os.path.join(tmp, "sr.tif"); unc = os.path.join(tmp, "unc.tif")
    meta = U.sh_ingest(bbox, DATE_FROM, DATE_TO, inp, size=SIZE, max_cloud=MAX_CLOUD)
    eng = R._get_engine("rgbn", 4, ["red", "green", "blue", "nir"])
    eng.cfg.target_scale = SCALE; eng.cfg.tta = 4; eng.cfg.uncertainty = True
    res = eng.enhance(inp, sr, unc)

    lr, hr = R._rgb3(inp), R._rgb3(sr)
    Hs, Ws = hr.shape[:2]
    lo, hi = np.percentile(hr, 2), np.percentile(hr, 98)
    st = lambda a: np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    bef = (st(R._square(cv2.resize(lr, (Ws, Hs), interpolation=cv2.INTER_NEAREST),
                        cv2.INTER_NEAREST)) * 255).astype(np.uint8)
    aft = (st(R._square(hr, cv2.INTER_AREA)) * 255).astype(np.uint8)

    with rasterio.open(unc) as ds:
        u = ds.read(1).astype(np.float32)
    lo2, hi2 = np.percentile(u, 1), np.percentile(u, 99)
    us = np.clip((u - lo2) / (hi2 - lo2 + 1e-6), 0, 1)
    u8 = (R._square(us, cv2.INTER_LINEAR) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    _, hb = cv2.imencode(".jpg", heat, [cv2.IMWRITE_JPEG_QUALITY, 88])
    unc_uri = "data:image/jpeg;base64," + base64.b64encode(hb).decode()

    sg = slug(name); tif = sg + ".tif"
    shutil.copy(sr, os.path.join(OUT, tif))
    before_uri, after_uri = R._datauri(bef), R._datauri(aft)

    for unc_on in (False, True):
        payload = {
            "before": before_uri, "after": after_uri,
            "uncertainty": unc_uri if unc_on else None,
            "meta": {"size": list(res["out_size"]), "crs": meta["crs"],
                     "bands": meta.get("bands", 4), "mode": "rgb",
                     "vram": res.get("peak_vram_gb"), "ingest_s": 0.0, "sr_s": 0.0,
                     "download_id": sg, "cached": True},
        }
        entry = {"key": {"bbox": [float(v) for v in bbox], "scale": SCALE,
                         "mode": "rgb", "unc": unc_on, "size": SIZE,
                         "date_from": DATE_FROM, "date_to": DATE_TO, "max_cloud": MAX_CLOUD},
                 "tif": tif, "payload": payload}
        with open(os.path.join(OUT, f"{sg}__unc{int(unc_on)}.json"), "w", encoding="utf-8") as f:
            json.dump(entry, f)
    print(f"  built {name}")


if __name__ == "__main__":
    print(f"writing precache -> {OUT}")
    for nm, bb in PRESETS.items():
        build(nm, bb)
    print("done.")
