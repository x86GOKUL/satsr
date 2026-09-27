"""
SATSR web application server.

Serves the front-end (web/) and exposes the real pipeline over HTTP so the page
can ingest a Sentinel-2 area (preset or drawn on the map) and super-resolve it
live, in true colour or SWIR/wildfire false colour, with a downloadable GeoTIFF.

Run:
    python run_web.py
    -> open http://127.0.0.1:8700
"""
from __future__ import annotations
import os, sys, time, base64, tempfile, uuid
import numpy as np, cv2, rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "app"))
sys.path.insert(0, HERE)

import satsr_app_utils as U                             # noqa: E402
from satsr import SatSRConfig                           # noqa: E402
from satsr.inference import SuperResolver               # noqa: E402
from satsr.pipeline import resolve_weights             # noqa: E402
import torch                                            # noqa: E402

from fastapi import FastAPI, Request                    # noqa: E402
from fastapi.responses import JSONResponse, FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles             # noqa: E402

# use every CPU core for inference (matters a lot on CPU-only hosts)
try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception:
    pass

app = FastAPI(title="SATSR")
DOWNLOADS: dict[str, str] = {}                          # id -> GeoTIFF path
_ENGINES: dict = {}                                     # (mode, in_ch) -> warm SuperResolver
_RESULT_CACHE: dict = {}                                # request key -> JSON payload


def _get_engine(mode: str, in_channels: int, band_order):
    """Build the model once and keep it resident; reused across every request."""
    key = (mode, in_channels)
    eng = _ENGINES.get(key)
    if eng is None:
        base = SatSRConfig(mode=mode, in_channels=in_channels, band_order=band_order,
                           device="auto", tile=384)
        base.weights = resolve_weights(base)
        eng = SuperResolver(base)
        _ENGINES[key] = eng
    return eng


def _ckey(bbox, scale, mode, unc, size, d_from, d_to, cloud):
    """Cache key. Includes date window + cloud %, so a custom date/cloud is a
    fresh fetch while default-window presets still hit the precache."""
    return (tuple(round(float(v), 4) for v in bbox), round(float(scale), 3),
            mode, bool(unc), int(size), str(d_from), str(d_to), int(cloud))


def _load_precache():
    """Load baked-in preset results so they return instantly, even after a cold
    restart on a free host. Built offline by build_precache.py."""
    import json, glob
    d = os.path.join(HERE, "precache")
    if not os.path.isdir(d):
        return
    n = 0
    for jf in glob.glob(os.path.join(d, "*.json")):
        try:
            e = json.load(open(jf, encoding="utf-8"))
            k = e["key"]
            key = _ckey(k["bbox"], k["scale"], k["mode"], k["unc"], k["size"],
                        k["date_from"], k["date_to"], k["max_cloud"])
            tif = os.path.join(d, e["tif"])
            if os.path.exists(tif):
                DOWNLOADS[e["payload"]["meta"]["download_id"]] = tif
            _RESULT_CACHE[key] = e["payload"]
            n += 1
        except Exception:
            pass
    if n:
        print(f"SATSR precache: loaded {n} preset result(s)")


_load_precache()


def _datauri(rgb_uint8: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _rgb3(path: str) -> np.ndarray:
    """First three bands as an RGB float image (works for RGB or SWIR-order stacks)."""
    with rasterio.open(path) as ds:
        a = np.transpose(ds.read(), (1, 2, 0))[..., :3].astype(np.float32) / 10000.0
    return a


def _square(x: np.ndarray, interp, out: int = 760) -> np.ndarray:
    m = min(x.shape[:2]); y0 = (x.shape[0]-m)//2; x0 = (x.shape[1]-m)//2
    return cv2.resize(x[y0:y0+m, x0:x0+m], (out, out), interpolation=interp)


@app.post("/api/run")
async def run(req: Request):
    body = await req.json()
    bbox = body.get("bbox")
    if not bbox or len(bbox) != 4:
        return JSONResponse({"error": "draw or pick an area first"}, status_code=400)
    bbox = [float(v) for v in bbox]
    scale = float(body.get("scale", 4.0))
    size = int(body.get("size", 384))
    mode = body.get("mode", "rgb")                      # "rgb" | "wildfire"
    d_from = body.get("date_from", "2026-01-01")
    d_to = body.get("date_to", "2026-06-30")
    max_cloud = int(body.get("max_cloud", 25))
    want_unc = bool(body.get("uncertainty", False))
    tta = 4 if want_unc else 1

    # Result cache: an identical area/scale/mode/date/cloud (e.g. a preset the jury
    # clicks) returns instantly; a custom date or cloud % recomputes live.
    ckey = _ckey(bbox, scale, mode, want_unc, size, d_from, d_to, max_cloud)
    cached = _RESULT_CACHE.get(ckey)
    if cached and os.path.exists(DOWNLOADS.get(cached["meta"]["download_id"], "")):
        out = dict(cached); out["meta"] = dict(cached["meta"]); out["meta"]["cached"] = True
        return JSONResponse(out)

    tmp = tempfile.mkdtemp(prefix="satsr_web_")
    inp = os.path.join(tmp, "in.tif"); sr = os.path.join(tmp, "sr.tif"); unc = os.path.join(tmp, "unc.tif")
    try:
        t0 = time.time()
        if mode == "wildfire":
            # SWIR false colour: B12 (SWIR2), B08 (NIR), B04 (Red)
            meta = U.sh_ingest_bands(bbox, d_from, d_to, inp,
                                     bands=["B12", "B08", "B04"], size=size,
                                     max_cloud=max_cloud)
            eng = _get_engine("per_band", 3, ["swir2", "nir", "red"])
        else:
            meta = U.sh_ingest(bbox, d_from, d_to, inp, size=size,
                               max_cloud=max_cloud)
            eng = _get_engine("rgbn", 4, ["red", "green", "blue", "nir"])
        ing = round(time.time() - t0, 1)
        t1 = time.time()
        eng.cfg.target_scale = scale
        eng.cfg.tta = tta
        eng.cfg.uncertainty = want_unc
        res = eng.enhance(inp, sr, unc if want_unc else None)
        srt = round(time.time() - t1, 1)

        lr, hr = _rgb3(inp), _rgb3(sr)
        Hs, Ws = hr.shape[:2]
        lo, hi = np.percentile(hr, 2), np.percentile(hr, 98)
        st = lambda a: np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
        bef = (st(_square(cv2.resize(lr, (Ws, Hs), interpolation=cv2.INTER_NEAREST),
                          cv2.INTER_NEAREST)) * 255).astype(np.uint8)
        aft = (st(_square(hr, cv2.INTER_AREA)) * 255).astype(np.uint8)

        unc_uri = None
        if want_unc and os.path.exists(unc):
            with rasterio.open(unc) as ds:
                u = ds.read(1).astype(np.float32)
            lo2, hi2 = np.percentile(u, 1), np.percentile(u, 99)
            us = np.clip((u - lo2) / (hi2 - lo2 + 1e-6), 0, 1)
            u8 = (_square(us, cv2.INTER_LINEAR) * 255).astype(np.uint8)
            heat = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)   # already BGR
            ok, buf = cv2.imencode(".jpg", heat, [cv2.IMWRITE_JPEG_QUALITY, 88])
            unc_uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        did = uuid.uuid4().hex
        DOWNLOADS[did] = sr
        payload = {
            "before": _datauri(bef), "after": _datauri(aft), "uncertainty": unc_uri,
            "meta": {"size": res["out_size"], "crs": meta["crs"],
                     "bands": meta.get("bands", 4), "mode": mode,
                     "vram": res.get("peak_vram_gb"), "ingest_s": ing, "sr_s": srt,
                     "download_id": did},
        }
        _RESULT_CACHE[ckey] = payload
        return JSONResponse(payload)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.get("/api/download/{did}")
def download(did: str):
    p = DOWNLOADS.get(did)
    if not p or not os.path.exists(p):
        return JSONResponse({"error": "file expired — run again"}, status_code=404)
    return FileResponse(p, media_type="image/tiff", filename="satsr_superresolved.tif")


# static front-end (mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=os.path.join(HERE, "web"), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8700"))
    print(f"SATSR web app -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
