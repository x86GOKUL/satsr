"""
SATSR — Gradio front-end (Hugging Face ZeroGPU compatible).

The RRDBNet super-resolution model is built once on CPU at startup; the actual
inference runs inside a @spaces.GPU call so on Hugging Face it grabs a free
ZeroGPU slice for a few seconds, then releases it. Locally, @spaces.GPU is a
no-op and it just uses whatever device is available.
"""
from __future__ import annotations
import os
import re
import sys
import time
import tempfile

import numpy as np
import cv2
import rasterio
import gradio as gr
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "app"))
sys.path.insert(0, HERE)

import satsr_app_utils as U                    # noqa: E402
from satsr import SatSRConfig                   # noqa: E402
from satsr.inference import SuperResolver       # noqa: E402
from satsr.pipeline import resolve_weights      # noqa: E402

# --- ZeroGPU decorator (no-op off Hugging Face) --------------------------------
try:
    import spaces
    GPU = spaces.GPU
except Exception:                               # local / non-HF
    def GPU(duration=None):
        def deco(fn):
            return fn
        return deco

PRESETS = {
    "Dubai, UAE":       [55.24, 25.18, 55.32, 25.25],
    "Mumbai, India":    [72.82, 19.02, 72.90, 19.09],
    "Bengaluru, India": [77.55, 12.95, 77.63, 13.02],
    "Cairo, Egypt":     [31.20, 30.02, 31.28, 30.09],
    "Nepal Trishuli":   [85.11, 27.88, 85.19, 27.96],
}
HAS_SLIDER = hasattr(gr, "ImageSlider")

# --- model, built once on CPU --------------------------------------------------
_cfg = SatSRConfig(mode="rgbn", in_channels=4,
                   band_order=["red", "green", "blue", "nir"],
                   device="cpu", tile=384)
_cfg.weights = resolve_weights(_cfg)
_ENGINE = SuperResolver(_cfg)


# --- helpers -------------------------------------------------------------------
def _rgb3(path):
    with rasterio.open(path) as ds:
        return np.transpose(ds.read(), (1, 2, 0))[..., :3].astype(np.float32) / 10000.0


def _square(x, interp, out=760):
    m = min(x.shape[:2]); y0 = (x.shape[0] - m) // 2; x0 = (x.shape[1] - m) // 2
    return cv2.resize(x[y0:y0 + m, x0:x0 + m], (out, out), interpolation=interp)


def _pair(inp, sr):
    lr, hr = _rgb3(inp), _rgb3(sr)
    Hs, Ws = hr.shape[:2]
    lo, hi = np.percentile(hr, 2), np.percentile(hr, 98)
    st = lambda a: np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    bef = (st(_square(cv2.resize(lr, (Ws, Hs), interpolation=cv2.INTER_NEAREST),
                      cv2.INTER_NEAREST)) * 255).astype(np.uint8)
    aft = (st(_square(hr, cv2.INTER_AREA)) * 255).astype(np.uint8)
    return bef, aft


def _unc_img(path):
    with rasterio.open(path) as ds:
        u = ds.read(1).astype(np.float32)
    lo, hi = np.percentile(u, 1), np.percentile(u, 99)
    us = np.clip((u - lo) / (hi - lo + 1e-6), 0, 1)
    u8 = (_square(us, cv2.INTER_LINEAR) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _bbox(area, coords):
    coords = (coords or "").strip()
    if coords:
        n = [float(x) for x in re.split(r"[,\s]+", coords) if x]
        if len(n) == 2:
            lon, lat = n
            return [lon - 0.04, lat - 0.035, lon + 0.04, lat + 0.035]
        if len(n) == 4:
            return [min(n[0], n[2]), min(n[1], n[3]), max(n[0], n[2]), max(n[1], n[3])]
    return PRESETS.get(area, PRESETS["Dubai, UAE"])


@GPU(duration=120)
def _sr_gpu(inp, sr, unc, scale, tta, want_unc):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _ENGINE.device = dev
    _ENGINE.model.to(dev)
    _ENGINE.cfg.target_scale = scale
    _ENGINE.cfg.tta = tta
    _ENGINE.cfg.uncertainty = want_unc
    return _ENGINE.enhance(inp, sr, unc if want_unc else None)


def run(area, coords, scale, want_unc):
    bbox = _bbox(area, coords)
    scale = float(scale)
    tta = 4 if want_unc else 1
    tmp = tempfile.mkdtemp(prefix="satsr_")
    inp = os.path.join(tmp, "in.tif"); sr = os.path.join(tmp, "sr.tif"); unc = os.path.join(tmp, "unc.tif")

    t0 = time.time()
    meta = U.sh_ingest(bbox, "2026-01-01", "2026-06-30", inp, size=384, max_cloud=25)
    ing = time.time() - t0
    t1 = time.time()
    res = _sr_gpu(inp, sr, unc, scale, tta, want_unc)
    srt = time.time() - t1

    bef, aft = _pair(inp, sr)
    unc_out = _unc_img(unc) if (want_unc and os.path.exists(unc)) else None
    gsd = 10.0 / scale
    info = (f"**{res['out_size'][1]}×{res['out_size'][0]} px** · {gsd:.1f} m ground "
            f"sample ({scale:g}×) · {meta['crs']} · ingest {ing:.1f}s · "
            f"super-resolution {srt:.1f}s" + ("  · uncertainty ✓" if unc_out is not None else ""))
    slider = (bef, aft) if HAS_SLIDER else None
    return (slider if HAS_SLIDER else bef), (None if HAS_SLIDER else aft), unc_out, info, sr


CSS = """
#hdr{background:#0b2545;border-bottom:3px solid #b8860b;border-radius:8px;padding:16px 20px;margin-bottom:8px;}
#hdr h1{color:#fff;font-family:Georgia,serif;margin:0;font-size:1.7rem;letter-spacing:.5px;}
#hdr p{color:#c4d1e6;margin:4px 0 0;font-size:.95rem;}
.gradio-container{max-width:1100px !important;}
footer{visibility:hidden;}
"""

with gr.Blocks(title="SATSR", theme=gr.themes.Soft(primary_hue="amber",
               secondary_hue="blue", neutral_hue="slate"), css=CSS) as demo:
    gr.HTML('<div id="hdr"><h1>SATSR — Satellite Imagery Super-Resolution</h1>'
            '<p>Sentinel-2 10 m → 4 m · georeferenced · uncertainty-aware · '
            'live from Copernicus · free ZeroGPU</p></div>')
    with gr.Row():
        with gr.Column(scale=1):
            area = gr.Dropdown(list(PRESETS), value="Dubai, UAE", label="Area")
            coords = gr.Textbox(label="…or coordinates",
                                placeholder="lon,lat  or  minLon,minLat,maxLon,maxLat")
            scale = gr.Dropdown(["2.5", "4"], value="4", label="Detail (× upscale)")
            want_unc = gr.Checkbox(label="Uncertainty map", value=False)
            btn = gr.Button("Run super-resolution", variant="primary")
            info = gr.Markdown()
            dl = gr.File(label="Download GeoTIFF")
        with gr.Column(scale=2):
            if HAS_SLIDER:
                out_slider = gr.ImageSlider(label="10 m input  ↔  SATSR 4 m", type="numpy")
                out_after = gr.Image(visible=False)
            else:
                out_slider = gr.Image(label="10 m input", type="numpy")
                out_after = gr.Image(label="SATSR 4 m", type="numpy")
            out_unc = gr.Image(label="Uncertainty (inferno) — bright = less certain",
                               type="numpy")

    btn.click(run, [area, coords, scale, want_unc],
              [out_slider, out_after, out_unc, info, dl])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
