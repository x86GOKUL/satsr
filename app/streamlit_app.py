"""
SATSR Lab — interactive web application on top of the SATSR super-resolution
framework.

Run:
    streamlit run app/streamlit_app.py      (from the SATSR project root)

Three capabilities:
  1. Web GIS      — before/after split-slider, layer inspection, map overlay, GeoTIFF download
  2. Downstream   — live NDVI/NDWI zonal stats + building/structure extraction
  3. Live ingest  — STAC (AWS Sentinel-2 L2A) bbox ingestion, then super-resolve
"""
from __future__ import annotations

import os
import sys
import io as _io
import tempfile

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import satsr_app_utils as U
import downstream_suite as D
from satsr import SatSRConfig
from satsr.pipeline import enhance_scene

st.set_page_config(page_title="SATSR — Satellite Imagery Super-Resolution Platform",
                   page_icon="🛰️", layout="wide")

# ----------------------------- institutional styling ----------------------
st.markdown("""
<style>
  :root{
    --ink:#16233a; --navy:#0b2545; --steel:#13315c; --line:#c9d2e0;
    --band:#0b2545; --accent:#b8860b; --accent2:#1c6b3c; --paper:#ffffff; --bg:#eef1f6;
  }
  .stApp{background:var(--bg);}
  .block-container{padding-top:0 !important; max-width:1240px;}
  header[data-testid="stHeader"]{background:transparent;}
  h1,h2,h3,h4{color:var(--navy); font-weight:700; letter-spacing:.1px;}
  hr{border-color:var(--line);}

  /* ---- top utility strip ---- */
  .gov-strip{background:var(--navy); color:#dbe3f0; font-size:.74rem; letter-spacing:.4px;
    padding:5px 18px; display:flex; justify-content:space-between; align-items:center;
    margin:0 -100vw; padding-left:calc(50vw - 620px); padding-right:calc(50vw - 620px);}
  .gov-strip span{opacity:.9;}

  /* ---- masthead ---- */
  .gov-mast{background:var(--paper); border-bottom:3px solid var(--accent);
    display:flex; align-items:center; gap:16px; padding:14px 4px 16px;}
  .gov-emblem{flex:0 0 auto;}
  .gov-id{display:flex; flex-direction:column; line-height:1.15;}
  .gov-over{font-size:.72rem; letter-spacing:2.2px; color:#5c6b82; text-transform:uppercase; font-weight:600;}
  .gov-name{font-family:Georgia,"Times New Roman",serif; font-size:2.05rem; font-weight:700;
    color:var(--navy); letter-spacing:.5px; margin:1px 0;}
  .gov-sub{font-size:.9rem; color:#3d4a60;}
  .gov-right{margin-left:auto; text-align:right; font-size:.78rem; color:#5c6b82;}
  .gov-right b{color:var(--navy);}
  .gov-rule{height:4px; background:linear-gradient(90deg,var(--accent) 0 33%,#e7e9ee 33% 66%,var(--accent2) 66% 100%);
    margin:0 0 6px;}

  /* ---- tabs as a formal nav bar ---- */
  .stTabs [data-baseweb="tab-list"]{background:var(--band); gap:0; border-radius:2px;
    padding:0; border:1px solid var(--navy);}
  .stTabs [data-baseweb="tab"]{height:44px; padding:0 22px; color:#c7d2e6 !important;
    font-weight:600; font-size:.92rem; letter-spacing:.3px; border-radius:0;
    border-right:1px solid #1c3357;}
  .stTabs [aria-selected="true"]{background:#12325c; color:#ffffff !important;
    box-shadow:inset 0 -3px 0 var(--accent);}
  .stTabs [data-baseweb="tab-highlight"]{background:transparent;}

  /* ---- buttons ---- */
  .stButton>button{background:var(--navy); color:#fff; border:1px solid var(--navy);
    border-radius:3px; font-weight:600; letter-spacing:.2px;}
  .stButton>button:hover{background:var(--steel); border-color:var(--steel); color:#fff;}
  .stDownloadButton>button{background:#fff; color:var(--navy); border:1px solid var(--navy);
    border-radius:3px; font-weight:600;}

  /* ---- cards / chips ---- */
  .metric-card{background:#fff;border:1px solid var(--line);border-left:4px solid var(--navy);
    border-radius:3px;padding:.7rem 1rem;}
  .tag{display:inline-block;background:#eef2f8;border:1px solid var(--line);color:var(--steel);
       border-radius:3px;padding:.15rem .55rem;font-size:.78rem;font-weight:600;margin-right:.3rem;}
  .good{color:var(--accent2);font-weight:700;}

  /* ---- sidebar ---- */
  section[data-testid="stSidebar"]{background:#f7f9fc; border-right:1px solid var(--line);}
  section[data-testid="stSidebar"] h2{font-size:1rem; text-transform:uppercase;
    letter-spacing:.6px; border-bottom:2px solid var(--accent); padding-bottom:4px;}

  /* ---- footer ---- */
  .gov-foot{background:var(--navy); color:#c7d2e6; font-size:.8rem; line-height:1.5;
    padding:16px 20px; margin:26px -100vw 0; padding-left:calc(50vw - 620px);
    padding-right:calc(50vw - 620px); border-top:3px solid var(--accent);}
  .gov-foot b{color:#fff;}
  .gov-foot .muted{color:#8ea1bd; font-size:.74rem;}
  footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

_EMBLEM = """<svg width="58" height="58" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="32" cy="32" r="30" fill="#0b2545"/><circle cx="32" cy="32" r="30" stroke="#b8860b" stroke-width="2"/>
<ellipse cx="32" cy="32" rx="12" ry="26" stroke="#7f93b5" stroke-width="1.4" fill="none"/>
<circle cx="32" cy="32" r="12" stroke="#7f93b5" stroke-width="1.4" fill="none"/>
<path d="M16 32h32M32 16v32" stroke="#7f93b5" stroke-width="1" opacity="0.5"/>
<circle cx="32" cy="32" r="6.5" fill="#1c6b3c"/>
<rect x="45" y="12" width="7" height="7" rx="1" fill="#b8860b" transform="rotate(45 48.5 15.5)"/>
</svg>"""

st.markdown(f"""
<div class="gov-strip"><span>SMART INDIA HACKATHON 2026 &nbsp;·&nbsp; GEOSPATIAL / SATELLITE IMAGERY</span>
<span>Prototype for evaluation</span></div>
<div class="gov-mast">
  <div class="gov-emblem">{_EMBLEM}</div>
  <div class="gov-id">
    <div class="gov-over">Satellite Imagery Super-Resolution &amp; Geospatial Intelligence Platform</div>
    <div class="gov-name">SATSR</div>
    <div class="gov-sub">Sentinel-2 10&nbsp;m &rarr; 4&nbsp;m · georeferenced · multispectral · uncertainty-aware</div>
  </div>
  <div class="gov-right"><b>Version 1.0.0</b><br>Data source: Copernicus Sentinel-2<br>Status: Operational (demo)</div>
</div>
<div class="gov-rule"></div>
""", unsafe_allow_html=True)

SCENES = U.available_scenes()


# ---------------------- session helpers ----------------------------------
def get_scene():
    """Return the currently selected scene dict (or an ingested one)."""
    if st.session_state.get("ingested"):
        return st.session_state["ingested"]
    if not SCENES:
        return None
    label = st.sidebar.selectbox("Scene", list(SCENES.keys()))
    return {"label": label, **SCENES[label]}


@st.cache_data(show_spinner=False)
def _detect(path, which):
    """Cached tiled YOLO building detection for a scene layer (path is the cache key)."""
    refl = U.read_reflectance(path)
    return U.extract_structures(U.to_rgb(refl))


@st.cache_data(show_spinner=False)
def _buildings(path):
    """Cached building footprint POLYGONS + stats + GeoJSON for a scene."""
    refl = U.read_reflectance(path)
    rgb = U.to_rgb(refl)
    polys = D.extract_building_polys(rgb)
    overlay = D.draw_polys(rgb, polys)
    stats = D.building_stats(polys, path, shape=rgb.shape[:2])
    gj = D.polys_to_geojson(polys, path) if polys else '{"type":"FeatureCollection","features":[]}'
    return overlay, stats, gj


@st.cache_data(show_spinner=False)
def _water(path):
    refl = U.read_reflectance(path)
    mask, nd, thr = D.water_layer(refl)
    ov, _ = D.water_overlay(U.to_rgb(refl), mask)
    area_m2, proj = D.water_area_m2(mask, path)
    cont = D.water_continuity(mask)
    return ov, area_m2, proj, cont, thr


@st.cache_data(show_spinner=False)
def _roads(path):
    rgb = U.to_rgb(U.read_reflectance(path))
    skel, _ = D.road_layer(rgb)
    km, proj = D.road_length_km(skel, path)
    return D.road_overlay(rgb, skel), km, proj


@st.cache_data(show_spinner=False)
def _fields(path):
    refl = U.read_reflectance(path)
    ov, mag = D.field_edge_overlay(refl)
    return ov, mag


@st.cache_data(show_spinner=False)
def _superres(input_path, scale, tta):
    """Run SATSR at an arbitrary scale on the fly (cached). scale 4.0 = 10 m→2.5 m."""
    import tempfile
    d = tempfile.mkdtemp(prefix="satsr_res_")
    out = os.path.join(d, f"sr_x{scale}.tif")
    unc = os.path.join(d, f"sr_x{scale}_unc.tif")
    cfg = SatSRConfig(in_channels=4, target_scale=float(scale),
                      band_order=["red", "green", "blue", "nir"], tta=tta, device="auto")
    res = enhance_scene(input_path, out, cfg, unc)
    return out, (unc if os.path.exists(unc) else None), res.get("out_size"), res.get("peak_vram_gb")


@st.cache_data(show_spinner=False)
def _builtup(path):
    refl = U.read_reflectance(path)
    score, mask, has_swir, cov = D.builtup_index(refl)
    ov = D.builtup_overlay(U.to_rgb(refl), score, mask)
    return ov, has_swir, cov


@st.cache_data(show_spinner=False)
def _wildfire(path):
    refl = U.read_reflectance(path)
    L = D.wildfire_layers(refl)
    if L is None:
        return None
    return D.wildfire_overlay(L), U.colorize(np.clip((L["nbr"]+1)/2, 0, 1), "RdYlGn"), \
        L["n_fire"], L["n_cloud"], L["burned_pct"]


@st.cache_data(show_spinner=False)
def _osm(path):
    refl = U.read_reflectance(path)
    rgb = U.to_rgb(refl)
    polys, n = D.osm_footprints_for_scene(path)
    ov = D.draw_polys(rgb, polys, color=(56, 189, 248), alpha=0.4)
    gj = D.polys_to_geojson(polys, path) if polys else '{"type":"FeatureCollection","features":[]}'
    return ov, n, gj


@st.cache_data(show_spinner=False)
def load_layers(inp, sr, unc):
    lr = U.read_reflectance(inp)
    hr = U.read_reflectance(sr)
    layers = {
        "lr_rgb": U.to_rgb(lr),
        "sr_rgb": U.to_rgb(hr),
        "lr_nir": U.to_nir_false_color(lr),
        "sr_nir": U.to_nir_false_color(hr),
        "unc": U.uncertainty_rgb(unc) if unc else None,
    }
    return layers, lr, hr


# --------------------------- sidebar -------------------------------------
st.sidebar.header("Controls")
if not SCENES and not st.session_state.get("ingested"):
    st.sidebar.warning("No demo scenes found on disk. Use the **Live Ingest** tab to pull one.")

scene = get_scene()

# ---- output-resolution control (applies to every tab) -------------------
res_note = "4 m (2.5×) — ready"
if scene:
    st.sidebar.markdown("**Output resolution**")
    st.session_state.setdefault("res_scale", {})
    key = scene["input"]
    # (label, target_scale) — 10 m / scale = ground sample distance
    res_opts = [("4 m · 2.5×", 2.5), ("2.5 m · 4×", 4.0), ("1 m · 10×", 10.0)]
    cols = st.sidebar.columns(len(res_opts))
    for c, (lbl, sc) in zip(cols, res_opts):
        if c.button(lbl):
            st.session_state["res_scale"][key] = sc
    scale = st.session_state["res_scale"].get(key, 2.5)
    if scale != 2.5:
        gsd = 10.0 / scale
        tta = 4 if scale <= 4.0 else 1          # extreme scales: single pass (huge output)
        with st.spinner(f"Super-resolving 10 m → {gsd:g} m ({scale:g}×) on GPU…"):
            sr, unc, size, vram = _superres(scene["input"], scale, tta)
        scene = {**scene, "sr": sr, "unc": unc}
        res_note = f"{gsd:g} m ({scale:g}×) — live · {size[0]}×{size[1]} px · {vram:.1f} GB"
        st.sidebar.success(f"Showing **{gsd:g} m** ({scale:g}×) product")
        if scale >= 10.0:
            st.sidebar.warning("≥10× is beyond the model's 4× native — extra detail is "
                               "**interpolated/speculative**, not recovered. Uncertainty saturates.")
    else:
        st.sidebar.caption("Showing 4 m (2.5×) · click a finer scale to render live")

tab_gis, tab_down, tab_ingest = st.tabs(
    ["  GIS  VIEWER  ", "  ANALYSIS  &  EXTRACTION  ", "  DATA  ACQUISITION  "])


# =========================================================================
# TAB 1 — WEB GIS
# =========================================================================
with tab_gis:
    if not scene:
        st.info("Select a scene in the sidebar, or ingest one in the Live Ingest tab.")
    else:
        layers, lr, hr = load_layers(scene["input"], scene["sr"], scene.get("unc"))
        st.subheader(f"Before → After · {scene.get('label','ingested scene')}")

        c1, c2 = st.columns([3, 1])
        with c2:
            layer = st.radio("Layer", ["True colour", "NIR false-colour", "Uncertainty"])
            st.markdown(f"<span class='tag'>10 m in</span><span class='tag'>{res_note}</span>",
                        unsafe_allow_html=True)
            geo = U.scene_geo(scene["sr"])
            if geo:
                st.markdown("**Geo-integrity**")
                st.write(f"CRS `{geo.crs}`")
                st.write(f"pixel ≈ {geo.res_m:.2f} m")
                st.caption("CRS preserved · grid rescaled · overlays in QGIS")

        with c1:
            if layer == "Uncertainty":
                if layers["unc"] is not None:
                    st.image(layers["unc"], caption="Per-pixel uncertainty (brighter = less certain)",
                             use_container_width=True)
                else:
                    st.warning("No uncertainty layer for this scene.")
            else:
                a, b = ("lr_rgb", "sr_rgb") if layer == "True colour" else ("lr_nir", "sr_nir")
                try:
                    from streamlit_image_comparison import image_comparison
                    image_comparison(layers[a], layers[b],
                                     label1="10 m input", label2="SATSR <4 m",
                                     width=900, in_memory=True)
                except Exception:
                    cc1, cc2 = st.columns(2)
                    cc1.image(layers[a], caption="10 m input", use_container_width=True)
                    cc2.image(layers[b], caption="SATSR <4 m", use_container_width=True)
            st.caption("Drag the handle to compare input vs super-resolved output.")

        # map overlay
        geo = U.scene_geo(scene["sr"])
        if geo:
            with st.expander("🗺️  Map context (georeferenced overlay)", expanded=False):
                try:
                    import folium
                    from streamlit_folium import st_folium
                    from PIL import Image
                    m = folium.Map(location=list(geo.center), zoom_start=13, tiles="CartoDB positron")
                    folium.raster_layers.ImageOverlay(
                        image=layers["sr_rgb"],
                        bounds=[[geo.south, geo.west], [geo.north, geo.east]],
                        opacity=0.9, name="SATSR <4 m",
                    ).add_to(m)
                    folium.LayerControl().add_to(m)
                    st_folium(m, height=420, use_container_width=True)
                except Exception as e:
                    st.caption(f"Map unavailable: {e}")

        # download
        with open(scene["sr"], "rb") as f:
            st.download_button("⬇️  Download super-resolved GeoTIFF",
                               f, file_name=os.path.basename(scene["sr"]),
                               mime="image/tiff")


# =========================================================================
# TAB 2 — DOWNSTREAM ACTION
# =========================================================================
with tab_down:
    if not scene:
        st.info("Select or ingest a scene first.")
    else:
        _, lr, hr = load_layers(scene["input"], scene["sr"], scene.get("unc"))
        st.subheader("Downstream suite — every target in the problem statement")

        mode = st.selectbox("Analysis mode", [
            "🏢 Real building footprints (OSM)",
            "🏗️ Built-up area (NDBI / spectral)",
            "🛣️ Road / linear networks",
            "🌊 Water & coastlines",
            "🌾 Agri field boundaries",
            "🔥 Wildfire (SWIR)",
            "🏠 Urban footprints — YOLO (experimental)",
        ])

        # ---------------- REAL OSM FOOTPRINTS -----------------------------
        if mode.startswith("🏢"):
            st.markdown("**Real building footprints** from OpenStreetMap, draped on the SR product. "
                        "<span class='tag'>vector source: OSM (free, no auth)</span>",
                        unsafe_allow_html=True)
            if st.button("Fetch real footprints for this scene", type="primary"):
                with st.spinner("Querying OpenStreetMap (Overpass)…"):
                    try:
                        ov, n, gj = _osm(scene["sr"])
                        st.image(ov, caption=f"SATSR <4 m + {n:,} real OSM building footprints",
                                 use_container_width=True)
                        st.markdown(f"<div class='metric-card'><b>{n:,}</b> real building footprints "
                                    f"(surveyed vectors) aligned to the super-resolved scene.</div>",
                                    unsafe_allow_html=True)
                        st.download_button("⬇️ Download footprints (.geojson)", gj,
                                           file_name="osm_buildings.geojson", mime="application/geo+json")
                        st.caption("These are authoritative surveyed footprints — the honest way to show "
                                   "the downstream urban-mapping use. SATSR sharpens the imagery under them; "
                                   "the vectors give exact rooftops for area/density/QGIS.")
                    except Exception as ex:
                        st.error(f"OSM fetch failed (needs internet): {ex}")
            else:
                st.info("Click **Fetch** to pull real footprints (needs internet).")

        # ---------------- BUILT-UP AREA (NDBI) ----------------------------
        elif mode.startswith("🏗️"):
            ov, has_swir, cov = _builtup(scene["sr"])
            src = "real NDBI (SWIR B11)" if has_swir else "no-SWIR proxy (bright · low-NDVI · low-NDWI)"
            st.markdown(f"Spectral **built-up area** — <span class='tag'>{src}</span>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns([2, 1])
            c1.image(ov, caption="Built-up likelihood on SATSR <4 m", use_container_width=True)
            c2.metric("Built-up coverage", f"{cov:.1f}% of AOI")
            if not has_swir:
                c2.caption("This scene is 4-band (no SWIR) → proxy index. Feed a 6-band SWIR "
                           "stack (per_band mode) for true NDBI = (B11−B8)/(B11+B8).")
            st.caption("NDBI is the built-up analogue of NDWI/NDVI — spectral, no training. It maps "
                       "settlement *area* (what 10 m supports), not individual footprints.")

        # ---------------- YOLO FOOTPRINTS (EXPERIMENTAL) ------------------
        elif mode.startswith("🏠"):
            _detector = U.model_label() if U.yolo_available() else "classical CV (fallback)"
            st.warning("⚠️ Experimental: the YOLO detector was trained on sub-metre imagery and "
                       "over-triggers on field parcels at Sentinel resolution — counts are unreliable. "
                       "Use **Real footprints (OSM)** or **Built-up area** for trustworthy outputs.")
            st.markdown("Vector **footprint polygons** (segmentation masks) + GIS stats. "
                        f"<span class='tag'>detector: {_detector}</span>", unsafe_allow_html=True)
            if st.button("Run building footprint extraction (10 m vs <4 m)", type="primary"):
                with st.spinner("Segmenting building footprints (tiled YOLO)…"):
                    ov_lr, st_lr, _ = _buildings(scene["input"])
                    ov_hr, st_hr, gj = _buildings(scene["sr"])
                if st_hr.get("ood"):
                    # OOD scene: NO vector mapping — suppress polygons, count, GeoJSON
                    st.error("⚠️ non-urban/OOD scene — detector unreliable here, trust the uncertainty layer")
                    if scene.get("unc"):
                        u = U.uncertainty_rgb(scene["unc"])
                        cc1, cc2 = st.columns(2)
                        cc1.image(U.to_rgb(hr), caption="SATSR <4 m (no footprints drawn)", use_container_width=True)
                        if u is not None:
                            cc2.image(u, caption="Uncertainty — saturates on unknown texture", use_container_width=True)
                    else:
                        st.image(U.to_rgb(hr), caption="SATSR <4 m (no footprints drawn)", use_container_width=True)
                    st.caption(f"Union footprint density {st_hr['density_pct']:.0f}% is implausibly high — "
                               f"the urban detector is firing on terrain texture (bare desert/rock), not "
                               f"buildings. No vectors or GeoJSON are emitted for OOD scenes.")
                else:
                    c1, c2 = st.columns(2)
                    c1.image(ov_lr, caption=f"10 m input — {st_lr['count']} footprints", use_container_width=True)
                    c2.image(ov_hr, caption=f"SATSR <4 m — {st_hr['count']} footprints", use_container_width=True)
                    gain = (st_hr["count"] / st_lr["count"]) if st_lr["count"] else 0
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Footprints", st_hr["count"], f"{gain:.2f}× vs 10 m")
                    if st_hr.get("projected"):
                        m2.metric("Rooftop area", f"{st_hr['footprint_m2']/1e4:.2f} ha")
                        m3.metric("Built-up density", f"{st_hr['density_pct']:.1f}%")
                        m4.metric("Median building", f"{st_hr.get('median_m2',0):.0f} m²")
                    else:
                        m2.metric("Footprint (px²)", f"{st_hr['area_px']:.0f}")
                    st.download_button("⬇️ Download building footprints (.geojson)", gj,
                                       file_name="satsr_buildings.geojson", mime="application/geo+json")
                    st.caption("Union-based density (footprint area ÷ AOI, ≤100%). Polygons carry real "
                               "lat/lon (EPSG:4326) — drag the GeoJSON straight into QGIS/ArcGIS.")
            else:
                st.info("Click **Run** to segment footprints (tiled CPU inference ~10–20 s).")

        # ---------------- 2. ROADS / LINEAR -------------------------------
        elif mode.startswith("🛣️"):
            st.markdown("Linear-feature centrelines via **Frangi ridge filter + skeleton** — "
                        "narrow roads/canals that vanish into mixed pixels at 10 m.")
            if st.button("Run linear-feature extraction (10 m vs <4 m)", type="primary"):
                with st.spinner("Extracting linear features…"):
                    ov_lr, km_lr, _ = _roads(scene["input"])
                    ov_hr, km_hr, proj = _roads(scene["sr"])
                c1, c2 = st.columns(2)
                c1.image(ov_lr, caption=f"10 m input — {km_lr:.1f} km", use_container_width=True)
                c2.image(ov_hr, caption=f"SATSR <4 m — {km_hr:.1f} km", use_container_width=True)
                gain = (km_hr / km_lr) if km_lr else 0
                st.markdown(f"<div class='metric-card'>Connected linear length: "
                            f"<b>{km_lr:.1f} km → {km_hr:.1f} km</b> "
                            f"(<span class='good'>{gain:.2f}× more</span> recovered on the <4 m product)"
                            f"</div>", unsafe_allow_html=True)
                st.caption("SR recovers continuous road/canal tracks for connectivity & flood-routing. "
                           "Ridge filter also picks up strong field edges — it's a linear-feature map, "
                           "not a road-only classifier.")
            else:
                st.info("Click **Run** to extract linear features.")

        # ---------------- 3. WATER & COASTLINES ---------------------------
        elif mode.startswith("🌊"):
            st.markdown("Sub-pixel **NDWI + Otsu** water mask with a crisp shoreline contour "
                        "on the <4 m product.")
            ov, area_m2, proj, cont, thr = _water(scene["sr"])
            c1, c2 = st.columns([2, 1])
            c1.image(ov, caption="SATSR <4 m — NDWI water + shoreline", use_container_width=True)
            with c2:
                if proj:
                    st.metric("Water area", f"{area_m2/1e6:.3f} km²")
                st.metric("Largest water body", f"{cont:.1f}% of water px")
                st.metric("Otsu NDWI threshold", f"{thr:.3f}")
            st.caption("NDWI = (Green − NIR)/(Green + NIR); Otsu picks the water/land split "
                       "automatically. The <4 m grid yields a continuous shoreline polygon with a "
                       "clean total surface area — no deep learning needed.")

        # ---------------- WILDFIRE (SWIR) ---------------------------------
        elif mode.startswith("🔥"):
            st.markdown("Active-fire + burn severity from **SWIR (B11/B12)**, with a "
                        "cloud mask. <span class='tag'>needs a 6-band SWIR scene</span>",
                        unsafe_allow_html=True)
            res_lr = _wildfire(scene["input"]); res_hr = _wildfire(scene["sr"])
            if res_hr is None:
                st.warning("This scene has no SWIR bands. Select the **🔥 Wildfire** scene "
                           "(or ingest a 6-band stack) — fire detection needs B11/B12.")
            else:
                ov_lr, _, nf_lr, nc_lr, _ = res_lr
                ov_hr, nbr_hr, nf_hr, nc_hr, burned = res_hr
                c1, c2 = st.columns(2)
                c1.image(ov_lr, caption=f"10 m · SWIR + fire (red), cloud masked (cyan) — {nf_lr} fire px",
                         use_container_width=True)
                c2.image(ov_hr, caption=f"SATSR 4 m · fire (red), cloud masked — {nf_hr} fire px",
                         use_container_width=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Active-fire pixels", nf_hr, f"{nf_hr-nf_lr:+d} vs 10 m")
                m2.metric("Cloud px (masked)", f"{nc_hr:,}")
                m3.metric("Burn severity (NBR<-0.1)", f"{burned:.1f}%")
                st.image(nbr_hr, caption="SATSR 4 m · NBR burn severity (green=healthy, red=burned)",
                         use_container_width=True)
                st.caption("Fire = SWIR2 (B12) exceeding SWIR1 & NIR (thermal emission), significant, "
                           "and NOT cloud (clouds are bright but SWIR1≥SWIR2). SWIR sees fire through "
                           "smoke; SR resolves the fire front finer. Add a scene-classification mask "
                           "for operational use.")

        # ---------------- 4. AGRI FIELD BOUNDARIES ------------------------
        else:
            st.markdown("**NDVI Sobel gradient** — field-boundary sharpness.")
            ov_lr, mag_lr = _fields(scene["input"])
            ov_hr, mag_hr = _fields(scene["sr"])
            c1, c2 = st.columns(2)
            c1.image(ov_lr, caption="10 m input — NDVI gradient", use_container_width=True)
            c2.image(ov_hr, caption="SATSR <4 m — NDVI gradient", use_container_width=True)
            st.markdown("<div class='metric-card'>Validated on the held-out stress battery: "
                        "SATSR recovers <b class='good'>91% of true field-boundary gradient</b> "
                        "vs bicubic's 70% (edge error cut 58%).</div>", unsafe_allow_html=True)
            st.caption("Field boundaries are the NDVI spatial gradient; the validated 91% figure comes "
                       "from the degrade→reconstruct protocol in the stress battery (stress_downstream.py).")


# =========================================================================
# TAB 3 — STAC LIVE INGEST
# =========================================================================
with tab_ingest:
    st.subheader("One-click Sentinel-2 ingest → super-resolve")
    source_label = st.radio("Data source", list(U.SEARCH_SOURCES.keys()), horizontal=True)
    source = U.SEARCH_SOURCES[source_label]
    if source == "cdse":
        st.caption("Copernicus Data Space Ecosystem via **Sentinel Hub** (OAuth client in "
                   "copernicus.env) — server-side crop, fast, no S3 keys needed.")
    else:
        st.caption("Public AWS Sentinel-2 L2A via Element84 Earth Search — no credentials needed. "
                   "Draw/enter a small bounding box, pick a low-cloud scene, ingest, and enhance.")

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("**Area of interest**")
        preset = st.selectbox("Preset", ["Mumbai", "Bengaluru", "Cairo", "Dubai",
                                         "Dharavi (Mumbai)", "📍 Custom point", "▢ Custom bbox"])
        presets = {
            "Mumbai": (72.82, 19.02, 72.90, 19.09),
            "Bengaluru": (77.55, 12.95, 77.63, 13.02),
            "Cairo": (31.20, 30.02, 31.28, 30.09),
            "Dubai": (55.24, 25.18, 55.32, 25.25),
            "Dharavi (Mumbai)": (72.83, 19.03, 72.87, 19.06),
        }
        if preset in presets:
            w, s, e, n = presets[preset]
        elif preset == "📍 Custom point":
            st.caption("Paste a point as **decimal** `27.38058, 33.63184` or **DMS** "
                       "`27°22'50.10\"N, 33°37'54.62\"E` — then pick a box size.")
            ptxt = st.text_input("Center point (lat, lon)", "27°22'50.10\"N, 33°37'54.62\"E")
            size_km = st.slider("Box size (km)", 1, 15, 6)
            pt = U.parse_latlon(ptxt)
            if pt is None:
                st.warning("Couldn't parse that — use `lat, lon` (decimal or DMS).")
                w, s, e, n = presets["Mumbai"]
            else:
                w, s, e, n = U.point_bbox(pt[0], pt[1], size_km)
                st.caption(f"→ center {pt[0]:.5f}, {pt[1]:.5f} · bbox "
                           f"[{w:.4f}, {s:.4f}, {e:.4f}, {n:.4f}]")
        else:  # Custom bbox
            w = st.number_input("West", value=72.82, format="%.4f")
            s = st.number_input("South", value=19.02, format="%.4f")
            e = st.number_input("East", value=72.90, format="%.4f")
            n = st.number_input("North", value=19.09, format="%.4f")
        bbox = [float(w), float(s), float(e), float(n)]
        date_range = st.text_input("Date range", "2023-01-01/2024-12-31")
        max_cloud = st.slider("Max cloud cover %", 0, 60, 10)

    with colB:
        try:
            import folium
            from streamlit_folium import st_folium
            cy, cx = (s + n) / 2, (w + e) / 2
            m = folium.Map(location=[cy, cx], zoom_start=11, tiles="CartoDB positron")
            folium.Rectangle([[s, w], [n, e]], color="#34d399", fill=True, fill_opacity=0.15).add_to(m)
            st_folium(m, height=300, use_container_width=True)
        except Exception as ex:
            st.caption(f"(map preview unavailable: {ex})")

    if st.button("🔎 Search Sentinel-2 scenes", type="primary"):
        try:
            with st.spinner(f"Querying STAC ({source_label})…"):
                rows = U.search(source, bbox, date_range, max_cloud)
            if not rows:
                st.warning("No scenes found for that AOI / date / cloud filter.")
            else:
                st.session_state["stac_rows"] = rows
                st.success(f"Found {len(rows)} scenes.")
        except Exception as ex:
            st.error(f"STAC search failed (needs internet): {ex}")

    rows = st.session_state.get("stac_rows")
    if rows:
        labels = [f"{r['date']}  ·  cloud {r['cloud']:.1f}%  ·  {r['id'][:24]}" for r in rows]
        pick = st.selectbox("Choose a scene (lowest cloud first)", range(len(rows)),
                            format_func=lambda i: labels[i])
        cols = st.columns(2)
        do_ingest = cols[0].button("⬇️  Ingest bbox window (10 m, R,G,B,NIR)")
        do_enhance = cols[1].checkbox("Super-resolve after ingest", value=True)
        if do_ingest:
            try:
                tmp = tempfile.mkdtemp(prefix="satsr_stac_")
                inp = os.path.join(tmp, "ingest_10m.tif")
                with st.spinner(f"Reading band window ({source_label})…"):
                    meta = U.ingest(rows[pick], bbox, inp)
                st.success(f"Ingested {meta['size']} px · CRS {meta['crs']}")
                out = {"label": f"STAC {rows[pick]['date']}", "input": inp, "sr": inp, "unc": None}
                if do_enhance:
                    sr = os.path.join(tmp, "ingest_sr.tif")
                    unc = os.path.join(tmp, "ingest_sr_uncertainty.tif")
                    with st.spinner("Running SATSR super-resolution…"):
                        cfg = SatSRConfig(in_channels=4, target_scale=2.5,
                                          band_order=["red", "green", "blue", "nir"],
                                          tta=4, device="auto")
                        enhance_scene(inp, sr, cfg, unc)
                    out["sr"] = sr
                    out["unc"] = unc if os.path.exists(unc) else None
                st.session_state["ingested"] = out
                st.info("Loaded into the app — open the **Web GIS** and **Downstream** tabs.")
            except Exception as ex:
                st.error(f"Ingest/enhance failed: {ex}")

st.markdown("""
<div class="gov-foot">
  <b>SATSR — Satellite Imagery Super-Resolution &amp; Geospatial Intelligence Platform</b><br>
  Engine: Real-ESRGAN / RRDBNet (PyTorch) · Analysis: YOLOv8 footprints, NDVI / NDWI / NDBI, SWIR wildfire, Frangi road centrelines.<br>
  Imagery: Copernicus Sentinel-2 L2A (ESA) — free and open data · Ingest: Sentinel Hub · Copernicus CDSE · AWS Earth Search.<br>
  <span class="muted">Prototype developed for Smart India Hackathon 2026. This is a demonstration system for evaluation and is not an official Government of India service.
  &nbsp;·&nbsp; © 2026 SATSR project team.</span>
</div>
""", unsafe_allow_html=True)
