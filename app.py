"""
SolarScope - full-pipeline Streamlit UI.

Wires together the CV/physics pipeline:
    address -> satellite -> auto-prompt -> SAM segment ->
    shading analysis -> panel layout -> PVWatts -> financials.

Lets the user override the SAM prompt by clicking anywhere on the
satellite image (handles cases where the geocoder pin lands off-roof
or the auto-picker grabs the wrong building).
"""

import io
import os
import sys

# Force UTF-8 on Windows so emoji in print statements (📍, 🛰️) don't
# trip the default cp1252 encoder when Streamlit captures stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Load .env BEFORE any project imports - utils.config reads env vars at
# module import time, so dotenv must populate them first.
from dotenv import load_dotenv
load_dotenv()

# Windows DLL workaround + early torch load (must precede pvlib/scipy)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
    # Streamlit's module-reloader walks `torch.classes.__path__._path`,
    # which torch's custom __getattr__ doesn't support - it spams a noisy
    # RuntimeError on every rerun. Override with an empty list to silence.
    torch.classes.__path__ = []
except Exception:
    pass

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from utils.config import config, validate_environment
from components.pipeline import run_full_analysis
from components.panel_layout import draw_panel_layout
from components import charts


# ---- page setup ------------------------------------------------------------
st.set_page_config(
    page_title="SolarScope",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main > div {
        padding: 1.5rem 1rem;
        max-width: 1500px;
        margin: 0 auto;
    }
    h1 { font-size: 2.2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem; }
    .subhead { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; }
    .stButton>button {
        background: #2563eb; color: white; border: none;
        padding: 0.7rem 1.6rem; border-radius: 8px; font-weight: 600;
    }
    .stButton>button:hover { background: #1d4ed8; }
    .small-note { color: #6b7280; font-size: 0.85rem; }

    /* ---- stat card system ----------------------------------------- */
    .solar-stat {
        background: white;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #94a3b8;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.7rem;
        transition: box-shadow 0.15s;
    }
    .solar-stat:hover { box-shadow: 0 1px 4px rgba(15,23,42,0.05); }
    .solar-stat .label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .solar-stat .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }
    .solar-stat .value .unit {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748b;
        margin-left: 0.2rem;
    }
    .solar-stat .sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* hero card for the headline number */
    .solar-stat.hero {
        border-left: 5px solid #2563eb;
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 65%);
        padding: 1.2rem 1.4rem;
    }
    .solar-stat.hero .value { font-size: 2.4rem; }
    .solar-stat.hero .label { color: #2563eb; }

    /* energy cards */
    .solar-stat.energy { border-left-color: #f59e0b; }

    /* money cards */
    .solar-stat.money { border-left-color: #16a34a; }

    /* impact cards (CO2, daily average) - teal */
    .solar-stat.impact { border-left-color: #0891b2; }

    /* lifetime cards (long-term metrics) - violet */
    .solar-stat.lifetime { border-left-color: #7c3aed; }

    /* footer card spans both grid columns visually */
    .solar-stat.footer { border-left-color: #475569; }
    /* tighten matplotlib charts inside Streamlit so they don't dominate */
    .stPlotly, .stPyplot { max-width: 100%; }
</style>
""", unsafe_allow_html=True)


# ---- helpers ---------------------------------------------------------------
def _crop_to_roof(arr: np.ndarray, mask: np.ndarray, pad_frac: float = 0.15) -> np.ndarray:
    """Crop a panel to the bounding box of `mask` plus padding."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return arr
    H, W = mask.shape
    pad_h = int((ys.max() - ys.min()) * pad_frac)
    pad_w = int((xs.max() - xs.min()) * pad_frac)
    y0 = max(0, ys.min() - pad_h); y1 = min(H, ys.max() + pad_h)
    x0 = max(0, xs.min() - pad_w); x1 = min(W, xs.max() + pad_w)
    return arr[y0:y1, x0:x1]


def _build_overlays(image_bytes: bytes, result: dict) -> dict:
    """Build the four overlay images from a pipeline result."""
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    seg = result["seg"]
    shading = result["shading"]
    layout = result["layout"]

    # Overlay 1: roof (green) + obstacles (red)
    roof_overlay = image.copy()
    roof_overlay[seg["mask"]] = (
        0.5 * roof_overlay[seg["mask"]] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)
    roof_overlay[shading["obstacle_mask"]] = (
        0.4 * roof_overlay[shading["obstacle_mask"]] + 0.6 * np.array([255, 0, 0])
    ).astype(np.uint8)

    # Overlay 2: usable area (cyan)
    usable_overlay = image.copy()
    usable = shading["usable_mask"]
    usable_overlay[usable] = (
        0.4 * usable_overlay[usable] + 0.6 * np.array([0, 200, 255])
    ).astype(np.uint8)

    # Overlay 3: panel layout
    layout_overlay = draw_panel_layout(image, layout["panels"])

    # Crop all to roof bbox + padding for tight framing
    roof_overlay = _crop_to_roof(roof_overlay, seg["mask"])
    usable_overlay = _crop_to_roof(usable_overlay, seg["mask"])
    layout_overlay = _crop_to_roof(layout_overlay, seg["mask"])

    return {
        "roof_overlay": roof_overlay,
        "usable_overlay": usable_overlay,
        "layout_overlay": layout_overlay,
    }


def _annotated_satellite(image_bytes: bytes, prompt_point: tuple[int, int]) -> Image.Image:
    """Satellite image with a red crosshair at the current prompt point."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    px, py = prompt_point
    draw.line([(px - 18, py), (px + 18, py)], fill=(255, 0, 0), width=3)
    draw.line([(px, py - 18), (px, py + 18)], fill=(255, 0, 0), width=3)
    draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline=(255, 0, 0), width=2)
    return img


# ---- caching ---------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _cached_fetch_image(address, lat, lng, zoom, scale):
    """Fetch the satellite image only - no segmentation. Cheap; cached."""
    from utils.image_fetch import fetch_satellite_image_complete
    return fetch_satellite_image_complete(
        address=address, lat=lat, lng=lng, zoom=zoom, scale=scale,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_run(address, lat, lng, zoom, scale, prompt_point):
    """Run the FULL pipeline (segment+shade+layout+PVWatts) at a given
    user-picked prompt. Cached by inputs so clicking elsewhere is the
    only thing that re-runs the heavy work."""
    return run_full_analysis(
        address=address, lat=lat, lng=lng,
        zoom=zoom, scale=scale,
        prompt_point=prompt_point,
    )


# ---- main page -------------------------------------------------------------
def main():
    st.markdown("<h1>SolarScope ☀️</h1>", unsafe_allow_html=True)
    st.markdown(
        '<div class="subhead">See how much solar energy your rooftop can produce - '
        'just from your address.</div>',
        unsafe_allow_html=True,
    )

    val = validate_environment()
    if val["status"] != "ready":
        st.error(f"⚠️ Configuration issue: {val['message']}")
        st.stop()

    # ---- input form --------------------------------------------------------
    with st.container():
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("##### Location")
            input_method = st.radio(
                "Input method",
                options=["Address", "Coordinates"],
                horizontal=True,
                label_visibility="collapsed",
                key="input_method",
            )

            if input_method == "Address":
                address_in = st.text_input(
                    "Enter address",
                    value=st.session_state.get("address", "E-87, Sarita Vihar, Delhi"),
                    placeholder="e.g. 123 Main Street, City, State",
                    key="address_field",
                )
                lat_in = lng_in = None
            else:
                lc, rc = st.columns(2)
                with lc:
                    lat_in = st.number_input(
                        "Latitude",
                        value=st.session_state.get("lat", 28.5239),
                        format="%.6f",
                        key="lat_field",
                    )
                with rc:
                    lng_in = st.number_input(
                        "Longitude",
                        value=st.session_state.get("lng", 77.1592),
                        format="%.6f",
                        key="lng_field",
                    )
                address_in = None

        with col_right:
            st.markdown("##### Settings")
            zoom_in = st.slider(
                "Image zoom level",
                min_value=18, max_value=23, value=21,
                help="Higher = more detail. Zoom 21 is the practical sweet spot.",
                key="zoom_slider",
            )
            st.write("")  # vertical spacer
            run_btn = st.button(
                "⚡ Analyze Solar Potential",
                type="primary",
                use_container_width=True,
            )

    # Trigger analysis on button click - clear any prior manual prompt
    if run_btn:
        if input_method == "Address":
            st.session_state.address = address_in
            st.session_state.lat = None
            st.session_state.lng = None
        else:
            st.session_state.address = None
            st.session_state.lat = lat_in
            st.session_state.lng = lng_in
        st.session_state.zoom = zoom_in
        st.session_state.input_method_used = input_method
        st.session_state.prompt_override = None  # auto-pick on first run
        st.session_state.has_run = True

    # Wait for first analyze click before computing
    if not st.session_state.get("has_run"):
        st.info("👆 Enter an address (or coordinates) and click **Analyze** to see your rooftop's solar potential.")
        return

    parsed_lat = st.session_state.get("lat")
    parsed_lng = st.session_state.get("lng")
    addr_str = st.session_state.get("address")
    prompt_override = st.session_state.get("prompt_override")
    zoom_used = st.session_state.get("zoom", 21)

    # ---- Step 1: fetch the satellite image (always cheap + cached) --------
    use_coords = parsed_lat is not None and parsed_lng is not None
    with st.spinner("Fetching satellite imagery..."):
        sat = _cached_fetch_image(
            address=None if use_coords else addr_str,
            lat=parsed_lat if use_coords else None,
            lng=parsed_lng if use_coords else None,
            zoom=zoom_used,
            scale=2,
        )

    if "error" in sat:
        st.error(f"Could not fetch satellite imagery: {sat['error']}")
        return

    image_bytes = sat["image_data"]
    formatted_addr = sat.get("formatted_address") or addr_str or (
        f"{parsed_lat:.4f}, {parsed_lng:.4f}" if use_coords else "unknown"
    )

    # ---- Step 2: if no prompt yet, show the clickable image and wait ------
    if prompt_override is None:
        st.markdown("---")
        st.markdown(
            "### 👇 Click on your rooftop to start the analysis\n"
            "*The pipeline (segmentation → shading → layout → simulation) "
            "runs only after you mark a point on the building.*"
        )
        sat_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        click = streamlit_image_coordinates(
            sat_pil,
            key=f"first_click_{formatted_addr}_{zoom_used}",
            width=720,
        )
        st.caption(f"📍 {formatted_addr}")

        if click is not None:
            displayed_w = 720
            actual_w = sat_pil.size[0]
            ratio = actual_w / displayed_w
            st.session_state.prompt_override = (
                int(click["x"] * ratio),
                int(click["y"] * ratio),
            )
            st.rerun()
        return  # nothing else to render until they click

    # ---- Step 3: full pipeline at the user's chosen prompt -----------------
    with st.spinner("Running analysis at your point (segment → shade → layout → simulate)..."):
        result = _cached_run(
            address=None if use_coords else addr_str,
            lat=parsed_lat if use_coords else None,
            lng=parsed_lng if use_coords else None,
            zoom=zoom_used,
            scale=2,
            prompt_point=prompt_override,
        )

    if not result.get("success"):
        st.error(f"Analysis failed: {result.get('error', 'unknown error')}")
        return

    st.session_state.last_result = result
    # `image_bytes` and `formatted_addr` already populated above from the
    # cached image fetch; the pipeline returns the same bytes.

    # ---- top row: clickable satellite + key metrics ------------------------
    st.markdown("---")
    col_img, col_metrics = st.columns([1.2, 1])

    with col_img:
        st.markdown("**Satellite view** - *click anywhere on the rooftop to redo segmentation at that point*")
        sat_img = _annotated_satellite(image_bytes, result["prompt_point"])
        click = streamlit_image_coordinates(
            sat_img,
            key=f"prompt_picker_{formatted_addr}_{st.session_state.get('zoom', 21)}",
            width=620,
        )
        st.caption(f"📍 {formatted_addr}")
        st.caption(
            f"Current prompt: ({result['prompt_point'][0]}, {result['prompt_point'][1]})"
            + ("  •  user-picked" if prompt_override else "  •  auto-picked")
        )

        if click is not None:
            click_x = int(click["x"])
            click_y = int(click["y"])
            # streamlit-image-coordinates returns coords scaled to the displayed
            # width. Convert back to original pixel space.
            displayed_w = 620
            actual_w = sat_img.size[0]
            ratio = actual_w / displayed_w
            actual_x = int(click_x * ratio)
            actual_y = int(click_y * ratio)
            new_point = (actual_x, actual_y)
            if new_point != prompt_override:
                st.session_state.prompt_override = new_point
                st.rerun()

        if prompt_override is not None:
            if st.button("↩️ Reset to auto-pick", use_container_width=True):
                st.session_state.prompt_override = None
                st.rerun()

    with col_metrics:
        seg = result["seg"]
        layout = result["layout"]
        pv = result.get("pvwatts")
        fin = result.get("financial")

        if pv is None or fin is None:
            st.warning("⚠️ No usable area found at this prompt point - either no "
                       "panels fit or the rooftop is too shaded. Click somewhere "
                       "else on the rooftop to retry.")
        else:
            # Hero card - annual energy is the headline number a customer cares about
            st.markdown(f"""
<div class="solar-stat hero">
    <div class="label">Annual generation</div>
    <div class="value">{pv['annual_kwh']:,.0f}<span class="unit">kWh / year</span></div>
    <div class="sub">{layout['panel_count']} panels · {layout['system_size_kw']:.2f} kW system · {seg['area_sqft']:,.0f} sq ft of roof</div>
</div>
            """, unsafe_allow_html=True)

            # ---- Compute homeowner-relatable derived stats ---------------
            daily_avg_kwh = pv["annual_kwh"] / 365.0

            # Effective cost of electricity (levelized) over 20 years with degradation
            degradation = 0.005
            yrs = np.arange(20)
            lifetime_kwh = float(np.sum(pv["annual_kwh"] * (1 - degradation) ** yrs))
            effective_cost_per_kwh = (
                fin["system_cost_inr"] / lifetime_kwh if lifetime_kwh > 0 else 0
            )
            grid_savings_multiple = (
                fin["electricity_rate"] / effective_cost_per_kwh
                if effective_cost_per_kwh > 0 else 0
            )

            # Best/worst month - for the seasonal-dip talking point
            monthly = pv["monthly_kwh"]
            best_month = monthly.idxmax()
            worst_month = monthly.idxmin()

            # ---- Row 1: Money ---------------------------------------------
            row1_l, row1_r = st.columns(2)
            with row1_l:
                st.markdown(f"""
<div class="solar-stat money">
    <div class="label">Annual savings</div>
    <div class="value">{fin['annual_savings_formatted']}</div>
</div>""", unsafe_allow_html=True)
            with row1_r:
                st.markdown(f"""
<div class="solar-stat money">
    <div class="label">Payback period</div>
    <div class="value">{fin['payback_years']:.1f}<span class="unit">years</span></div>
</div>""", unsafe_allow_html=True)

            # ---- Row 2: Rooftop areas (side by side) ----------------------
            row2_l, row2_r = st.columns(2)
            with row2_l:
                st.markdown(f"""
<div class="solar-stat energy">
    <div class="label">Total rooftop area</div>
    <div class="value">{seg['area_sqft']:,.0f}<span class="unit">sq ft</span></div>
</div>""", unsafe_allow_html=True)
            with row2_r:
                st.markdown(f"""
<div class="solar-stat lifetime">
    <div class="label">Usable rooftop space</div>
    <div class="value">{result['shading']['usable_area_sqft']:,.0f}<span class="unit">sq ft</span></div>
</div>""", unsafe_allow_html=True)

            # ---- Row 3: Use-case (subtitles kept) -------------------------
            row3_l, row3_r = st.columns(2)
            with row3_l:
                st.markdown(f"""
<div class="solar-stat energy">
    <div class="label">Daily average</div>
    <div class="value">{daily_avg_kwh:.1f}<span class="unit">kWh / day</span></div>
    <div class="sub">runs a 1.5 ton AC for ~{daily_avg_kwh/1.5:.0f} hrs daily</div>
</div>""", unsafe_allow_html=True)
            with row3_r:
                st.markdown(f"""
<div class="solar-stat lifetime">
    <div class="label">Effective electricity cost</div>
    <div class="value">₹{effective_cost_per_kwh:.2f}<span class="unit">/ kWh</span></div>
    <div class="sub">vs grid ₹{fin['electricity_rate']}/kWh - {grid_savings_multiple:.1f}× cheaper</div>
</div>""", unsafe_allow_html=True)

            # ---- Row 4: Seasonal -----------------------------------------
            row4_l, row4_r = st.columns(2)
            with row4_l:
                st.markdown(f"""
<div class="solar-stat energy">
    <div class="label">Best month</div>
    <div class="value">{best_month}<span class="unit">{monthly[best_month]:,.0f} kWh</span></div>
</div>""", unsafe_allow_html=True)
            with row4_r:
                st.markdown(f"""
<div class="solar-stat energy">
    <div class="label">Slowest month</div>
    <div class="value">{worst_month}<span class="unit">{monthly[worst_month]:,.0f} kWh</span></div>
</div>""", unsafe_allow_html=True)

            # Footer - full-width cost card
            st.markdown(f"""
<div class="solar-stat footer">
    <div class="label">System cost (after MNRE subsidy)</div>
    <div class="value">{fin['system_cost_formatted']}<span class="unit">  · pre-subsidy {config.format_currency(fin['system_cost_pre_subsidy_inr'])}</span></div>
    <div class="sub">Subsidy applied: 40% on first 3 kW, 20% above (MNRE 2024)</div>
</div>""", unsafe_allow_html=True)

    # ---- visualizations ----------------------------------------------------
    if pv is not None:
        st.markdown("---")
        st.markdown("### Pipeline visualizations")
        overlays = _build_overlays(image_bytes, result)
        v1, v2, v3 = st.columns(3)
        with v1:
            st.image(
                overlays["roof_overlay"],
                caption=f"Roof (green) + obstacles (red) - {seg['area_sqft']:,.0f} sq ft",
                use_container_width=True,
            )
        with v2:
            st.image(
                overlays["usable_overlay"],
                caption=f"Usable area - avg shade {result['shading']['avg_shade_pct']:.1f}%",
                use_container_width=True,
            )
        with v3:
            st.image(
                overlays["layout_overlay"],
                caption=f"Panel layout - {layout['panel_count']} panels, "
                        f"{layout['orientation']}",
                use_container_width=True,
            )

        # ---- monthly generation curve --------------------------------------
        st.markdown("### Monthly generation")
        monthly = pv["monthly_kwh"]
        # Rename the series + give the index a label so Streamlit's tooltip
        # shows "Month: Feb, kWh: 899" instead of "0   899   index Feb".
        monthly_for_chart = monthly.rename("kWh per month").rename_axis("Month")
        st.bar_chart(monthly_for_chart, height=260, color="#2563eb")
        st.caption(
            f"Specific yield: {pv['specific_yield']:,.0f} kWh/kWp/year   •   "
            f"Capacity factor: {pv['capacity_factor_pct']:.1f}%   •   "
            f"Peak AC: {pv['peak_ac_kw']:.2f} kW"
        )

        # ---- deep-dive tabs ------------------------------------------------
        st.markdown("---")
        st.markdown("### 📊 Deep-dive analytics")
        tab_gen, tab_geom, tab_loss, tab_finance = st.tabs([
            "Generation patterns",
            "Solar geometry",
            "Loss attribution",
            "Financial story",
        ])

        # ---- Tab 1: Generation patterns ----
        with tab_gen:
            st.markdown(
                "**How energy flows through your day, season, and year.** "
            )
            cg1, cg2 = st.columns(2)
            with cg1:
                st.pyplot(charts.chart_seasonal_daily_curves(pv["hourly_ac_kw"]),
                          use_container_width=True)
                st.caption("Hour-by-hour generation averaged within each Indian season. "
                           "Longer arcs in summer, lower & shorter in monsoon.")
            with cg2:
                st.pyplot(charts.chart_cumulative_generation(monthly),
                          use_container_width=True)
                st.caption("Running total of kWh through the year. The slope flattens "
                           "during monsoon.")

            if result.get("weather") is not None:
                st.pyplot(charts.chart_peak_sun_hours(result["weather"]),
                          use_container_width=True)
                st.caption("Daily peak-sun-hours, how usable each month's sky is.")

        # ---- Tab 2: Solar geometry ----
        with tab_geom:
            st.markdown(
                "**Where the sun is and where shadows fall on your roof.** "
            )
            cg1, cg2 = st.columns([1, 1])
            coords = result["sat_result"]["coordinates"]
            with cg1:
                st.pyplot(charts.chart_sun_path(coords["lat"], coords["lng"]),
                          use_container_width=True)
                st.caption(
                    "Sun's daily arc on the equinoxes and solstices. Centre = directly "
                    "overhead, edge = horizon. Summer arcs are higher; winter arcs hug "
                    "the south."
                )
            with cg2:
                st.pyplot(
                    charts.chart_roof_shade_heatmap(
                        image_bytes, seg["mask"],
                        result["shading"]["shade_fraction_map"],
                    ),
                    use_container_width=True,
                )
                st.caption("Annual shade fraction per roof pixel - green = sunny, red = "
                           "heavily shaded.")

        # ---- Tab 3: Loss attribution ----
        with tab_loss:
            st.markdown(
                "**Where every kWh goes between sunlight and your meter.** "
                "'tells why isn't this number 100%?'"
            )
            if result.get("weather") is not None:
                st.pyplot(
                    charts.chart_ideal_vs_actual(
                        pv, layout["system_size_kw"], result["weather"]
                    ),
                    use_container_width=True,
                )
                st.caption("'Ideal' = nameplate × annual peak-sun-hours, no losses or "
                           "heat. 'Actual' = after PVWatts loss stack + temperature derating + inverter.")

            # ---- Full loss breakdown ---------------------------------------
            ws = result.get("weather_summary") or {}
            ghi_annual = ws.get("annual_ghi_kwh_m2", 0)
            ideal_kwh = layout["system_size_kw"] * ghi_annual
            actual_kwh = pv["annual_kwh"]
            total_loss_pct = ((1 - actual_kwh / ideal_kwh) * 100) if ideal_kwh > 0 else 0
            stack_loss_pct = pv["loss_breakdown"]["total_combined_pct"]

            # Inverter loss is roughly (1 - eta_inv_nom) under most loading
            inverter_eff = pv["system_specs"]["inverter_efficiency"]
            inv_loss_pct = (1 - inverter_eff) * 100

            # Temperature + POA effects: the residual after stack and inverter,
            # solved multiplicatively so the three combine to total_loss_pct.
            # (1-stack)(1-inv)(1-temp_poa) = (1-total)
            try:
                retain_total = 1 - total_loss_pct / 100
                retain_stack = 1 - stack_loss_pct / 100
                retain_inv = 1 - inv_loss_pct / 100
                retain_temp_poa = retain_total / (retain_stack * retain_inv)
                temp_poa_loss_pct = (1 - retain_temp_poa) * 100
            except Exception:
                temp_poa_loss_pct = max(total_loss_pct - stack_loss_pct - inv_loss_pct, 0)

            st.markdown("##### Complete loss breakdown")
            stack_factors_str = (
                "soiling, shading, snow, mismatch, wiring, connections, "
                "LID (light-induced degradation), nameplate rating, age, availability"
            )
            df_full = pd.DataFrame([
                {"category": "Named PVWatts loss stack",
                 "loss %": f"{stack_loss_pct:.2f}",
                 "what it covers": stack_factors_str},
                {"category": "Temperature derating + POA tilt effects",
                 "loss %": f"{temp_poa_loss_pct:.2f}",
                 "what it covers": "panels running hotter than 25°C STC (computed hourly from NASA POWER ambient temp + wind); tilt/azimuth choice vs perfect sun-tracking"},
                {"category": "Inverter DC→AC conversion",
                 "loss %": f"{inv_loss_pct:.2f}",
                 "what it covers": f"~{inverter_eff*100:.0f}% nominal inverter efficiency (AC output / DC input)"},
                {"category": "TOTAL OBSERVED LOSS",
                 "loss %": f"{total_loss_pct:.2f}",
                 "what it covers": "end-to-end simulation loss vs ideal nameplate × annual irradiance"},
            ])
            st.dataframe(df_full, hide_index=True, use_container_width=True)
            st.caption("Three loss buckets combine multiplicatively: "
                       "(1−stack)·(1−temp)·(1−inverter) = (1−total).")

            with st.expander("📋 Named loss stack - individual factors (PVWatts v5)"):
                losses = pv["loss_breakdown"]
                df_losses = pd.DataFrame(
                    [{"loss factor": k.replace("_", " "), "%": v}
                     for k, v in losses.items()]
                )
                st.dataframe(df_losses, hide_index=True, use_container_width=True)
                st.caption("These 10 factors combine multiplicatively into the "
                           f"{stack_loss_pct:.2f}% named stack shown in the main table.")

        # ---- Tab 4: Financial story ----
        with tab_finance:
            st.markdown(
                "**The customer's view: what it costs, what you save, when it pays off.**"
            )
            st.pyplot(charts.chart_money_flow_waterfall(fin),
                      use_container_width=True)
            st.caption("Each bar is one year. Year 0 is the system cost "
                       "(below zero). Amber bars are years where you're still "
                       "paying yourself back; green bars are pure profit.")

            st.pyplot(charts.chart_cost_breakdown(fin),
                      use_container_width=True)
            st.caption("System cost split between ur money and government "
                       "subsidy (MNRE 2024: 40% on first 3 kW, 20% above).")


if __name__ == "__main__":
    main()
