"""
End-to-end SolarScope pipeline orchestrator.

Runs the full chain in one call:
    address -> satellite image -> CV-pick prompt -> SAM segment ->
    shading analysis -> panel layout -> PVWatts simulation -> financials

Public API:
    run_full_analysis(address=None, lat=None, lng=None, ...) -> dict
        returns a unified results bundle with everything Streamlit needs
        to render the analysis.
"""

import os

# Same Windows DLL workaround used in shading_analyzer
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401
except Exception:
    pass

from utils.image_fetch import fetch_satellite_image_complete
from utils.config import config
from components.roof_segmenter import segment_roof, auto_pick_prompt_point
from components.shading_analyzer import analyze_shading
from components.panel_layout import optimize_panel_layout
from components.pvwatts_engine import simulate_annual_generation
from components.nasa_power import fetch_hourly_weather


# ---- financial model (Indian market) ---------------------------------------
def _compute_financials(
    annual_kwh: float,
    system_size_kw: float,
    cost_per_watt: float = None,
    electricity_rate: float = None,
    central_subsidy: float = None,
    system_lifetime: int = None,
) -> dict:
    """
    Compute Indian-market financials: system cost (with central subsidies
    capped at 3 kW), annual savings, payback period, 25-year ROI.

    Subsidy structure (Indian gov't 2024):
        First 3 kW: 40% subsidy
        Above 3 kW: 20% subsidy on the marginal kW
    """
    cost_per_watt = cost_per_watt or config.COST_PER_WATT_INSTALLED
    electricity_rate = electricity_rate or config.DEFAULT_ELECTRICITY_RATE
    central_subsidy = central_subsidy or config.CENTRAL_SUBSIDY
    system_lifetime = system_lifetime or config.SYSTEM_LIFETIME

    system_cost = system_size_kw * 1000.0 * cost_per_watt

    if system_size_kw <= 3.0:
        final_cost = system_cost * (1.0 - central_subsidy)
    else:
        first_3kw_cost = 3.0 * 1000.0 * cost_per_watt
        remaining_kw = system_size_kw - 3.0
        remaining_cost = remaining_kw * 1000.0 * cost_per_watt
        final_cost = (
            first_3kw_cost * (1.0 - central_subsidy)
            + remaining_cost * (1.0 - 0.20)  # 20% subsidy above 3 kW
        )

    annual_savings = annual_kwh * electricity_rate
    payback_years = final_cost / annual_savings if annual_savings > 0 else 0
    total_lifetime_savings = annual_savings * system_lifetime
    roi_pct = (
        (total_lifetime_savings - final_cost) / final_cost * 100
        if final_cost > 0
        else 0
    )

    return {
        "system_cost_inr": round(final_cost, 0),
        "system_cost_pre_subsidy_inr": round(system_cost, 0),
        "system_cost_formatted": config.format_currency(final_cost),
        "annual_savings_inr": round(annual_savings, 0),
        "annual_savings_formatted": config.format_currency(annual_savings),
        "payback_years": round(payback_years, 1),
        "lifetime_savings_inr": round(total_lifetime_savings, 0),
        "lifetime_savings_formatted": config.format_currency(total_lifetime_savings),
        "roi_pct": round(roi_pct, 1),
        "electricity_rate": electricity_rate,
        "cost_per_watt": cost_per_watt,
        "central_subsidy_pct": central_subsidy * 100,
        "system_lifetime_years": system_lifetime,
    }


# ---- main API --------------------------------------------------------------
def run_full_analysis(
    address: str | None = None,
    lat: float | None = None,
    lng: float | None = None,
    zoom: int = 21,
    scale: int = 2,
    prompt_point: tuple[int, int] | None = None,
    panel_wattage: int = 330,
    panel_height_m: float = 1.65,
    panel_width_m: float = 1.00,
    debug: bool = False,
) -> dict:
    """
    Run the full SolarScope analysis end-to-end.

    Either `address` OR (`lat`, `lng`) must be provided.

    `prompt_point=None` triggers CV auto-pick. Pass an explicit (x, y) for
    user-clicked override.

    Returns a single bundled dict with keys:
        success, error?, sat_result, seg, shading, layout, weather_summary,
        pvwatts, financial.
    """
    sat = fetch_satellite_image_complete(
        address=address, lat=lat, lng=lng, zoom=zoom, scale=scale
    )
    if "error" in sat:
        return {"success": False, "error": sat["error"]}

    image_bytes = sat["image_data"]
    site_lat = sat["coordinates"]["lat"]
    site_lng = sat["coordinates"]["lng"]

    # 1. CV auto-pick if no manual override
    if prompt_point is None:
        prompt_point = auto_pick_prompt_point(image_bytes)

    # 2. Segment the rooftop
    seg = segment_roof(
        image_bytes,
        lat=site_lat,
        zoom=zoom,
        scale=scale,
        prompt_point=prompt_point,
        debug=debug,
    )

    # 3. Annual shading
    shading = analyze_shading(
        image_bytes=image_bytes,
        roof_mask=seg["mask"],
        lat=site_lat,
        lng=site_lng,
        m_per_pixel=seg["m_per_pixel"],
        debug=debug,
    )

    # 4. Panel layout
    layout = optimize_panel_layout(
        usable_mask=shading["usable_mask"],
        obstacle_mask=shading["obstacle_mask"],
        m_per_pixel=seg["m_per_pixel"],
        panel_height_m=panel_height_m,
        panel_width_m=panel_width_m,
        panel_wattage=panel_wattage,
        debug=debug,
    )

    # 5. NASA POWER + PVWatts simulation (only if we placed any panels)
    pvwatts = None
    weather_summary = None
    weather = None
    if layout["panel_count"] > 0:
        weather = fetch_hourly_weather(site_lat, site_lng)
        weather_summary = {
            "n_hours": int(len(weather)),
            "annual_ghi_kwh_m2": float(weather["ghi"].sum()) / 1000.0,
            "year": int(weather.index[0].year),
        }
        pvwatts = simulate_annual_generation(
            weather=weather,
            latitude=site_lat,
            longitude=site_lng,
            system_size_kw=layout["system_size_kw"],
        )

    # 6. Financials (only if we have generation)
    financial = None
    if pvwatts is not None:
        financial = _compute_financials(
            annual_kwh=pvwatts["annual_kwh"],
            system_size_kw=layout["system_size_kw"],
        )

    return {
        "success": True,
        "sat_result": sat,
        "prompt_point": prompt_point,
        "seg": seg,
        "shading": shading,
        "layout": layout,
        "weather_summary": weather_summary,
        "weather": weather,                  # full DataFrame for charts
        "pvwatts": pvwatts,
        "financial": financial,
    }
