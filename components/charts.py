"""
Chart factory for SolarScope.

Each function returns a matplotlib Figure ready for st.pyplot(). Centralized
here so the Streamlit app stays focused on layout and the styling/data
choices for each chart are auditable in one place.
"""

import io
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from pvlib import solarposition


# Centralised palette — slate-base with restrained accents so charts don't
# scream. Each accent is high-contrast against the slate text + grid.
COLOR_PRIMARY = "#2563eb"       # blue-600
COLOR_ACCENT = "#16a34a"        # green-600
COLOR_WARM = "#ea580c"          # orange-600
COLOR_DANGER = "#dc2626"        # red-600
COLOR_MUTED = "#94a3b8"         # slate-400
COLOR_GRID = "#e2e8f0"          # slate-200
COLOR_BG = "#f8fafc"
COLOR_TEXT = "#0f172a"          # slate-900

INDIAN_SEASONS = [
    ("Winter (Dec–Feb)",       [12, 1, 2],   "#2563eb"),
    ("Pre-monsoon (Mar–May)",  [3, 4, 5],    "#ea580c"),
    ("Monsoon (Jun–Sep)",      [6, 7, 8, 9], "#0891b2"),
    ("Post-monsoon (Oct–Nov)", [10, 11],     "#7c3aed"),
]

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _style_axes(ax, title=None, ylabel=None, xlabel=None, title_pad=8):
    if title:
        ax.set_title(title, fontsize=11, fontweight="600",
                     color=COLOR_TEXT, pad=title_pad)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=COLOR_TEXT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=COLOR_TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_GRID)
    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    ax.grid(axis="y", alpha=0.5, linewidth=0.5, color=COLOR_GRID)


# ---- G2: seasonal daily generation curves ----------------------------------
def chart_seasonal_daily_curves(hourly_ac_kw: pd.Series) -> plt.Figure:
    """One curve per Indian season showing the average hour-of-day generation."""
    ist = hourly_ac_kw.index.tz_convert("Asia/Kolkata")
    df = pd.DataFrame({"kw": hourly_ac_kw.values, "month": ist.month, "hour": ist.hour})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, months, color in INDIAN_SEASONS:
        subset = df[df["month"].isin(months)]
        if subset.empty:
            continue
        avg = subset.groupby("hour")["kw"].mean()
        ax.plot(avg.index, avg.values, label=label, color=color, linewidth=2.2)
        ax.fill_between(avg.index, 0, avg.values, color=color, alpha=0.08)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)
    _style_axes(ax,
                title="Average hourly generation by season (IST)",
                ylabel="Power output (kW)",
                xlabel="Hour of day")
    fig.tight_layout()
    return fig


# ---- G3: cumulative annual generation --------------------------------------
def chart_cumulative_generation(monthly_kwh: pd.Series) -> plt.Figure:
    """Running total of generation across the year, with the slowest months
    annotated."""
    cum = monthly_kwh.cumsum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(MONTH_LABELS[:len(cum)], cum.values, color=COLOR_PRIMARY,
            linewidth=2.6, marker="o", markersize=6)
    ax.fill_between(MONTH_LABELS[:len(cum)], 0, cum.values,
                    color=COLOR_PRIMARY, alpha=0.10)

    # Annotate slowest stretch (June–September dip)
    monsoon_idx = [i for i, m in enumerate(MONTH_LABELS[:len(cum)]) if m in ("Jul", "Aug")]
    if monsoon_idx:
        i = monsoon_idx[0]
        ax.annotate("monsoon",
                    xy=(MONTH_LABELS[i], cum.values[i]),
                    xytext=(MONTH_LABELS[i], cum.values[i] * 1.20),
                    color=COLOR_DANGER, fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color=COLOR_DANGER, lw=1.2))

    _style_axes(ax,
                title="Cumulative annual generation",
                ylabel="kWh produced (running total)",
                xlabel=None)
    fig.tight_layout()
    return fig


# ---- W3: peak sun-hours by month -------------------------------------------
def chart_peak_sun_hours(weather_df: pd.DataFrame) -> plt.Figure:
    """Average daily peak-sun-hours (kWh/m²/day) per month."""
    ist = weather_df.index.tz_convert("Asia/Kolkata")
    df = pd.DataFrame({"ghi": weather_df["ghi"].values, "month": ist.month,
                       "date": ist.date})
    daily = df.groupby(["month", "date"])["ghi"].sum() / 1000.0  # Wh -> kWh
    monthly_avg = daily.groupby("month").mean()
    months = monthly_avg.index.tolist()
    values = monthly_avg.values.tolist()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([MONTH_LABELS[m - 1] for m in months], values,
                  color=COLOR_WARM, alpha=0.9, edgecolor="white", linewidth=1.2)
    avg = float(np.mean(values))
    ax.axhline(avg, color=COLOR_MUTED, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(values) - 0.5, avg + 0.05, f"avg {avg:.2f}",
            color=COLOR_MUTED, fontsize=9, ha="right")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.1f}",
                ha="center", fontsize=8, color=COLOR_TEXT)
    _style_axes(ax,
                title="Peak sun-hours by month",
                ylabel="kWh/m² per day")
    fig.tight_layout()
    return fig


# ---- W4: sun-path diagram (polar) ------------------------------------------
def chart_sun_path(lat: float, lng: float) -> plt.Figure:
    """Polar plot of the sun's daily arc on the 21st of each month.
    Radial axis = zenith angle (centre = overhead, edge = horizon).
    Angular axis = compass azimuth (N at top going clockwise)."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise so E is on the right
    ax.set_rlim(0, 90)
    ax.set_rticks([15, 30, 45, 60, 75])
    ax.set_yticklabels([f"{int(90 - r)}°" for r in [15, 30, 45, 60, 75]],
                        fontsize=8, color=COLOR_MUTED)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=11, color=COLOR_TEXT)

    # The two equinoxes (Mar 21 and Sep 21) trace nearly identical paths —
    # the sun crosses the celestial equator on both dates, so the geometry
    # is the same. Showing both as separate lines is misleading; we render
    # ONE "equinox" path (using Sep) labeled accordingly.
    paths_to_plot = [
        # (month, label, color, linestyle, linewidth)
        (6,  "Summer solstice (Jun 21)",  "#dc2626", "-",  2.4),
        (9,  "Equinoxes (Mar 21 / Sep 21)", "#ea580c", "-",  2.4),
        (12, "Winter solstice (Dec 21)",  "#2563eb", "-",  2.4),
    ]

    for m, label, color, ls, lw in paths_to_plot:
        times = pd.date_range(
            start=f"2025-{m:02d}-21 00:00",
            end=f"2025-{m:02d}-21 23:00",
            freq="15min", tz="Asia/Kolkata",
        )
        sp = solarposition.get_solarposition(times, lat, lng)
        daylight = sp["apparent_elevation"] > 0
        az_rad = np.deg2rad(sp.loc[daylight, "azimuth"].values)
        zen = sp.loc[daylight, "apparent_zenith"].values
        ax.plot(az_rad, zen, color=color, linewidth=lw,
                linestyle=ls, label=label)

    ax.set_title(f"Sun path at {lat:.2f}°N, {lng:.2f}°E",
                 fontsize=13, fontweight="600", color=COLOR_TEXT, pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.20, 0.0),
              frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ---- L2: annual energy ideal vs actual -------------------------------------
def chart_ideal_vs_actual(pvwatts: dict, system_size_kw: float,
                          weather_df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing nameplate-ideal generation to actual after losses
    and temperature derating."""
    annual_ghi_kwh_m2 = float(weather_df["ghi"].sum()) / 1000.0
    # Ideal = system size * annual peak-sun-hours (no losses, no temp derate).
    ideal = system_size_kw * annual_ghi_kwh_m2
    actual = pvwatts["annual_kwh"]
    loss_kwh = max(ideal - actual, 0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(["Ideal\n(no losses)", "Actual\n(after losses)"],
                  [ideal, actual],
                  color=[COLOR_MUTED, COLOR_ACCENT],
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, [ideal, actual]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + ideal * 0.015,
                f"{val:,.0f} kWh", ha="center", fontsize=10,
                color=COLOR_TEXT, fontweight="600")
    pct = (loss_kwh / ideal * 100) if ideal > 0 else 0
    ax.text(0.5, ideal * 0.55,
            f"loss\n−{loss_kwh:,.0f} kWh\n({pct:.1f}%)",
            ha="center", fontsize=10, color=COLOR_DANGER,
            transform=ax.transData)
    _style_axes(ax,
                title="Annual energy: ideal vs actual",
                ylabel="kWh per year")
    fig.tight_layout()
    return fig


# ---- F1b: money-flow waterfall (replaces simple cumulative line) ---------
def chart_money_flow_waterfall(
    financial: dict,
    system_lifetime: int = 20,
    degradation: float = 0.005,
) -> plt.Figure:
    """Waterfall: each year is one bar that adds to the running cumulative
    position. Year 0 is the system cost (red, dropping below zero); years
    1..N are annual savings (amber while still recovering, green once in
    profit). A connecting line shows the running net position."""
    cost = financial["system_cost_inr"]
    annual_save0 = financial["annual_savings_inr"]

    yearly_savings = annual_save0 * (1 - degradation) ** np.arange(system_lifetime)

    deltas = np.concatenate([[-cost], yearly_savings])      # year 0..N
    starts = np.concatenate([[0], np.cumsum(deltas)[:-1]])  # bar starts where prev ended
    cumulative = np.cumsum(deltas)

    # Colour each bar:
    #   year 0   → red (the cost)
    #   recovery → amber (running total still negative)
    #   profit   → green (running total now positive)
    bar_colors = []
    for i in range(len(deltas)):
        end_pos = starts[i] + deltas[i]
        if i == 0:
            bar_colors.append(COLOR_DANGER)
        elif end_pos < 0:
            bar_colors.append(COLOR_WARM)
        else:
            bar_colors.append(COLOR_ACCENT)

    fig, ax = plt.subplots(figsize=(9, 4.0))
    xs = np.arange(len(deltas))
    ax.bar(xs, deltas, bottom=starts, color=bar_colors,
           edgecolor="white", linewidth=0.6, width=0.78)

    # Running-total line on top of the bars
    ax.plot(xs, cumulative, color=COLOR_PRIMARY, linewidth=1.6,
            marker="o", markersize=3, alpha=0.7)
    ax.axhline(0, color=COLOR_MUTED, linewidth=0.7)

    # Find payback year (first year cumulative >= 0)
    payback_year = next((i for i, c in enumerate(cumulative) if c >= 0), None)
    if payback_year is not None and payback_year > 0:
        ax.axvline(payback_year, color=COLOR_DANGER, linestyle="--",
                   linewidth=1.2, alpha=0.7)
        ax.annotate(f"payback @ year {payback_year}",
                    xy=(payback_year, 0),
                    xytext=(payback_year + 0.6, cost * 0.35),
                    fontsize=9, color=COLOR_DANGER,
                    arrowprops=dict(arrowstyle="->", color=COLOR_DANGER, lw=1.0))

    # Annotate the first bar (cost) and the final cumulative position.
    ax.text(0, deltas[0] / 2, f"−₹{cost / 100000:.1f}L\ncost",
            ha="center", va="center", color="white",
            fontsize=8, fontweight="700")

    final_total = cumulative[-1]
    ax.text(xs[-1], final_total + cost * 0.05,
            f"₹{final_total / 100000:.1f}L\nnet",
            ha="center", va="bottom", color=COLOR_TEXT,
            fontsize=9, fontweight="700",
            bbox=dict(facecolor="white", edgecolor=COLOR_GRID,
                      boxstyle="round,pad=0.25"))

    # Legend (above the chart so it doesn't overlap any bars)
    legend_handles = [
        Patch(facecolor=COLOR_DANGER, label="Year 0 — system cost"),
        Patch(facecolor=COLOR_WARM, label="Recovery years (still in red)"),
        Patch(facecolor=COLOR_ACCENT, label="Profit years (in the green)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=9,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, columnspacing=2.0)

    ax.set_xticks(range(0, len(deltas), 2))
    _style_axes(ax,
                title=f"Money flow - pay down cost, then profit accumulates ",
                ylabel="Net position (₹)",
                xlabel="Year",
                title_pad=30)
    fig.tight_layout()
    return fig


# ---- F1: cumulative savings vs years -------------------------------------
def chart_cumulative_savings(
    financial: dict,
    system_lifetime: int = 20,
    degradation: float = 0.005,
) -> plt.Figure:
    """Cumulative savings curve. Today's electricity tariff held constant
    (no forward-looking inflation modeled — that's outside the project's
    scope) but panel degradation IS modeled because it's physics, not a
    forecast.

    Crosses zero at the payback year."""
    cost = financial["system_cost_inr"]
    annual_save0 = financial["annual_savings_inr"]

    # Each year's savings = year-1 savings reduced by cumulative degradation.
    yearly = annual_save0 * (1 - degradation) ** np.arange(system_lifetime)
    cumulative = np.concatenate([[-cost], -cost + np.cumsum(yearly)])
    years = np.arange(0, system_lifetime + 1)

    # Find the payback year (linear-interpolate the zero crossing)
    payback_real = None
    for i in range(1, len(cumulative)):
        if cumulative[i - 1] < 0 <= cumulative[i]:
            frac = -cumulative[i - 1] / (cumulative[i] - cumulative[i - 1])
            payback_real = (i - 1) + frac
            break

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.fill_between(years, cumulative, 0,
                    where=(cumulative >= 0), color=COLOR_ACCENT, alpha=0.20,
                    interpolate=True)
    ax.fill_between(years, cumulative, 0,
                    where=(cumulative < 0), color=COLOR_DANGER, alpha=0.20,
                    interpolate=True)
    ax.plot(years, cumulative, color=COLOR_PRIMARY, linewidth=2.6)
    ax.axhline(0, color=COLOR_MUTED, linestyle="-", linewidth=0.8)

    if payback_real is not None:
        ax.axvline(payback_real, color=COLOR_DANGER, linestyle="--", linewidth=1.4)
        ax.annotate(f"payback @ {payback_real:.1f} yr",
                    xy=(payback_real, 0), xytext=(payback_real + 0.6, cost * 0.45),
                    fontsize=10, color=COLOR_DANGER,
                    arrowprops=dict(arrowstyle="->", color=COLOR_DANGER, lw=1.2))

    ax.set_xticks(range(0, system_lifetime + 1, 2))
    title_pb = f"{payback_real:.1f}" if payback_real else f"{financial['payback_years']:.1f}"
    _style_axes(ax,
                title=f"Cumulative savings — payback in {title_pb} years "
                      f"(–{degradation*100:.1f}%/yr panel degradation, "
                      f"electricity tariff held at today's rate)",
                ylabel="Net cumulative ₹",
                xlabel="Year")
    fig.tight_layout()
    return fig


# ---- F2: monthly bill before vs after --------------------------------------
def chart_monthly_bill(monthly_kwh: pd.Series, financial: dict,
                       avg_household_kwh_month: float = 350) -> plt.Figure:
    """Side-by-side bars: typical pre-solar bill vs post-solar bill per month.

    avg_household_kwh_month is a sane default for an Indian residential
    household (~350 kWh/month). Bill = consumption × electricity_rate.
    Post-solar = max(0, consumption - solar_generation) × rate.
    """
    rate = financial["electricity_rate"]
    months = MONTH_LABELS[:len(monthly_kwh)]
    pre_bill = np.array([avg_household_kwh_month * rate] * len(months))
    post_bill = np.array([
        max(0, avg_household_kwh_month - kwh) * rate
        for kwh in monthly_kwh.values
    ])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(months))
    width = 0.40
    ax.bar(x - width / 2, pre_bill, width, label="Before solar",
           color=COLOR_MUTED, alpha=0.85, edgecolor="white", linewidth=1)
    ax.bar(x + width / 2, post_bill, width, label="After solar",
           color=COLOR_ACCENT, alpha=0.95, edgecolor="white", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax,
                title=f"Monthly electricity bill — before vs after "
                      f"(assumes {avg_household_kwh_month:.0f} kWh/mo household)",
                ylabel="₹ per month")
    fig.tight_layout()
    return fig


# ---- F3: cost breakdown ----------------------------------------------------
def chart_cost_breakdown(financial: dict) -> plt.Figure:
    """Stacked horizontal bar: pre-subsidy cost, with subsidy carved out.
    Legend is placed ABOVE the bar so it doesn't overlap the green segment."""
    pre = financial["system_cost_pre_subsidy_inr"]
    post = financial["system_cost_inr"]
    subsidy = max(pre - post, 0)

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.barh(["System cost"], [post], color=COLOR_PRIMARY, label="You pay")
    ax.barh(["System cost"], [subsidy], left=[post],
            color=COLOR_ACCENT, label="Government subsidy")

    ax.text(post / 2, 0, f"₹{post/100000:.1f}L",
            ha="center", va="center", color="white",
            fontsize=11, fontweight="700")
    ax.text(post + subsidy / 2, 0, f"₹{subsidy/100000:.1f}L",
            ha="center", va="center", color="white",
            fontsize=11, fontweight="700")

    # Place legend above the bar, below the title — outside the graph data.
    ax.legend(frameon=False, fontsize=10,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, columnspacing=2.5,
              handlelength=1.2)
    ax.set_xlim(0, pre * 1.05)
    ax.set_yticks([])
    _style_axes(ax,
                title=f"System cost breakdown (₹{pre:,.0f} pre-subsidy)",
                title_pad=30)
    fig.tight_layout()
    return fig


# ---- F4: lifetime cash flow with degradation + inflation -------------------
def chart_lifetime_cashflow(financial: dict, degradation: float = 0.005,
                            rate_inflation: float = 0.03,
                            system_lifetime: int = 20) -> plt.Figure:
    """Per-year savings bars factoring in panel degradation and Indian
    electricity-rate inflation."""
    annual_save0 = financial["annual_savings_inr"]
    years = np.arange(1, system_lifetime + 1)
    savings = annual_save0 * (1 - degradation) ** (years - 1) * (1 + rate_inflation) ** (years - 1)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.bar(years, savings / 1000.0, color=COLOR_ACCENT,
                  alpha=0.85, edgecolor="white", linewidth=0.8)
    bars[0].set_color(COLOR_PRIMARY)
    ax.set_xticks(range(1, system_lifetime + 1, 2))
    total = float(savings.sum())
    ax.text(years[-1], (savings / 1000).max() * 0.95,
            f"{system_lifetime}-yr total\n₹{total/100000:.1f} L",
            ha="right", va="top", fontsize=9, color=COLOR_TEXT,
            bbox=dict(facecolor="white", edgecolor=COLOR_MUTED,
                      boxstyle="round,pad=0.4"))
    _style_axes(ax,
                title=f"{system_lifetime}-year savings trajectory "
                      f"(–{degradation*100:.1f}%/yr degradation, "
                      f"+{rate_inflation*100:.0f}%/yr rate inflation)",
                ylabel="Savings (₹ thousand)",
                xlabel="Year")
    fig.tight_layout()
    return fig


# Backwards-compatible alias used by app.py
def chart_20_year_cashflow(financial: dict) -> plt.Figure:
    return chart_lifetime_cashflow(financial, degradation=0.005,
                                    rate_inflation=0.03, system_lifetime=20)


# ---- F5: sensitivity — payback vs electricity rate -------------------------
def chart_sensitivity_rate(financial: dict, annual_kwh: float) -> plt.Figure:
    """How payback period changes if the electricity rate is higher/lower
    than today's assumption."""
    rates = np.linspace(4.0, 10.0, 50)
    cost = financial["system_cost_inr"]
    payback_curve = np.where(rates > 0, cost / (annual_kwh * rates), np.nan)

    current_rate = financial["electricity_rate"]
    current_payback = financial["payback_years"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rates, payback_curve, color=COLOR_PRIMARY, linewidth=2.4)
    ax.fill_between(rates, payback_curve, color=COLOR_PRIMARY, alpha=0.10)
    ax.scatter([current_rate], [current_payback], color=COLOR_DANGER, zorder=5, s=80)
    ax.annotate(f"current\n₹{current_rate}/kWh\n→ {current_payback:.1f} yr",
                xy=(current_rate, current_payback),
                xytext=(current_rate + 0.5, current_payback + 0.5),
                fontsize=9, color=COLOR_DANGER,
                arrowprops=dict(arrowstyle="->", color=COLOR_DANGER, lw=1.2))
    _style_axes(ax,
                title="Payback period sensitivity to electricity rate",
                ylabel="Payback (years)",
                xlabel="Electricity rate (₹/kWh)")
    fig.tight_layout()
    return fig


# ---- S1: per-panel shade ---------------------------------------------------
def chart_per_panel_shade(panels: list, shade_fraction_map: np.ndarray) -> plt.Figure:
    """Per-panel annual shade fraction, sorted ascending."""
    if not panels:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No panels placed.", ha="center", va="center",
                color=COLOR_MUTED, fontsize=12)
        ax.set_axis_off()
        return fig

    shades = []
    for x, y, w, h in panels:
        slot = shade_fraction_map[y:y + h, x:x + w]
        shades.append(float(slot.mean()) * 100)
    shades_sorted = sorted(shades)

    fig, ax = plt.subplots(figsize=(9, 4.0))
    indices = np.arange(1, len(shades_sorted) + 1)
    colors = [COLOR_ACCENT if s < 5 else COLOR_WARM if s < 10 else COLOR_DANGER
              for s in shades_sorted]
    ax.bar(indices, shades_sorted, color=colors,
           edgecolor="white", linewidth=0.6)
    ax.axhline(10, color=COLOR_DANGER, linestyle="--", linewidth=1)
    ax.text(len(indices) - 0.5, 10.5, "10% threshold",
            ha="right", color=COLOR_DANGER, fontsize=9)

    legend_elems = [
        Patch(facecolor=COLOR_ACCENT, label="< 5% (great)"),
        Patch(facecolor=COLOR_WARM, label="5–10% (ok)"),
        Patch(facecolor=COLOR_DANGER, label="> 10% (poor)"),
    ]
    ax.legend(handles=legend_elems, frameon=False, fontsize=9, loc="upper left")
    _style_axes(ax,
                title=f"Per-panel annual shade ({len(panels)} panels, sorted)",
                ylabel="Shade fraction (%)",
                xlabel="Panel rank (least → most shaded)")
    fig.tight_layout()
    return fig


# ---- S3: bigger roof shade heatmap -----------------------------------------
def chart_roof_shade_heatmap(image_bytes: bytes, mask: np.ndarray,
                              shade_fraction_map: np.ndarray) -> plt.Figure:
    """Roof outline with shade-fraction colormap painted on, big and clean."""
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    norm = np.clip(shade_fraction_map / max(shade_fraction_map.max(), 1e-6), 0, 1)
    cmap = plt.get_cmap("RdYlGn_r")
    heat_rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)

    overlay = image.copy()
    overlay[mask] = (0.40 * overlay[mask] + 0.60 * heat_rgb[mask]).astype(np.uint8)

    # Crop to roof bbox + 15% pad for tight framing
    ys, xs = np.where(mask)
    if ys.size > 0:
        H, W = mask.shape
        pad_h = int((ys.max() - ys.min()) * 0.15)
        pad_w = int((xs.max() - xs.min()) * 0.15)
        y0 = max(0, ys.min() - pad_h); y1 = min(H, ys.max() + pad_h)
        x0 = max(0, xs.min() - pad_w); x1 = min(W, xs.max() + pad_w)
        overlay = overlay[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_axis_off()
    ax.set_title("Annual shade map (red = heavy shade, green = sunny)",
                 fontsize=13, fontweight="600", color=COLOR_TEXT, pad=10)

    # Compact horizontal colorbar at bottom
    cax = fig.add_axes([0.20, 0.04, 0.60, 0.025])
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=float(shade_fraction_map.max()) * 100),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Annual shade (%)", fontsize=9, color=COLOR_TEXT)
    cb.ax.tick_params(labelsize=8, colors=COLOR_MUTED)
    return fig
