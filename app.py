# app.py  (LDES 1-year perfect-foresight merchant DA revenue backtest)
#
# Drop this in your repo root (next to /src) and run:
#   streamlit run app.py
#
# Assumes you already have:
#   src/entsoe_prices.py   (with get_day_ahead_prices_range)
# And your Streamlit secrets contain:
#   [api_keys]
#   entsoe = "YOUR_TOKEN"

from __future__ import annotations

import sys
import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# allow: from src import ...
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src import entsoe_prices as ep  # uses get_day_ahead_prices_range

# ---------------------------
# Optimizer (1-year LP w/ efficiency)
# ---------------------------
def _pick_solver():
    import cvxpy as cp

    have = set(cp.installed_solvers())
    for s in ("HIGHS", "HiGHS", "ECOS", "OSQP", "SCS", "GLPK"):
        if s in have:
            return s
    raise RuntimeError("No LP solver available for cvxpy (install HiGHS or ECOS).")


def optimise_ldes_year(
    prices_df: pd.DataFrame,  # columns: time (tz-aware ok), price (€/MWh)
    e_max_mwh: float,
    p_charge_max_mw: float,
    p_discharge_max_mw: float,
    soc0_mwh: float,
    eta_rt: float = 0.60,
    dt_hours: float = 1.0,
    terminal_soc: str = "free",  # "free" | "same_as_start" | "empty" | "full"
) -> dict:
    """
    Perfect-foresight arbitrage on a fixed price series.

    Sign convention in outputs:
      power_mw > 0  => discharge (sell)
      power_mw < 0  => charge (buy)

    Efficiency model:
      SOC_{t+1} = SOC_t + eta_c * Pc_t*dt - (1/eta_d) * Pd_t*dt
      with Pc,Pd >= 0 and eta_c = eta_d = sqrt(eta_rt)
    """
    import cvxpy as cp

    df = prices_df.copy()
    if "time" not in df.columns or "price" not in df.columns:
        raise ValueError("prices_df must have columns: time, price (€/MWh)")

    # ensure sorted hourly series
    df = df.sort_values("time", ignore_index=True)
    price = df["price"].to_numpy(dtype=float)  # €/MWh
    T = price.size

    eta_c = float(np.sqrt(eta_rt))
    eta_d = float(np.sqrt(eta_rt))

    # decision variables (MW)
    Pc = cp.Variable(T, nonneg=True)  # charge power (MW)
    Pd = cp.Variable(T, nonneg=True)  # discharge power (MW)
    soc = cp.Variable(T + 1)          # MWh

    constraints = [
        soc[0] == soc0_mwh,
        soc[1:] == soc[:-1] + eta_c * Pc * dt_hours - (Pd * dt_hours) / eta_d,
        soc >= 0,
        soc <= e_max_mwh,
        Pc <= p_charge_max_mw,
        Pd <= p_discharge_max_mw,
    ]

    if terminal_soc == "same_as_start":
        constraints += [soc[-1] == soc0_mwh]
    elif terminal_soc == "empty":
        constraints += [soc[-1] == 0.0]
    elif terminal_soc == "full":
        constraints += [soc[-1] == e_max_mwh]
    elif terminal_soc == "free":
        pass
    else:
        raise ValueError("terminal_soc must be one of: free, same_as_start, empty, full")

    # revenue in €: sum_t price(€/MWh) * (Pd - Pc)(MW) * dt(h)
    revenue = cp.sum(cp.multiply(price, (Pd - Pc) * dt_hours))
    prob = cp.Problem(cp.Maximize(revenue), constraints)
    prob.solve(solver=cp.OSQP, verbose=True, eps_abs=1e-6, eps_rel=1e-6, max_iter=200000)
    print("status:", prob.status)
    Pc_opt = np.asarray(Pc.value, dtype=float).ravel()
    Pd_opt = np.asarray(Pd.value, dtype=float).ravel()
    soc_opt = np.asarray(soc.value, dtype=float).ravel()

    power_mw = (Pd_opt - Pc_opt)
    # €/h per step (dt=1h): price * MW
    revenue_per_step = price * power_mw * dt_hours

    schedule = pd.DataFrame(
        {
            "time": df["time"].to_numpy(),
            "price_eur_per_mwh": price,
            "charge_mw": Pc_opt,
            "discharge_mw": Pd_opt,
            "power_mw": power_mw,     # + discharge, - charge
            "soc_mwh_start": soc_opt[:-1],
            "revenue_eur": revenue_per_step,
        }
    )

    # simple KPIs
    total_charge_mwh = float(np.sum(Pc_opt * dt_hours))
    total_discharge_mwh = float(np.sum(Pd_opt * dt_hours))
    throughput_mwh = total_charge_mwh + total_discharge_mwh
    equiv_cycles = throughput_mwh / (2.0 * e_max_mwh) if e_max_mwh > 0 else np.nan

    return dict(
        status=str(prob.status),
        objective=float(prob.value),
        schedule=schedule,
        total_charge_mwh=total_charge_mwh,
        total_discharge_mwh=total_discharge_mwh,
        throughput_mwh=throughput_mwh,
        equiv_cycles=equiv_cycles,
    )


# ---------------------------
# Plot helpers (matplotlib, lightweight)
# ---------------------------
def plot_price_series(df: pd.DataFrame, max_points: int = 2500):
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"])
    d = d.sort_values("time")
    if len(d) > max_points:
        step = int(np.ceil(len(d) / max_points))
        d = d.iloc[::step, :]

    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot(d["time"], d["price"], linewidth=1.0)
    ax.set_ylabel("Price (€/MWh)")
    ax.set_title("Day-ahead prices (downsampled for display)")
    ax.xaxis.set_major_formatter(DateFormatter("%b"))
    fig.tight_layout()
    return fig


def plot_soc_and_power(schedule: pd.DataFrame, e_max_mwh: float, max_points: int = 2500):
    d = schedule.copy()
    d["time"] = pd.to_datetime(d["time"])
    d = d.sort_values("time")
    if len(d) > max_points:
        step = int(np.ceil(len(d) / max_points))
        d = d.iloc[::step, :]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=150, sharex=True)

    ax1.plot(d["time"], d["soc_mwh_start"], linewidth=1.2)
    ax1.set_ylabel("SOC (MWh)")
    ax1.set_ylim(-0.02 * e_max_mwh, 1.02 * e_max_mwh)
    ax1.set_title("Optimized SOC and power (downsampled for display)")

    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.plot(d["time"], d["power_mw"], linewidth=1.0)
    ax2.set_ylabel("Power (MW)  (+ discharge / - charge)")
    ax2.xaxis.set_major_formatter(DateFormatter("%b"))
    fig.tight_layout()
    return fig


def plot_daily_and_cumulative_revenue(schedule: pd.DataFrame):
    d = schedule.copy()
    d["time"] = pd.to_datetime(d["time"])
    d["date"] = d["time"].dt.date
    daily = d.groupby("date", as_index=False)["revenue_eur"].sum()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date", ignore_index=True)
    daily["cum_revenue"] = daily["revenue_eur"].cumsum()

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    ax.bar(daily["date"], daily["revenue_eur"], width=0.8, alpha=0.85, label="Daily revenue (€)")
    ax.set_ylabel("Daily revenue (€)")
    ax.xaxis.set_major_formatter(DateFormatter("%b"))

    # symmetric around 0 if negatives exist
    rmax = float(daily["revenue_eur"].max())
    rmin = float(daily["revenue_eur"].min())
    if rmin < 0:
        lim = max(abs(rmax), abs(rmin)) * 1.1
        ax.set_ylim(-lim, lim)
        ax.axhline(0, linestyle="--", linewidth=1)

    ax2 = ax.twinx()
    ax2.plot(daily["date"], daily["cum_revenue"], linewidth=2.0, label="Cumulative revenue (€)")
    ax2.set_ylabel("Cumulative revenue (€)")

    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

    start = daily["date"].iloc[0]
    end = daily["date"].iloc[-1]
    ax.set_title(f"Daily & cumulative revenue — {start:%d %b %Y} to {end:%d %b %Y}")
    fig.tight_layout()
    return fig


# ---------------------------
# Streamlit UI (single page)
# ---------------------------
st.set_page_config(page_title="LDES Merchant DA Revenue (1-year)", page_icon="⚡", layout="wide")
st.title("LDES Merchant Day-Ahead Revenue — 1-year perfect-foresight backtest")

left, right = st.columns([1.0, 1.2], gap="large")

with left:
    st.subheader("Market + year")
    market = st.text_input("Market (ISO code, e.g. DE, NL, FR)", value="DE", max_chars=8)

    # keep simple: pick year, always Jan 1 → Dec 31
    year = st.selectbox("Year", options=list(range(2015, dt.date.today().year + 1))[::-1], index=0)

    st.subheader("LDES parameters (defaults = 200 MWh / 2 MW / 60% RTE)")
    e_max_mwh = st.number_input("Energy capacity E (MWh)", min_value=0.0, value=200.0, step=10.0)
    p_mw = st.number_input("Power limit P (MW) (charge & discharge)", min_value=0.0, value=2.0, step=0.5)
    eta_rt = st.number_input("Round-trip efficiency η_rt", min_value=0.01, max_value=1.0, value=0.60, step=0.01)

    soc0_pct = st.number_input("Starting SOC (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    soc0_mwh = e_max_mwh * soc0_pct / 100.0

    terminal_soc = st.selectbox(
        "Terminal SOC constraint",
        options=["free", "same_as_start", "empty", "full"],
        index=1,
        help="For annual backtests, 'same_as_start' is often the cleanest assumption.",
    )

    run = st.button("Run 1-year backtest", type="primary")

with right:
    st.subheader("What this does")
    st.write(
        "Optimizes charge/discharge over the full year using the realized day-ahead prices "
        "(perfect prediction / backtest). Objective is merchant arbitrage revenue only "
        "(no degradation, no fees, no cycle limits)."
    )
    st.write(
        "Model uses separate charge/discharge variables and an SOC balance with efficiency "
        "ηc = ηd = √ηrt."
    )

# ---------------------------
# Data load (cached)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_prices_year(country: str, year: int, api_key: str) -> pd.DataFrame:
    start = dt.date(year, 1, 1)
    end = dt.date(year, 12, 31)
    df = ep.get_day_ahead_prices_range(country, start, end, api_key=api_key, tz_out="Europe/Brussels")

    # Expected to be hourly. Keep only what we need.
    df = df[["time", "price", "country"]].copy()

    # robust: enforce datetime and sort
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time", ignore_index=True)

    # quick sanity: drop duplicates if any
    df = df.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)

    return df

# ---------------------------
# Run
# ---------------------------
if run:
    if "api_keys" not in st.secrets or "entsoe" not in st.secrets["api_keys"]:
        st.error("Missing Streamlit secret: st.secrets['api_keys']['entsoe']")
        st.stop()

    api_key = st.secrets["api_keys"]["entsoe"]

    with st.spinner("Loading day-ahead prices for the full year..."):
        prices = load_prices_year(market, int(year), api_key=api_key)

    if prices.empty:
        st.error("No prices returned for that market/year.")
        st.stop()

    # Basic check: expected ~8760/8784 points
    n = len(prices)
    st.caption(f"Loaded {n:,} hourly prices for {market} {year}.")

    with st.spinner("Solving 1-year LP..."):
        res = optimise_ldes_year(
            prices_df=prices.rename(columns={"price": "price"}),  # keep name
            e_max_mwh=float(e_max_mwh),
            p_charge_max_mw=float(p_mw),
            p_discharge_max_mw=float(p_mw),
            soc0_mwh=float(soc0_mwh),
            eta_rt=float(eta_rt),
            dt_hours=1.0,
            terminal_soc=str(terminal_soc),
        )

    sched = res["schedule"]

    # KPIs
    total_rev = float(sched["revenue_eur"].sum())
    avg_rev_day = total_rev / 365.0
    duration_h = (e_max_mwh / p_mw) if (p_mw > 0) else np.inf

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total revenue (€)", f"{total_rev:,.0f}")
    k2.metric("Avg revenue per day (€)", f"{avg_rev_day:,.0f}")
    k3.metric("Equivalent cycles (throughput / 2E)", f"{res['equiv_cycles']:.2f}")
    k4.metric("Duration (E/P)", f"{duration_h:.1f} h")

    # Plots
    st.subheader("Prices")
    st.pyplot(plot_price_series(prices), use_container_width=True)

    st.subheader("SOC + Power (optimized)")
    st.pyplot(plot_soc_and_power(sched, e_max_mwh=float(e_max_mwh)), use_container_width=True)

    st.subheader("Daily + cumulative revenue")
    st.pyplot(plot_daily_and_cumulative_revenue(sched), use_container_width=True)

    # Download
    st.subheader("Download")
    csv = sched.to_csv(index=False).encode("utf-8")
    st.download_button("Download schedule CSV", data=csv, file_name=f"ldes_schedule_{market}_{year}.csv", mime="text/csv")

else:
    st.info("Set parameters and click **Run 1-year backtest**.")
