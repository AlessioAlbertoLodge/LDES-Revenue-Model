"""
Live day‑ahead prices from ENTSO‑E.

Requirements
------------
    pip install entsoe-py pandas

This *lean* helper sticks to the original
`country‑string → bidding‑zone‑code` table so you can extend it by hand.

Add or change entries in the ``_BIDDING_ZONE`` dictionary below whenever you
need another market.  If a zone ever fails with *NoMatchingDataError*, double
check that the right ENTSO‑E bidding‑zone identifier is in the mapping.
"""
from __future__ import annotations
from datetime import timedelta
import datetime as dt
from typing import Final
import pandas as pd
from entsoe import EntsoePandasClient  # returns a pandas Series
# ---------------------------------------------------------------------------
# 1)  Minimal mapping ISO‑2 / common names → ENTSO‑E bidding‑zone codes
# ---------------------------------------------------------------------------
_BIDDING_ZONE: Final[dict[str, str]] = {
    # Core Central‑Western Europe
    "DE": "DE_LU",           # Germany (+ Luxembourg)
    "GERMANY": "DE_LU",
    "LU": "DE_LU",          # Luxembourg (same zone as DE)
    "LUXEMBOURG": "DE_LU",
    "FR": "FR",             # France
    "FRANCE": "FR",
    "BE": "BE",             # Belgium
    "BELGIUM": "BE",
    "NL": "NL",             # Netherlands
    "NETHERLANDS": "NL",

    # Iberian Peninsula
    "ES": "ES",             # Spain
    "SPAIN": "ES",
    "PT": "PT",             # Portugal (shares MIBEL market)
    "PORTUGAL": "PT",

    # British Isles
    "GB": "GB",             # Great Britain bidding zone
    "UK": "UK",             # Whole UK identifier (used by some data)
    "UNITED KINGDOM": "UK",

    # Nordics
    "DK1": "DK_1",          # Jutland / Funen
    "DK2": "DK_2",          # Zealand
    "DK": "DK_1",           # default Denmark → west (edit to taste)
    "SE4": "SE_4",          # Sweden — south (main export link to DE/NL)
    "SE_4": "SE_4",
    "NO2": "NO_2",          # Norway – south (link to NL/GB/DK)
    "NO_2": "NO_2",

    # Alps & neighbours
    "CH": "CH",             # Switzerland
    "SWITZERLAND": "CH",
    "AT": "AT",             # Austria
    "AUSTRIA": "AT",
    "IT": "IT_NORD",
    "ITALY": "IT_NORD",

}


def _to_bidding_zone(country: str) -> str:
    """Return the ENTSO‑E bidding‑zone code for *country* (case‑insensitive)."""
    try:
        return _BIDDING_ZONE[country.upper()]
    except KeyError as err:
        raise ValueError(
            f"Unknown country '{country}'. Please extend _BIDDING_ZONE with the correct ENTSO‑E code."
        ) from err


# ---------------------------------------------------------------------------
# 2)  Public helper: get_day_ahead_prices
# ---------------------------------------------------------------------------

def get_day_ahead_prices_single_day(
    country: str,
    date: dt.date,
    *,
    api_key: str,
) -> pd.DataFrame:
    """Fetch ENTSO‑E day‑ahead prices for *country* on *date*.

    Parameters
    ----------
    country  : "DE", "Germany", "FR", "NL", … (see ``_BIDDING_ZONE``)
    date     : :class:`datetime.date`  – delivery day (calendar date)
    api_key  : ENTSO‑E API key (Transparency Platform)
    """
    zone = _to_bidding_zone(country)

    # ENTSO‑E expects tz‑aware timestamps in *Europe/Brussels*
    start = pd.Timestamp(date, tz="Europe/Brussels")
    end = start + pd.Timedelta(days=1)

    client = EntsoePandasClient(api_key=api_key)
    series = client.query_day_ahead_prices(zone, start=start, end=end)

    # Convert to UTC and tidy up
    df = (
        series.tz_convert("Europe/Brussels")
        .rename("price")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "time"})
        .assign(country=country)
        .sort_values("time", ignore_index=True)
    )
    return df

def get_day_ahead_prices_range(
    country: str,
    start: dt.date,
    end: dt.date,
    *,
    api_key: str,
    tz_out: str = "Europe/Brussels",               # or "Europe/Brussels"
) -> pd.DataFrame:
    """
    Day-ahead prices for *country* from 00:00 local on *start*
    up to (but NOT including) 00:00 local on *end + 1 day*.

    tz_out:  "UTC"  or  "Europe/Brussels"
    """
    zone = _to_bidding_zone(country)

    # One day padding on each side so we can trim precisely
    fetch_start = pd.Timestamp(start - timedelta(days=1), tz="Europe/Brussels")
    fetch_end   = pd.Timestamp(end   + timedelta(days=2), tz="Europe/Brussels")

    client = EntsoePandasClient(api_key=api_key)
    series = client.query_day_ahead_prices(zone, start=fetch_start, end=fetch_end)

    # Work *in Brussels time* so “delivery days” are easy to see
    df = (
        series.tz_convert("Europe/Brussels")
        .rename("price")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "time"})
        .assign(country=country)
    )

    # Trim to the exact window the user expects
    window_start = pd.Timestamp(start, tz="Europe/Brussels")
    window_end   = pd.Timestamp(end + timedelta(days=1), tz="Europe/Brussels")

    df = df.loc[(df["time"] >= window_start) & (df["time"] < window_end)]

    # Optional: convert to UTC for downstream code
    if tz_out.upper() == "UTC":
        df["time"] = df["time"].dt.tz_convert("UTC")

    df = df.sort_values("time", ignore_index=True)
    return df

def get_forecasted_day_ahead_prices(country: str,date: dt.date,api_key: str) -> pd.DataFrame:
    """Fetch ENTSO‑E day‑ahead prices for *country* on *date*.

    Parameters
    ----------
    country  : "DE", "Germany", "FR", "NL", … (see ``_BIDDING_ZONE``)
    date     : :class:`datetime.date`  – delivery day (calendar date)
    api_key  : ENTSO‑E API key (Transparency Platform)
    """
    # ---------------------
    # First, get regressors
    # ---------------------
    date_minus_7  = get_day_ahead_prices_single_day(country, date-7, api_key=api_key)
    date_minus_3  = get_day_ahead_prices_single_day(country, date-3, api_key=api_key)
    date_minus_2  = get_day_ahead_prices_single_day(country, date-2, api_key=api_key)
    date_minus_2  = get_day_ahead_prices_single_day(country, date-1, api_key=api_key)
    #LEAR = fc.load_model()
    df_forecasted = get_day_ahead_prices_single_day(country, date, api_key=api_key)
    return df_forecasted