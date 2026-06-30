"""
SAHELI — Real NOAA ERSST sea surface temperature ingestion, via ERDDAP
subsetting (essay Layer 4, West African Monsoon seasonal signal).

Previous version downloaded the full 154MB global grid, which failed
repeatedly on a slow connection (confirmed: 7 resume attempts, NOAA's
own server returning timeouts/504s — a real, documented infrastructure
limitation, not a fictional one).

This version uses NOAA's own ERDDAP server (coastwatch.pfeg.noaa.gov),
which supports griddap subsetting: requesting ONLY the six small
oceanic boxes we actually need, as small CSVs (a few hundred KB total
instead of 154MB). Same real ERSSTv5 dataset, same six basins the essay
names as West African Monsoon drivers — just fetched far more
efficiently.

Run this on a machine with normal internet access — this sandbox's
network is restricted to pypi/npm/github and cannot reach noaa.gov.
"""
import requests
import pandas as pd
import os
import time

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "noaa_sst_basins.csv")
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/nceiErsstv5_LonPM180.csv"

TIME_RANGE = "(1990-01-15):1:(2026-05-15)"

# (name, lat_min, lat_max, lon_min, lon_max) — longitude in -180..180
# convention (LonPM180 dataset variant), matching the essay's six basins.
BASINS = {
    "tropical_north_atlantic": (5, 25, -80, -20),
    "gulf_of_guinea":          (-5, 5, -20, 10),
    "mediterranean":           (30, 42, 0, 36),
    "north_atlantic":          (40, 60, -80, -10),
    "equatorial_indian_ocean": (-10, 10, 50, 100),
    "enso_nino34":             (-5, 5, -170, -120),
}


def fetch_basin(name, lat_min, lat_max, lon_min, lon_max):
    query = f"sst[{TIME_RANGE}][(0):1:(0)][({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
    url = f"{ERDDAP_BASE}?{query}"
    print(f"\n[{name}]\n  {url}")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
    except Exception as e:
        print(f"  request failed: {e}")
        return pd.DataFrame()

    from io import StringIO
    df = pd.read_csv(StringIO(r.text), skiprows=[1])  # row 1 is units, not data
    print(f"  {len(df)} grid-point-months downloaded")
    monthly = df.groupby("time")["sst"].mean().reset_index()
    monthly.columns = ["date", f"sst_{name}"]
    return monthly


def main():
    series = []
    for name, (lat_min, lat_max, lon_min, lon_max) in BASINS.items():
        s = fetch_basin(name, lat_min, lat_max, lon_min, lon_max)
        if not s.empty:
            series.append(s)
        time.sleep(2)

    if not series:
        print("\nNo basins retrieved.")
        return

    merged = series[0]
    for s in series[1:]:
        merged = merged.merge(s, on="date", how="outer")
    merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
    merged = merged.sort_values("date")

    for col in merged.columns:
        if col.startswith("sst_"):
            merged["_month"] = merged["date"].dt.month
            clim = merged.groupby("_month")[col].transform("mean")
            merged[col.replace("sst_", "sst_anom_")] = merged[col] - clim
    merged = merged.drop(columns=["_month"], errors="ignore")

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(merged)} months x {len(series)} basins to {OUTPUT_PATH}")
    print(merged.tail(6).to_string())


if __name__ == "__main__":
    main()
