"""
SAHELI — Real Sentinel-2 NDVI ingestion via Microsoft Planetary Computer
(essay Layer 1, satellite vegetation health).

No registration required: the Planetary Computer's STAC API is public,
and asset URLs are signed automatically (short-lived SAS tokens) via
the `planetary_computer` package. This script:
  1. For each of SAHELI's 18 districts, searches for the most recent
     low-cloud (<30%) Sentinel-2 L2A scene within the last 60 days.
  2. Reads ONLY a tiny pixel window (5x5 pixels, ~50x50m) of the Red
     (B04) and Near-Infrared (B08) bands directly from the cloud-
     optimized GeoTIFF via HTTP range requests — NOT the full scene
     (which would be several hundred MB per district).
  3. Computes real NDVI = (NIR-RED)/(NIR+RED) and saves one row per
     district: data/sentinel2_ndvi.csv

Run this on a machine with normal internet access — this sandbox's
network is restricted to pypi/npm/github and cannot reach
planetarycomputer.microsoft.com or Azure Blob Storage.
"""
import os
import time
import numpy as np
import pandas as pd
import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sentinel2_ndvi.csv")
CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

DISTRICTS = {
    "Niamey": (13.51, 2.11), "Zinder": (13.80, 8.99), "Maradi": (13.50, 7.10),
    "Tahoua": (14.88, 5.27), "Agadez": (16.97, 7.99), "Diffa": (13.31, 12.61),
    "Bamako": (12.65, -8.00), "Mopti": (14.49, -4.19), "Timbuktu": (16.77, -3.00),
    "Gao": (16.27, -0.04), "Ouagadougou": (12.36, -1.53), "Dori": (14.03, -0.03),
    "Djibo": (14.10, -1.63), "NDjamena": (12.10, 15.04), "Abeche": (13.83, 20.83),
    "Nouakchott": (18.08, -15.97), "Kiffa": (16.62, -11.40), "Matam": (15.65, -13.25),
}

BOX_DEG = 0.01  # ~1km bbox for the STAC search (we only read 5x5 px from the result)


def fetch_district(catalog, name, lat, lon):
    bbox = [lon - BOX_DEG, lat - BOX_DEG, lon + BOX_DEG, lat + BOX_DEG]
    try:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=60):%Y-%m-%d}/{pd.Timestamp.utcnow():%Y-%m-%d}",
            query={"eo:cloud_cover": {"lt": 30}},
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            max_items=1,
        )
        items = list(search.items())
    except Exception as e:
        print(f"  {name}: search failed — {e}")
        return None

    if not items:
        print(f"  {name}: no low-cloud Sentinel-2 scene in the last 60 days")
        return None

    item = pc.sign(items[0])
    cloud_cover = item.properties.get("eo:cloud_cover")
    date = item.properties.get("datetime")

    try:
        with rasterio.open(item.assets["B04"].href) as red_ds:
            x_native, y_native = warp_transform("EPSG:4326", red_ds.crs, [lon], [lat])
            row, col = red_ds.index(x_native[0], y_native[0])
            window = Window(col - 2, row - 2, 5, 5)
            red = red_ds.read(1, window=window).astype(float)
        with rasterio.open(item.assets["B08"].href) as nir_ds:
            x_native, y_native = warp_transform("EPSG:4326", nir_ds.crs, [lon], [lat])
            row, col = nir_ds.index(x_native[0], y_native[0])
            window = Window(col - 2, row - 2, 5, 5)
            nir = nir_ds.read(1, window=window).astype(float)
    except Exception as e:
        print(f"  {name}: band read failed — {e}")
        return None

    if red.size == 0 or nir.size == 0 or np.all(red == 0):
        print(f"  {name}: read returned empty/all-zero window (likely still off-target) — flagging, not faking a value")
        return {
            "district": name, "ndvi": None, "scene_date": date,
            "cloud_cover_pct": cloud_cover, "item_id": item.id, "note": "empty_window",
        }

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi_mean = float(np.nanmean(ndvi))
    print(f"  {name}: NDVI={ndvi_mean:.3f}  (scene {date}, cloud_cover={cloud_cover:.1f}%)")
    return {
        "district": name, "ndvi": ndvi_mean, "scene_date": date,
        "cloud_cover_pct": cloud_cover, "item_id": item.id,
    }


def main():
    catalog = pystac_client.Client.open(CATALOG_URL)
    rows = []
    for name, (lat, lon) in DISTRICTS.items():
        print(f"\n[{name}]")
        r = fetch_district(catalog, name, lat, lon)
        if r:
            rows.append(r)
        time.sleep(1)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} real NDVI snapshots to {OUTPUT_PATH}")
    if not df.empty:
        print(df.to_string())


if __name__ == "__main__":
    main()
