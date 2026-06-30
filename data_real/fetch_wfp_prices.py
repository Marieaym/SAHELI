"""
SAHELI — Real WFP food price ingestion via HDX (Humanitarian Data Exchange).

HDX is built on CKAN, which exposes a public, unauthenticated REST API.
Each of our 6 countries has its own maintained WFP food price dataset on
HDX, named "wfp-food-prices-for-{country}". This script:
  1. Queries the CKAN API for each country's dataset metadata.
  2. Downloads the actual CSV resource (real WFP VAM price records).
  3. Filters to staple cereals (the most food-security-relevant commodities)
     and saves one combined CSV: data/wfp_food_prices.csv

Run this on a machine with normal internet access — this sandbox's
network is restricted to pypi/npm/github and cannot reach humdata.org.
"""
import requests
import pandas as pd
import os
import time

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "wfp_food_prices.csv")

COUNTRY_SLUGS = {
    "Niger": "wfp-food-prices-for-niger",
    "Mali": "wfp-food-prices-for-mali",
    "Burkina Faso": "wfp-food-prices-for-burkina-faso",
    "Chad": "wfp-food-prices-for-chad",
    "Mauritania": "wfp-food-prices-for-mauritania",
    "Senegal": "wfp-food-prices-for-senegal",
}

CKAN_API = "https://data.humdata.org/api/3/action/package_show"

STAPLE_KEYWORDS = ["millet", "sorghum", "maize", "rice", "cowpea", "wheat"]


def fetch_country(country, slug):
    print(f"\n[{country}] querying {slug}...")
    try:
        r = requests.get(CKAN_API, params={"id": slug}, timeout=30)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  metadata request failed: {e}")
        return pd.DataFrame()

    if not payload.get("success"):
        print(f"  CKAN returned success=false: {payload.get('error')}")
        return pd.DataFrame()

    resources = payload["result"]["resources"]
    csv_resources = [r for r in resources if r.get("format", "").upper() == "CSV"]
    if not csv_resources:
        print(f"  no CSV resource found among {len(resources)} resources")
        return pd.DataFrame()

    resource_url = csv_resources[0]["url"]
    print(f"  downloading {resource_url}")
    try:
        df = pd.read_csv(resource_url, low_memory=False)
    except Exception as e:
        print(f"  download/parse failed: {e}")
        return pd.DataFrame()

    print(f"  {len(df)} raw rows, columns: {df.columns.tolist()[:12]}...")
    df["sahel_country"] = country
    return df


def main():
    all_frames = []
    for country, slug in COUNTRY_SLUGS.items():
        df = fetch_country(country, slug)
        if not df.empty:
            all_frames.append(df)
        time.sleep(1)

    if not all_frames:
        print("\nNo data retrieved from any country. Saving empty file.")
        pd.DataFrame().to_csv(OUTPUT_PATH, index=False)
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # WFP HXL-tagged files have a header row of HXL hashtags as row 0 of data
    # (e.g. "#date","#adm1+name"...) — drop it if present.
    first_row_str = combined.iloc[0].astype(str).str.startswith("#")
    if first_row_str.any():
        combined = combined.iloc[1:].reset_index(drop=True)

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} total rows across {len(all_frames)} countries to {OUTPUT_PATH}")
    print(combined.head(5))


if __name__ == "__main__":
    main()
