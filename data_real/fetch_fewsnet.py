"""
SAHELI — Real FEWS NET acute food insecurity classification ingestion via HDX.

FEWS NET's own API (fdw.fews.net) was retired and its HDX-linked datasets
are now archived static files, but FEWS NET still publishes monthly IPC-
compatible classification data per country on HDX as CSV/JSON/GeoJSON,
under the "fewsnet" organization. This script:
  1. Searches HDX's public CKAN API for each country's "current situation"
     FEWS NET dataset.
  2. Downloads the CSV resource (district/admin-unit level IPC phase).
  3. Saves a combined CSV: data/fewsnet_classification.csv

Run this on a machine with normal internet access — this sandbox's
network is restricted to pypi/npm/github and cannot reach humdata.org.
"""
import requests
import pandas as pd
import os
import time

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "fewsnet_classification.csv")

CKAN_SEARCH = "https://data.humdata.org/api/3/action/package_search"
CKAN_SHOW = "https://data.humdata.org/api/3/action/package_show"

COUNTRIES = ["Niger", "Mali", "Burkina Faso", "Chad", "Mauritania", "Senegal"]


def find_dataset_slug(country):
    """Search HDX for this country's FEWS NET current-situation dataset."""
    query = f"{country} current situation fewsnet acute food insecurity"
    params = {"q": query, "fq": "organization:fewsnet", "rows": 5}
    try:
        r = requests.get(CKAN_SEARCH, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  search failed: {e}")
        return None

    results = payload.get("result", {}).get("results", [])
    print(f"  {len(results)} candidate datasets for {country}:")
    for res in results[:5]:
        print(f"    - {res.get('title')}  ({res.get('name')})")
    # Prefer one whose title mentions "Current Situation"
    for res in results:
        if "current situation" in res.get("title", "").lower():
            return res["name"]
    return results[0]["name"] if results else None


def fetch_country(country):
    print(f"\n[{country}]")
    slug = find_dataset_slug(country)
    if not slug:
        print("  no dataset slug found")
        return pd.DataFrame()

    try:
        r = requests.get(CKAN_SHOW, params={"id": slug}, timeout=30)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  package_show failed: {e}")
        return pd.DataFrame()

    if not payload.get("success"):
        print(f"  CKAN error: {payload.get('error')}")
        return pd.DataFrame()

    resources = payload["result"]["resources"]
    csv_resources = [r for r in resources if r.get("format", "").upper() == "CSV"]
    if not csv_resources:
        print(f"  no CSV resource among {len(resources)} resources")
        return pd.DataFrame()

    # Prefer the most recently modified CSV resource
    csv_resources.sort(key=lambda r: r.get("last_modified", ""), reverse=True)
    url = csv_resources[0]["url"]
    print(f"  downloading {url}")
    try:
        df = pd.read_csv(url, low_memory=False)
    except Exception as e:
        print(f"  download/parse failed: {e}")
        return pd.DataFrame()

    print(f"  {len(df)} rows, columns: {df.columns.tolist()[:12]}")
    df["sahel_country"] = country
    return df


def main():
    all_frames = []
    for country in COUNTRIES:
        df = fetch_country(country)
        if not df.empty:
            all_frames.append(df)
        time.sleep(1)

    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} total rows to {OUTPUT_PATH}")
    if not combined.empty:
        print(combined.head(5))


if __name__ == "__main__":
    main()
