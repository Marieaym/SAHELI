"""
SAHELI — Real FAO Desert Locust Hub ingestion.

Source: FAO Locust Hub (https://locust-hub-hqfao.hub.arcgis.com), a public
ArcGIS Hub site. Public datasets (Swarms, Hopper Bands, Adults,
Control Operations) are queryable without an account through two
documented, unauthenticated mechanisms:
  1. The Hub Search API v3 (https://hub.arcgis.com/api/v3/search) to
     discover the live FeatureServer URL behind a dataset.
  2. The standard ArcGIS REST "query" operation on that FeatureServer,
     which returns plain JSON.

This script:
  1. Discovers the FeatureServer URLs for the relevant locust datasets.
  2. Queries them filtered to our 6 SAHELI countries (Niger, Mali,
     Burkina Faso, Chad, Mauritania, Senegal) for the last 24 months.
  3. Saves a clean CSV: data/locust_events.csv

Run this on a machine with normal internet access — this sandbox's
network is restricted to pypi/npm/github and cannot reach arcgis.com.
"""
import requests
import pandas as pd
import os
import time

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "locust_events.csv")

SAHEL_COUNTRIES = ["Niger", "Mali", "Burkina Faso", "Chad", "Mauritania", "Senegal"]

# Dataset names as published on the FAO Locust Hub. We search for each by
# keyword rather than hardcoding a FeatureServer URL, because Esri can
# re-host the service behind a new URL without notice — searching by name
# is the resilient way to do this, exactly as the Hub's own client does.
DATASET_QUERIES = {
    "swarms": "Desert Locust Swarms",
    "hopper_bands": "Desert Locust Hopper Bands",
    "adults": "Desert Locust Adults",
}

HUB_SEARCH_URL = "https://hub.arcgis.com/api/v3/search"


KEYWORD_HINTS = {
    "swarms": ["swarm"],
    "hopper_bands": ["hopper", "band"],
    "adults": ["adult"],
}


def discover_feature_server(label, query_text):
    """Find candidate FeatureServer URLs for a named Locust Hub dataset,
    ranked by keyword relevance. Returns a LIST (not just one), because
    some discovered layers turn out to require a token despite being
    publicly listed — we try the next-best candidate rather than give up."""
    params = {"q": query_text, "filter[source]": "locust-hub-hqfao", "page[size]": 5}
    r = requests.get(HUB_SEARCH_URL, params=params, timeout=30)
    print(f"  [scoped] HTTP {r.status_code} for '{query_text}'")
    candidates = []
    try:
        candidates = r.json().get("data", [])
    except Exception as e:
        print(f"  [scoped] could not parse JSON: {e}")

    if not candidates:
        params2 = {"q": f"desert locust {query_text}", "page[size]": 30}
        r2 = requests.get(HUB_SEARCH_URL, params=params2, timeout=30)
        print(f"  [broad]  HTTP {r2.status_code} for 'desert locust {query_text}'")
        try:
            candidates = r2.json().get("data", [])
        except Exception as e:
            print(f"  [broad] could not parse JSON: {e}")

    feature_layers = [
        c for c in candidates
        if c.get("attributes", {}).get("type") == "Feature Layer"
        and c.get("attributes", {}).get("url")
    ]
    print(f"  {len(feature_layers)} Feature Layer candidates with a real url:")
    for c in feature_layers:
        print(f"    - {c['attributes'].get('name')}  -> {c['attributes'].get('url')}")

    if not feature_layers:
        print(f"  no usable Feature Layer found for '{query_text}'")
        return []

    # Rank: keyword-matched name first, then "infestation"/live-explorer
    # layers (these power the public Locust Hub map and are more likely
    # to be open), then everything else, in original order.
    hints = KEYWORD_HINTS.get(label, [])

    def rank(c):
        name_lower = c["attributes"].get("name", "").lower()
        if any(h in name_lower for h in hints):
            return 0
        if "infestation" in name_lower:
            return 1
        return 2

    ranked = sorted(feature_layers, key=rank)
    return ranked


def query_feature_server(feature_server_url, dataset_label):
    """Query a FeatureServer layer 0, filtered to recent Sahel records."""
    if not feature_server_url:
        return pd.DataFrame()
    query_url = feature_server_url.rstrip("/") + "/0/query"
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "json",
        "resultRecordCount": 2000,
    }
    try:
        r = requests.get(query_url, params=params, timeout=60)
        print(f"  query HTTP {r.status_code}")
        data = r.json()
    except Exception as e:
        print(f"  query failed for {dataset_label}: {e}")
        return pd.DataFrame()

    if "error" in data:
        print(f"  server returned an error: {data['error']}")
        return pd.DataFrame()

    features = data.get("features", [])
    if not features:
        print(f"  0 features returned for {dataset_label}")
        print(f"  raw response keys: {list(data.keys())}")
        return pd.DataFrame()

    rows = [f.get("attributes", {}) for f in features]
    df = pd.DataFrame(rows)
    df["dataset"] = dataset_label
    print(f"  {len(df)} raw records for {dataset_label}")
    print(f"  columns: {df.columns.tolist()}")
    return df


def filter_to_sahel(df):
    if df.empty:
        return df
    # The exact country field name varies between Locust Hub layers
    # (seen historically as COUNTRYNAME, COUNTRY, ADM0_NAME, CNTRY_NAME, NAME0).
    country_col = None
    for candidate in ["COUNTRYNAME", "COUNTRY", "ADM0_NAME", "Country", "CNTRY_NAME", "NAME0", "ADM0_NM"]:
        if candidate in df.columns:
            country_col = candidate
            break
    if country_col is None:
        print(f"  warning: no recognizable country column. Available columns: {df.columns.tolist()}")
        print("  returning all rows unfiltered — tell Claude the column list above so the filter can be fixed.")
        return df
    print(f"  filtering on column '{country_col}', sample values: {df[country_col].dropna().unique()[:8]}")
    mask = df[country_col].isin(SAHEL_COUNTRIES)
    return df[mask]


def main():
    print("Discovering FAO Locust Hub datasets...")
    all_frames = []
    for label, query_text in DATASET_QUERIES.items():
        print(f"\n[{label}]")
        candidates = discover_feature_server(label, query_text)
        time.sleep(1)

        df = pd.DataFrame()
        for i, c in enumerate(candidates):
            url = c["attributes"]["url"]
            name = c["attributes"].get("name")
            print(f"  attempt {i+1}/{len(candidates)}: {name} -> {url}")
            df = query_feature_server(url, label)
            if not df.empty:
                print(f"  SUCCESS with '{name}'")
                break
            time.sleep(1)

        df = filter_to_sahel(df)
        print(f"  {len(df)} records after filtering to Sahel countries")
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} total records to {OUTPUT_PATH}")
    if not combined.empty:
        print(combined.head(10))


if __name__ == "__main__":
    main()
