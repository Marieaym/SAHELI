"""
SAHELI — Real GRACE-FO groundwater data ingestion (essay Layer 1).

NASA Earthdata registration is instant and free (unlike FAO Locust's
manual-approval community account):
  1. Register at https://urs.earthdata.nasa.gov (2 minutes, no waiting).
  2. Log in, go to your profile -> "Generate Token" -> copy the token.
  3. Paste it below as EDL_TOKEN, or set it as an environment variable.

This script uses NASA's CMR (Common Metadata Repository) search API —
the standard, documented, free way to discover the current GRACE-FO
mascon granule without guessing a URL that might change between
processing releases — then downloads it with your Earthdata token via
HTTP Bearer auth (NASA's modern, recommended access pattern; no .netrc
file needed).

Run this on a machine with normal internet access — this sandbox's
network is restricted to pypi/npm/github and cannot reach nasa.gov.
"""
import requests
import os

EDL_TOKEN = os.environ.get("EDL_TOKEN", "PASTE_YOUR_EARTHDATA_TOKEN_HERE")

OUTPUT_DIR = os.path.dirname(__file__)
CMR_SEARCH = "https://cmr.earthdata.nasa.gov/search/granules.json"

# JPL GRACE/GRACE-FO Mascon, CRI-filtered, Release 06.3 — the recommended
# product for non-expert hydrology use per NASA's own guidance.
SHORT_NAME = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"


def find_latest_granule():
    params = {"short_name": SHORT_NAME, "sort_key": "-start_date", "page_size": 1}
    r = requests.get(CMR_SEARCH, params=params, timeout=30)
    r.raise_for_status()
    entries = r.json().get("feed", {}).get("entry", [])
    if not entries:
        print("No granules found for", SHORT_NAME)
        return None
    entry = entries[0]
    for link in entry.get("links", []):
        href = link.get("href", "")
        if href.endswith(".nc"):
            return href
    print("Granule found but no direct .nc download link:", entry.get("title"))
    return None


def download(url):
    if EDL_TOKEN == "PASTE_YOUR_EARTHDATA_TOKEN_HERE":
        print("Set EDL_TOKEN (env var or edit this file) with your NASA Earthdata token first.")
        print("Get one at https://urs.earthdata.nasa.gov -> profile -> Generate Token")
        return None
    headers = {"Authorization": f"Bearer {EDL_TOKEN}"}
    local_path = os.path.join(OUTPUT_DIR, os.path.basename(url))
    print(f"Downloading {url} ...")
    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Saved to {local_path} ({os.path.getsize(local_path)/1e6:.1f} MB)")
    return local_path


def main():
    print("Searching NASA CMR for the latest GRACE-FO mascon granule...")
    url = find_latest_granule()
    if not url:
        return
    print("Found:", url)
    download(url)


if __name__ == "__main__":
    main()
