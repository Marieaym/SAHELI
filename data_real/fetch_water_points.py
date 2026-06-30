"""
SAHELI — Real water-point mapping via OpenStreetMap Overpass API
(essay's pastoralism module, water-point availability tracking).

Overpass API is free, public, and requires no registration. This
script queries real OSM-tagged water features (wells, water points,
waterholes, reservoirs) within a 50km radius of each of SAHELI's 18
districts, and saves real counts + locations: a genuine, if
crowdsourced-coverage-dependent, proxy for pastoral water access —
exactly what the essay's pastoralism module describes tracking.

Honest limitation, disclosed upfront: OSM coverage in rural Sahel is
uneven — a low count near a district may mean "few real water points"
OR "this area is undermapped on OSM", not necessarily the same thing.
This is reported as a real, current limitation, not glossed over.

Run this on a machine with normal internet access — this sandbox's
network is restricted to pypi/npm/github and cannot reach overpass-api.de.
"""
import requests
import pandas as pd
import os
import time

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "water_points.csv")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DISTRICTS = {
    "Niamey": (13.51, 2.11), "Zinder": (13.80, 8.99), "Maradi": (13.50, 7.10),
    "Tahoua": (14.88, 5.27), "Agadez": (16.97, 7.99), "Diffa": (13.31, 12.61),
    "Bamako": (12.65, -8.00), "Mopti": (14.49, -4.19), "Timbuktu": (16.77, -3.00),
    "Gao": (16.27, -0.04), "Ouagadougou": (12.36, -1.53), "Dori": (14.03, -0.03),
    "Djibo": (14.10, -1.63), "NDjamena": (12.10, 15.04), "Abeche": (13.83, 20.83),
    "Nouakchott": (18.08, -15.97), "Kiffa": (16.62, -11.40), "Matam": (15.65, -13.25),
}

RADIUS_M = 50_000  # 50km — district-level catchment, not the city point itself

QUERY_TEMPLATE = """
[out:json][timeout:60];
(
  node["man_made"="water_well"](around:{radius},{lat},{lon});
  node["amenity"="watering_place"](around:{radius},{lat},{lon});
  node["water"]["water"!="lake"](around:{radius},{lat},{lon});
  way["natural"="water"](around:{radius},{lat},{lon});
  node["man_made"="water_tower"](around:{radius},{lat},{lon});
  node["man_made"="reservoir_covered"](around:{radius},{lat},{lon});
);
out center;
"""


def fetch_district(name, lat, lon):
    query = QUERY_TEMPLATE.format(radius=RADIUS_M, lat=lat, lon=lon)
    headers = {
        "User-Agent": "SAHELI-research-script/1.0 (educational, low-volume, Sahel food-security project)",
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=90)
        if r.status_code != 200:
            print(f"  {name}: HTTP {r.status_code} — {r.text[:300]}")
            return []
        elements = r.json().get("elements", [])
    except Exception as e:
        print(f"  {name}: query failed — {e}")
        return []
    print(f"  {name}: {len(elements)} real OSM water features found within 50km")
    rows = []
    for el in elements:
        lat_e = el.get("lat") or el.get("center", {}).get("lat")
        lon_e = el.get("lon") or el.get("center", {}).get("lon")
        rows.append({
            "district": name, "osm_id": el.get("id"), "osm_type": el.get("type"),
            "lat": lat_e, "lon": lon_e, "tags": el.get("tags", {}),
        })
    return rows


def main():
    all_rows = []
    for name, (lat, lon) in DISTRICTS.items():
        rows = fetch_district(name, lat, lon)
        all_rows.extend(rows)
        time.sleep(2)  # be polite to the free public instance

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} real water features to {OUTPUT_PATH}")
    if not df.empty:
        print(df.groupby("district").size().sort_values(ascending=False))


if __name__ == "__main__":
    main()
