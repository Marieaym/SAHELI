"""
SAHELI Backend — Districts & Risk Map endpoints
All endpoints require authentication and are scoped to the user's own country.
"""
from fastapi import APIRouter, HTTPException, Depends
import sys, os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import get_scored_df, get_metrics, get_latest_snapshot, RISK_COLORS, assert_district_access
from routers.auth import get_current_user

router = APIRouter(prefix="/api", tags=["districts"])


def clean_record(d: dict) -> dict:
    """Replace NaN/inf with None so JSON serialization never fails."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        elif isinstance(v, pd.Timestamp):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


@router.get("/districts")
def list_districts(user: dict = Depends(get_current_user)):
    latest = get_latest_snapshot()
    latest = latest[latest["country"] == user["country"]]
    cols = ["district", "country", "zone", "lat", "lon", "predicted_risk",
            "drought_index", "consec_dry_days", "prob_low", "prob_medium", "prob_high", "prob_critical",
            "water_point_count_50km", "groundwater_anomaly_cm", "ipc_phase_observed",
            "sentinel2_ndvi", "sentinel2_scene_date", "sentinel2_cloud_cover_pct"]
    records = latest[cols].to_dict(orient="records")
    return {"count": len(records), "districts": [clean_record(r) for r in records], "risk_colors": RISK_COLORS, "scoped_to": user["country"]}


@router.get("/districts/{district_name}")
def district_detail(district_name: str, user: dict = Depends(get_current_user)):
    assert_district_access(district_name, user["country"])
    latest = get_latest_snapshot()
    row = latest[latest["district"] == district_name]
    return clean_record(row.iloc[0].to_dict())


@router.get("/districts/{district_name}/history")
def district_history(district_name: str, days: int = 365, user: dict = Depends(get_current_user)):
    assert_district_access(district_name, user["country"])
    df = get_scored_df()
    hist = df[df["district"] == district_name].sort_values("date").tail(days)
    cols = ["date", "precip_30d", "drought_index", "consec_dry_days", "predicted_risk"]
    records = hist[cols].to_dict(orient="records")
    return {"district": district_name, "count": len(records), "history": [clean_record(r) for r in records]}


@router.get("/zones/summary")
def zones_summary(user: dict = Depends(get_current_user)):
    latest = get_latest_snapshot()
    latest = latest[latest["country"] == user["country"]]
    summary = latest.groupby(["zone", "predicted_risk"]).size().reset_index(name="count")
    return {"summary": summary.to_dict(orient="records"), "scoped_to": user["country"]}


@router.get("/model/metrics")
def model_metrics():
    # Model performance metrics are not country-specific data; no auth required.
    return get_metrics()
