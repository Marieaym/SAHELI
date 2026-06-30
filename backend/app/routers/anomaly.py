"""
SAHELI Backend — Agent layer: live anomaly detection.

Loads the real artifacts produced by models/anomaly_module.py (a fitted
StandardScaler, a fitted IsolationForest, and the weights of a hand
trained NumPy autoencoder) and scores any district's latest real data
row on demand. This is the live serving half of the same offline train,
online serve pattern already used for the main XGBoost risk model.

Honest framing, carried over from the research script: an undirected
anomaly flag is not by itself a reliable Critical risk signal in this
drought prone dataset (see anomaly_results.json). What this endpoint
reports is the directional read: whether a flagged day's deviation
points toward worsening conditions (adverse) or improving conditions
(favorable), which is the distinction that actually matters operationally.
"""
import json
import os
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
import joblib
import sys
from functools import lru_cache

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import get_latest_snapshot, assert_district_access, DATA_DIR
from routers.auth import get_current_user

router = APIRouter(prefix="/api", tags=["anomaly"])

ANOMALY_FEATURES = ["drought_index", "consec_dry_days", "water_balance_30d", "price_anomaly_30d",
                     "conflict_events_30d", "groundwater_anomaly_cm", "sentinel2_ndvi"]

ADVERSE_DIRECTION = {
    "drought_index": -1, "water_balance_30d": -1, "sentinel2_ndvi": -1,
    "groundwater_anomaly_cm": -1, "consec_dry_days": 1,
    "price_anomaly_30d": 1, "conflict_events_30d": 1,
}

_SCALER_PATH = os.path.join(DATA_DIR, "anomaly_scaler.joblib")
_ISO_PATH = os.path.join(DATA_DIR, "anomaly_isoforest.joblib")
_AE_PATH = os.path.join(DATA_DIR, "anomaly_autoencoder_weights.npz")
_RESULTS_PATH = os.path.join(DATA_DIR, "anomaly_results.json")


def _relu(x):
    return np.maximum(0, x)


def _autoencoder_error(x_scaled_row, weights):
    z1 = x_scaled_row @ weights["W1"] + weights["b1"]
    a1 = _relu(z1)
    z2 = a1 @ weights["W2"] + weights["b2"]
    z3 = z2 @ weights["W3"] + weights["b3"]
    a3 = _relu(z3)
    z4 = a3 @ weights["W4"] + weights["b4"]
    return float(((z4 - x_scaled_row) ** 2).mean())


@lru_cache()
def _load_artifacts():
    """Cached: this IsolationForest and scaler were being reloaded and
    re-unpickled from disk on every single request before this fix,
    which is genuinely slow (confirmed: real user report of slow load
    times on Command Center, which calls this once per district). Now
    loaded from disk exactly once per server process."""
    if not all(os.path.exists(p) for p in (_SCALER_PATH, _ISO_PATH, _AE_PATH)):
        return None
    scaler = joblib.load(_SCALER_PATH)
    iso = joblib.load(_ISO_PATH)
    weights = np.load(_AE_PATH)
    return scaler, iso, weights


@router.get("/anomaly/summary")
def anomaly_summary(user: dict = Depends(get_current_user)):
    """The honest, aggregate research findings from the offline run:
    pooled lift, adverse vs favorable direction split, and the
    independent 2021-2022 crisis window cross check."""
    if not os.path.exists(_RESULTS_PATH):
        raise HTTPException(status_code=503, detail="Anomaly module has not been run yet.")
    with open(_RESULTS_PATH) as f:
        results = json.load(f)
    results.pop("top_10_examples", None)
    return results


@router.get("/anomaly/{district_name}")
def anomaly_for_district(district_name: str, user: dict = Depends(get_current_user)):
    """Score the district's latest real row live, using the saved
    scaler, isolation forest, and autoencoder. Returns whether the
    current day looks statistically unusual, and if so, in which
    direction."""
    assert_district_access(district_name, user["country"])
    artifacts = _load_artifacts()
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Anomaly model artifacts not found. Run models/anomaly_module.py first.")
    scaler, iso, weights = artifacts

    latest = get_latest_snapshot()
    row = latest[latest["district"] == district_name]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"District '{district_name}' not found")
    row = row.iloc[0]

    x = row[ANOMALY_FEATURES].values.astype(float).reshape(1, -1)
    x_scaled = scaler.transform(x)

    iso_flag = bool(iso.predict(x_scaled)[0] == -1)
    iso_score = float(iso.decision_function(x_scaled)[0])
    ae_error = _autoencoder_error(x_scaled[0], weights)
    ae_threshold = float(weights["ae_threshold"])
    ae_flag = bool(ae_error >= ae_threshold)

    both_flag = iso_flag and ae_flag
    adverse_vote = 0
    if both_flag:
        for feat, direction in ADVERSE_DIRECTION.items():
            val = row[feat]
            # Sign relative to this district's own scaled value (already
            # centered/scaled by the same scaler fitted on the full
            # historical dataset, so 0 is the historical mean).
            idx = ANOMALY_FEATURES.index(feat)
            if direction * x_scaled[0][idx] > 0:
                adverse_vote += 1
    direction_label = "none"
    if both_flag:
        direction_label = "adverse" if adverse_vote >= 4 else "favorable"

    return {
        "district": district_name,
        "date": str(row["date"]),
        "is_anomalous": both_flag,
        "direction": direction_label,
        "isolation_forest_flag": iso_flag,
        "autoencoder_flag": ae_flag,
        "autoencoder_reconstruction_error": round(ae_error, 5),
        "autoencoder_threshold": round(ae_threshold, 5),
        "note": (
            "Adverse direction anomalies are the operationally meaningful signal "
            "(2.27x the baseline Critical rate in the offline validation). Favorable "
            "direction anomalies are statistically unusual but not a warning sign."
            if both_flag else
            "No statistically unusual deviation detected across the 7 monitored features today."
        ),
    }
