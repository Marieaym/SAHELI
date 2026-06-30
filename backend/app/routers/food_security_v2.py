"""
SAHELI Backend — the real food-security-targeted model (v2), live.

Serves predictions from models/food_security_v2_module.py: a model
trained directly on real FEWS NET IPC ground truth (10 districts,
32,443 real observations), not the original rule-derived climate proxy.
For the 8 districts without real ground truth, the same trained model
is still applied (it only needs real climate/conflict/price/groundwater/
NDVI features, all available for all 18 districts) but every response
honestly flags whether THIS district was part of the real validated set
or is an extrapolation.
"""
import json
import os
import sys
import numpy as np
import xgboost as xgb
import joblib
from functools import lru_cache
from fastapi import APIRouter, Depends, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import get_scored_df, get_latest_snapshot, assert_district_access, DATA_DIR
from routers.auth import get_current_user

router = APIRouter(prefix="/api", tags=["food_security_v2"])

_MODEL_PATH = os.path.join(DATA_DIR, "food_security_v2_model.json")
_FEATURES_PATH = os.path.join(DATA_DIR, "food_security_v2_features.joblib")
_RESULTS_PATH = os.path.join(DATA_DIR, "food_security_v2_results.json")

_VALIDATED_DISTRICTS = {"Agadez", "Bamako", "Diffa", "Gao", "Maradi", "Mopti",
                         "Niamey", "Nouakchott", "Tahoua", "Zinder"}


def _to_risk_level(ipc_value):
    if ipc_value >= 2.5:
        return "Critical"
    if ipc_value >= 2.0:
        return "High"
    if ipc_value >= 1.5:
        return "Medium"
    return "Low"


@lru_cache()
def _load_model():
    """Cached: same bug, same fix as anomaly.py and forecast.py. This
    router's own dedicated /api/food-security-v2/{district} endpoint
    has its own model loader, separate from data_access.py's already-
    cached get_v2_model_artifacts(), and it was missing this fix too."""
    if not (os.path.exists(_MODEL_PATH) and os.path.exists(_FEATURES_PATH)):
        return None, None
    model = xgb.XGBRegressor()
    model.load_model(_MODEL_PATH)
    features = joblib.load(_FEATURES_PATH)
    return model, features


@router.get("/food-security-v2/summary")
def v2_summary(user: dict = Depends(get_current_user)):
    """The honest, aggregate validation findings from the offline run:
    real correlation with FEWS NET ground truth, versus both baselines."""
    if not os.path.exists(_RESULTS_PATH):
        raise HTTPException(status_code=503, detail="food_security_v2 model has not been run yet.")
    with open(_RESULTS_PATH) as f:
        results = json.load(f)
    results.pop("per_district_breakdown", None)
    return results


@router.get("/food-security-v2/{district_name}")
def v2_for_district(district_name: str, user: dict = Depends(get_current_user)):
    assert_district_access(district_name, user["country"])
    model, features = _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Run models/food_security_v2_module.py first.")

    latest = get_latest_snapshot()
    row = latest[latest["district"] == district_name]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"District '{district_name}' not found")
    row = row.iloc[0]

    x = row[features].values.astype(float).reshape(1, -1)
    pred_ipc = float(np.clip(model.predict(x)[0], 1.0, 5.0))

    is_validated = district_name in _VALIDATED_DISTRICTS

    return {
        "district": district_name,
        "predicted_ipc_phase": round(pred_ipc, 3),
        "predicted_risk_level": _to_risk_level(pred_ipc),
        "original_model_risk_level": row.get("predicted_risk", None),
        "ground_truth_status": "validated" if is_validated else "extrapolated",
        "note": (
            "This district has real FEWS NET ground truth in SAHELI's training and "
            "test data; this prediction was validated against real, held-out IPC "
            "observations (see /api/food-security-v2/summary)."
            if is_validated else
            "This district has no real FEWS NET ground truth in SAHELI's data yet. "
            "This prediction uses the same model trained on 10 other real Sahelian "
            "districts, applied here by extrapolation, not locally validated."
        ),
    }
