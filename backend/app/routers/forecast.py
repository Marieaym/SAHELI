"""
SAHELI Backend — Agent layer: live multi horizon forecasting.

Loads the real weights trained by models/tft_lite_module.py (a temporal
self attention encoder) and produces real 4, 8, and 12 week ahead
drought_index forecasts for any district, using that district's actual
most recent 12 weeks of data as the lookback window. Same offline train,
online serve pattern as the anomaly and main risk modules.
"""
import os
import sys
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import get_scored_df, assert_district_access, DATA_DIR, get_tft_weights
from routers.auth import get_current_user

router = APIRouter(prefix="/api", tags=["forecast"])

WEEKLY_FEATURES = ["drought_index", "water_balance_30d", "sentinel2_ndvi",
                    "price_anomaly_30d", "conflict_events_30d"]
HORIZONS = [4, 8, 12]


def _relu(x):
    return np.maximum(0, x)


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _forward(X, w):
    """X: (1, 12, 5) already standardized. Mirrors
    TemporalAttentionForecaster.forward in models/tft_lite_module.py,
    using the persisted weights instead of a live training object."""
    n_heads = int(w["n_heads"])
    d_model = int(w["d_model"])
    d_head = d_model // n_heads
    B, T, F = X.shape

    embed = _relu(X @ w["W_embed"] + w["b_embed"])
    Q = embed @ w["W_q"]
    K = embed @ w["W_k"]
    V = embed @ w["W_v"]

    Qh = Q.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
    Kh = K.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
    Vh = V.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)

    scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(d_head)
    attn = _softmax(scores, axis=-1)
    head_out = attn @ Vh
    attn_out = head_out.transpose(0, 2, 1, 3).reshape(B, T, n_heads * d_head)
    residual = embed + attn_out @ w["W_o"]
    pooled = residual.mean(axis=1)
    ctx = _relu(pooled @ w["W_ctx"] + w["b_ctx"])
    preds = ctx @ w["W_heads"].T + w["b_heads"]
    return preds


@router.get("/forecast/{district_name}")
def forecast_district(district_name: str, user: dict = Depends(get_current_user)):
    assert_district_access(district_name, user["country"])
    w = get_tft_weights()
    if w is None:
        raise HTTPException(status_code=503, detail="Forecast model not found. Run models/tft_lite_module.py first.")

    df = get_scored_df()
    dist_df = df[df["district"] == district_name].copy()
    if dist_df.empty:
        raise HTTPException(status_code=404, detail=f"District '{district_name}' not found")

    dist_df["week"] = dist_df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    weekly = dist_df.groupby("week")[WEEKLY_FEATURES].mean().reset_index().sort_values("week")
    lookback_weeks = int(w["lookback_weeks"])
    if len(weekly) < lookback_weeks:
        raise HTTPException(status_code=422, detail="Not enough history for this district to forecast.")

    window = weekly[WEEKLY_FEATURES].values[-lookback_weeks:]
    last_value = float(window[-1, WEEKLY_FEATURES.index("drought_index")])
    last_week = str(weekly["week"].iloc[-1].date())

    X = ((window - w["feat_mean"]) / w["feat_std"])[None, :, :]
    delta_preds_s = _forward(X, w)[0]
    delta_preds = delta_preds_s * float(w["delta_std"]) + float(w["delta_mean"])
    forecasts = (delta_preds + last_value).tolist()

    return {
        "district": district_name,
        "as_of_week": last_week,
        "current_drought_index": round(last_value, 4),
        "forecasts": [
            {"horizon_weeks": h, "forecast_drought_index": round(f, 4),
             "change_from_current": round(f - last_value, 4)}
            for h, f in zip(HORIZONS, forecasts)
        ],
        "note": (
            "Real temporal attention forecast (see models/tft_lite_module.py). "
            "Validated on 2023-2024 held out data: ties naive persistence at 4 "
            "weeks, beats it by 8.1% at 8 weeks and 21.2% at 12 weeks (mean "
            "absolute error). More negative forecast values indicate worsening "
            "drought conditions relative to today."
        ),
    }
