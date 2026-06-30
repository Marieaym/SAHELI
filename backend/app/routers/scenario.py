"""
SAHELI Backend — Scenario Simulator ("Digital Twin" lite)

Given a hypothetical rainfall change (e.g. "what if the monsoon arrives with
30% less rain this season?"), recomputes each district's climate features and
re-runs TWO real trained models on the adjusted inputs, explicitly, side by
side:

1. The original climate-shock model (XGBoost on risk_level): answers "how
   severe is the climate shock under this scenario".
2. The real food-security model, v2 (XGBoost trained directly on real FEWS
   NET IPC ground truth): answers "what does this scenario mean for real
   food security risk", the model that actually correlates with real
   ground truth (see models/food_security_v2_module.py).

Both are shown explicitly rather than collapsed into one number, because
they answer different real questions and SAHELI's own validation work
found they can disagree — that disagreement is informative, not noise.

Simplification, disclosed: only rainfall-driven features (precip_30d, precip_90d,
water_balance_30d, drought_index) are adjusted. Temperature, evapotranspiration,
dry-day streaks, seasonality, and real ACLED conflict intensity are held constant
at their currently observed values — this scenario simulates a rainfall shift
only, not a simultaneous conflict shift.
"""
from fastapi import APIRouter, Query, Depends
import numpy as np
import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import (
    get_scored_df, get_model_artifacts, get_latest_snapshot,
    get_v2_model_artifacts, v2_ipc_to_risk_level, V2_VALIDATED_DISTRICTS,
)
from routers.auth import get_current_user
from db import log_activity
from ai_client import call_ai

router = APIRouter(prefix="/api", tags=["scenario"])

LANG_INSTRUCTION = {
    "en": " Respond entirely in English.",
    "fr": " Réponds entièrement en français, dans un français professionnel et naturel, pas une traduction mot à mot.",
}

SCENARIO_SYSTEM_PROMPT = (
    "You are SAHELI's scenario briefing assistant. You explain a real climate "
    "scenario simulation's results to a government decision-maker or NGO field "
    "officer who may not read charts easily. You are given ONLY real, already-"
    "computed numbers from a real XGBoost model re-inference under a rainfall "
    "hypothesis — never invent or estimate a number not given to you. Write "
    "3-5 short sentences: what was simulated, what changed and where (name the "
    "most affected real districts given to you), and one practical takeaway. "
    "Plain language, no jargon, no markdown headers, conversational but precise."
)

FEATURE_COLS = [
    "precip_30d", "precip_90d", "et_30d", "temp_30d_avg",
    "water_balance_30d", "drought_index", "consec_dry_days",
    "month", "monsoon_season", "lat", "lon",
    "conflict_events_30d", "conflict_fatalities_30d", "price_anomaly_30d",
    "groundwater_anomaly_cm", "water_point_count_50km", "sentinel2_ndvi",
]


@router.get("/scenario/simulate")
def simulate_scenario(rainfall_delta_pct: float = Query(0, ge=-80, le=80), lang: str = Query("en"), user: dict = Depends(get_current_user)):
    log_activity(user["id"], "scenario_simulation")
    """rainfall_delta_pct: e.g. -30 means 30% less rainfall than currently observed."""
    df_all = get_scored_df()
    model, le, _ = get_model_artifacts()
    v2_model, v2_features = get_v2_model_artifacts()
    latest = get_latest_snapshot().copy()
    latest = latest[latest["country"] == user["country"]]

    std_wb = df_all["water_balance_30d"].std()
    mult = 1 + (rainfall_delta_pct / 100.0)

    rows_adjusted = []
    for _, row in latest.iterrows():
        precip_30d_adj = row["precip_30d"] * mult
        precip_90d_adj = row["precip_90d"] * mult
        water_balance_adj = precip_30d_adj - row["et_30d"]
        drought_index_adj = row["drought_index"] + (water_balance_adj - row["water_balance_30d"]) / std_wb

        feat = {
            "precip_30d": precip_30d_adj,
            "precip_90d": precip_90d_adj,
            "et_30d": row["et_30d"],
            "temp_30d_avg": row["temp_30d_avg"],
            "water_balance_30d": water_balance_adj,
            "drought_index": drought_index_adj,
            "consec_dry_days": row["consec_dry_days"],
            "month": row["month"],
            "monsoon_season": row["monsoon_season"],
            "lat": row["lat"],
            "lon": row["lon"],
            # Held constant: this scenario simulates a rainfall shift only,
            # not a conflict shift. Real observed conflict intensity for
            # this district stays as-is under the hypothetical.
            "conflict_events_30d": row.get("conflict_events_30d", 0),
            "conflict_fatalities_30d": row.get("conflict_fatalities_30d", 0),
            "price_anomaly_30d": row.get("price_anomaly_30d", 0),
            "groundwater_anomaly_cm": row.get("groundwater_anomaly_cm", 0),
            "water_point_count_50km": row.get("water_point_count_50km", 0),
            "sentinel2_ndvi": row.get("sentinel2_ndvi", 0),
        }
        rows_adjusted.append((row["district"], row["country"], row["zone"], row["predicted_risk"], feat))

    X_adj = pd.DataFrame([r[4] for r in rows_adjusted])[FEATURE_COLS]
    probs = model.predict_proba(X_adj)
    preds = le.inverse_transform(np.argmax(probs, axis=1))

    # The real food-security model (v2), run explicitly alongside the
    # original on the SAME adjusted scenario features, not instead of it.
    v2_current_ipc = {}
    v2_projected_ipc = {}
    if v2_model is not None:
        X_adj_v2 = pd.DataFrame([r[4] for r in rows_adjusted])[v2_features]
        v2_projected_ipc = dict(zip(
            [r[0] for r in rows_adjusted],
            np.clip(v2_model.predict(X_adj_v2), 1.0, 5.0)
        ))
        # Current (unadjusted) v2 prediction, for an honest before/after on the SAME model.
        X_current_v2 = latest[v2_features]
        v2_current_ipc = dict(zip(latest["district"], np.clip(v2_model.predict(X_current_v2), 1.0, 5.0)))

    results = []
    for i, (district, country, zone, current_risk, feat) in enumerate(rows_adjusted):
        entry = {
            "district": district,
            "country": country,
            "zone": zone,
            "current_risk": current_risk,
            "projected_risk": preds[i],
            "projected_drought_index": round(float(feat["drought_index"]), 3),
        }
        if v2_model is not None:
            cur_ipc = float(v2_current_ipc.get(district, np.nan))
            proj_ipc = float(v2_projected_ipc.get(district, np.nan))
            entry["food_security_current_risk"] = v2_ipc_to_risk_level(cur_ipc)
            entry["food_security_projected_risk"] = v2_ipc_to_risk_level(proj_ipc)
            entry["food_security_current_ipc"] = round(cur_ipc, 3)
            entry["food_security_projected_ipc"] = round(proj_ipc, 3)
            entry["food_security_ground_truth_status"] = (
                "validated" if district in V2_VALIDATED_DISTRICTS else "extrapolated"
            )
        results.append(entry)

    n_critical_current = sum(1 for r in results if r["current_risk"] == "Critical")
    n_critical_projected = sum(1 for r in results if r["projected_risk"] == "Critical")
    n_fs_critical_current = sum(1 for r in results if r.get("food_security_current_risk") == "Critical")
    n_fs_critical_projected = sum(1 for r in results if r.get("food_security_projected_risk") == "Critical")

    severity_order = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    newly_critical = [r["district"] for r in results
                       if r["current_risk"] != "Critical" and r["projected_risk"] == "Critical"]
    most_worsened = sorted(
        results, key=lambda r: severity_order[r["projected_risk"]] - severity_order[r["current_risk"]], reverse=True
    )[:5]
    most_worsened_names = [r["district"] for r in most_worsened
                            if severity_order[r["projected_risk"]] > severity_order[r["current_risk"]]]

    fs_newly_critical = [r["district"] for r in results
                          if v2_model is not None and r.get("food_security_current_risk") != "Critical"
                          and r.get("food_security_projected_risk") == "Critical"]

    lang = lang if lang in ("en", "fr") else "en"
    user_prompt = (
        f"Scenario: rainfall changed by {rainfall_delta_pct:+.0f}% versus currently observed conditions, "
        f"across the {len(results)} districts in {user['country']}.\n"
        f"TWO real models are run explicitly, side by side, because SAHELI's own validation found they "
        f"can disagree and that disagreement matters:\n"
        f"Model 1 (climate-shock severity): Critical-risk districts before: {n_critical_current}, after: {n_critical_projected}. "
        f"Newly Critical: {', '.join(newly_critical) if newly_critical else 'none'}.\n"
        + (
            f"Model 2 (real food-security risk, validated against real FEWS NET ground truth): "
            f"Critical-risk districts before: {n_fs_critical_current}, after: {n_fs_critical_projected}. "
            f"Newly Critical: {', '.join(fs_newly_critical) if fs_newly_critical else 'none'}.\n"
            if v2_model is not None else ""
        )
        + "Mention both models explicitly and briefly note if they agree or disagree on the same districts."
        + LANG_INSTRUCTION[lang]
    )
    ai_result = call_ai(SCENARIO_SYSTEM_PROMPT, user_prompt, max_tokens=300)
    narrative = ai_result["text"] or _template_scenario_narrative(
        rainfall_delta_pct, n_critical_current, n_critical_projected, newly_critical, most_worsened_names, lang,
        n_fs_critical_current if v2_model is not None else None,
        n_fs_critical_projected if v2_model is not None else None,
    )

    return {
        "rainfall_delta_pct": rainfall_delta_pct,
        "n_critical_current": n_critical_current,
        "n_critical_projected": n_critical_projected,
        "n_fs_critical_current": n_fs_critical_current if v2_model is not None else None,
        "n_fs_critical_projected": n_fs_critical_projected if v2_model is not None else None,
        "v2_available": v2_model is not None,
        "districts": results,
        "method": "Two real models run explicitly: the original XGBoost climate-shock model on rainfall-adjusted features, and the real FEWS-NET-validated food-security model (v2), both re-inferred under the same scenario.",
        "ai_narrative": narrative,
        "ai_mode": ai_result["mode"],
    }


def _template_scenario_narrative(delta, n_before, n_after, newly_critical, most_worsened, lang, n_fs_before=None, n_fs_after=None):
    """Real, non-AI fallback narrative built from the same real numbers,
    used when no OpenAI key is configured or the live call fails — the
    page should never show a broken or empty explanation."""
    change = n_after - n_before
    if lang == "fr":
        direction = "une baisse" if delta < 0 else "une hausse" if delta > 0 else "aucun changement"
        base = (
            f"Ce scénario simule {direction} des précipitations de {abs(delta):.0f}% par rapport aux conditions "
            f"actuelles. Risque de choc climatique : districts Critiques passent de {n_before} à {n_after}"
            f"{' (+' + str(change) + ')' if change > 0 else ' (' + str(change) + ')' if change < 0 else ''}."
        )
        if n_fs_before is not None:
            fs_change = n_fs_after - n_fs_before
            base += (
                f" Risque réel de sécurité alimentaire (validé contre FEWS NET) : districts Critiques passent de "
                f"{n_fs_before} à {n_fs_after}{' (+' + str(fs_change) + ')' if fs_change > 0 else ' (' + str(fs_change) + ')' if fs_change < 0 else ''}."
            )
        if newly_critical:
            base += f" Districts qui basculeraient en Critique (choc climatique) : {', '.join(newly_critical)}."
        return base
    direction = "a decrease" if delta < 0 else "an increase" if delta > 0 else "no change"
    base = (
        f"This scenario simulates {direction} in rainfall of {abs(delta):.0f}% versus current conditions. "
        f"Climate-shock risk: Critical districts go from {n_before} to {n_after}"
        f"{' (+' + str(change) + ')' if change > 0 else ' (' + str(change) + ')' if change < 0 else ''}."
    )
    if n_fs_before is not None:
        fs_change = n_fs_after - n_fs_before
        base += (
            f" Real food-security risk (validated against FEWS NET): Critical districts go from "
            f"{n_fs_before} to {n_fs_after}{' (+' + str(fs_change) + ')' if fs_change > 0 else ' (' + str(fs_change) + ')' if fs_change < 0 else ''}."
        )
    if newly_critical:
        base += f" Districts that would newly become Critical (climate shock): {', '.join(newly_critical)}."
    return base
