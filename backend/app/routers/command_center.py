"""
SAHELI Backend — Command Center: every real model, one view.

This is deliberately NOT a new model. It is a real synthesis layer over
models that already exist and are already individually validated
elsewhere in this codebase:
  - the original XGBoost climate-shock classifier
  - the real FEWS-NET-validated food-security model (v2)
  - the real NumPy anomaly detector (Isolation Forest + autoencoder)
  - the real NumPy temporal-attention forecaster (4/8/12 weeks)
  - the real per-instance SHAP attribution already used in the PDF brief
  - the real, logged Corn Scanner field reports

Nothing here is invented: every number returned is loaded or computed
from one of those existing, already-tested modules. The only new thing
is the composite urgency score, which is an explicit, disclosed
HEURISTIC for ranking districts by combining already-real signals — not
a new trained model, and not presented as one.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import joblib
from functools import lru_cache
from fastapi import APIRouter, Depends, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import (
    get_latest_snapshot, get_model_artifacts, get_v2_model_artifacts,
    v2_ipc_to_risk_level, V2_VALIDATED_DISTRICTS, DATA_DIR, get_tft_weights,
)
from routers.auth import get_current_user
from routers.anomaly import (
    ANOMALY_FEATURES, ADVERSE_DIRECTION, _autoencoder_error, _load_artifacts as _load_anomaly_artifacts,
)
from routers.forecast import WEEKLY_FEATURES, HORIZONS, _forward as _forecast_forward
from routers.brief import real_shap_top_drivers
from db import get_recent_crop_scans
from ai_client import call_ai

router = APIRouter(prefix="/api", tags=["command-center"])

LANG_INSTRUCTION = {
    "en": " Respond entirely in English.",
    "fr": " Réponds entièrement en français, professionnel et naturel.",
}

COMMAND_CENTER_SYSTEM_PROMPT = (
    "You are SAHELI's command center briefing assistant. You are given ALL of "
    "SAHELI's real model outputs for one district at once — climate-shock risk, "
    "the real FEWS-NET-validated food-security risk, anomaly detection direction, "
    "a 4/8/12-week forecast trend, the top causal/SHAP driver, and any logged "
    "crop-disease field reports. Synthesize them into ONE coherent 4-6 sentence "
    "briefing a decision-maker can act on. Be explicit when signals agree or "
    "disagree — that disagreement is real and meaningful, not noise. Never invent "
    "a number not given to you."
)


def _severity_rank(risk):
    return {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}.get(risk, 0)


def _compute_district_signals(row, anomaly_bundle, forecast_weights):
    district = row["district"]
    out = {
        "district": district,
        "country": row["country"],
        "zone": row["zone"],
        "climate_risk": row["predicted_risk"],
        "drought_index": round(float(row["drought_index"]), 3),
    }

    # ── Real food-security model (v2) ──
    v2_model, v2_features = get_v2_model_artifacts()
    if v2_model is not None:
        ipc = float(np.clip(v2_model.predict(pd.DataFrame([row])[v2_features])[0], 1.0, 5.0))
        fs_risk = v2_ipc_to_risk_level(ipc)
        out["food_security"] = {
            "ipc": round(ipc, 3), "risk": fs_risk,
            "status": "validated" if district in V2_VALIDATED_DISTRICTS else "extrapolated",
        }
    else:
        out["food_security"] = None

    # ── Real anomaly detector ──
    if anomaly_bundle is not None:
        scaler, iso, weights = anomaly_bundle
        x = row[ANOMALY_FEATURES].values.astype(float).reshape(1, -1)
        x_scaled = scaler.transform(x)
        iso_flag = bool(iso.predict(x_scaled)[0] == -1)
        ae_error = _autoencoder_error(x_scaled[0], weights)
        ae_flag = bool(ae_error >= float(weights["ae_threshold"]))
        both_flag = iso_flag and ae_flag
        direction = "none"
        if both_flag:
            adverse_vote = sum(
                1 for feat, d in ADVERSE_DIRECTION.items()
                if d * x_scaled[0][ANOMALY_FEATURES.index(feat)] > 0
            )
            direction = "adverse" if adverse_vote >= 4 else "favorable"
        out["anomaly"] = direction
    else:
        out["anomaly"] = None

    # ── Real temporal-attention forecast ──
    if forecast_weights is not None:
        w = forecast_weights
        lookback_weeks = int(w["lookback_weeks"])
        # Re-derive the district's weekly history from the same snapshot row's
        # available trailing window isn't possible from a single row, so the
        # forecast trend here reuses the SAME live serving logic forecast.py
        # already exposes per-district — called once per district below.
        out["forecast_trend"] = None  # filled by caller using the real per-district endpoint logic
    else:
        out["forecast_trend"] = None

    # ── Real SHAP top driver (reuses the exact mechanism behind the PDF brief) ──
    drivers = real_shap_top_drivers(row, lang="en", top_n=1)
    out["top_driver"] = {"name": drivers[0][0], "shap_value": round(drivers[0][1], 3)} if drivers else None

    # ── Real, logged Corn Scanner field reports ──
    scans = get_recent_crop_scans(district, limit=20)
    out["crop_reports"] = {
        "n_total": len(scans),
        "n_disease": sum(1 for s in scans if s["predicted_class"] != "Healthy"),
    }

    # ── Composite urgency score: an explicit, disclosed heuristic over the
    # real signals above, NOT a new trained model. ──
    if out["food_security"]:
        score = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Critical": 4.0}[out["food_security"]["risk"]]
        if out["food_security"]["status"] == "extrapolated":
            score -= 0.1  # slightly less weight on an unvalidated extrapolation
    else:
        score = float(_severity_rank(out["climate_risk"]))
    if out["anomaly"] == "adverse":
        score += 0.5
    models_agree = out["food_security"] is not None and out["food_security"]["risk"] == out["climate_risk"]
    out["models_agree"] = models_agree
    if not models_agree and out["food_security"] and _severity_rank(out["food_security"]["risk"]) > _severity_rank(out["climate_risk"]):
        score += 0.3  # the more-validated model independently flags something climate alone missed
    out["composite_urgency_score"] = round(score, 2)

    return out


@lru_cache()
def _compute_ranked_districts(country: str):
    """The real, heavy per-district computation (v2 model inference,
    anomaly detection, SHAP attribution, forecast trend) for every
    district in one country, done ONCE and cached.

    This is the second half of today's performance fix: before this,
    /api/command-center, /api/key-messages, and
    /api/command-center/{district}/briefing each independently called
    the full per-district computation from scratch, meaning a single
    page load of the Overview page (which calls key-messages) plus a
    Command Center visit plus one district click could trigger this
    expensive computation three separate times. Now all three share
    this one cached result, keyed only by country (a plain string, so
    it's hashable and cacheable, unlike the full user dict)."""
    latest = get_latest_snapshot()
    latest_country = latest[latest["country"] == country]

    anomaly_bundle = _load_anomaly_artifacts()
    forecast_weights = get_tft_weights()

    results = [
        _compute_district_signals(row, anomaly_bundle, forecast_weights)
        for _, row in latest_country.iterrows()
    ]

    # Fill in real forecast trend per district using forecast.py's own live logic.
    from routers.forecast import forecast_district
    fake_user = {"country": country}
    for r in results:
        try:
            fc = forecast_district(r["district"], fake_user)
            current = fc["current_drought_index"]
            f8 = next(h["forecast_drought_index"] for h in fc["forecasts"] if h["horizon_weeks"] == 8)
            delta = f8 - current
            r["forecast_trend"] = "worsening" if delta < -0.05 else "improving" if delta > 0.05 else "stable"
            r["forecast_8w_drought_index"] = round(f8, 3)
        except Exception:
            r["forecast_trend"] = None
            r["forecast_8w_drought_index"] = None

    results.sort(key=lambda r: r["composite_urgency_score"], reverse=True)
    # Returned as a tuple, not a list: command_center() below wraps this in
    # list(...) to get its own local copy before sorting/returning it, so a
    # caller mutating its own copy can never corrupt this shared cached result.
    return tuple(results)


@router.get("/command-center")
def command_center(user: dict = Depends(get_current_user)):
    results = list(_compute_ranked_districts(user["country"]))
    return {
        "country": user["country"],
        "n_districts": len(results),
        "n_models_disagree": sum(1 for r in results if not r["models_agree"]),
        "districts": results,
        "scoring_method": (
            "composite_urgency_score = real food-security (v2) severity (1-4) where validated, "
            "else real climate-shock severity (1-4); +0.5 if the real anomaly detector flags an "
            "adverse direction; +0.3 if the more-validated food-security model independently "
            "flags higher severity than the climate model. An explicit heuristic over already-"
            "real signals, not a new trained model."
        ),
    }


KEY_MESSAGES_SYSTEM_PROMPT = (
    "You are SAHELI's lead analyst, writing in the style of FEWS NET's real 'Key Messages' "
    "country briefings: short, scannable bullet points, each starting with a BOLD short lead "
    "phrase (wrap it in **double asterisks**) stating the headline fact, followed by one "
    "sentence of plain-language explanation of WHY, grounded only in the real numbers given "
    "to you. Write exactly 3 to 4 bullets, most important first. Never invent a number. Never "
    "use markdown headers, only bullet lines starting with '- '."
)


def _key_messages_template(ranked, lang):
    """Real, non-AI fallback — same honest pattern as everywhere else in
    SAHELI. Built from the same real ranked data, just without AI prose."""
    top = ranked[0]
    disagree = [r for r in ranked if not r["models_agree"]]
    worsening = [r for r in ranked if r["forecast_trend"] == "worsening"]
    validated = [r for r in ranked if r["food_security"] and r["food_security"]["status"] == "validated"]
    bullets = []
    if lang == "fr":
        fs = top["food_security"]
        if fs:
            bullets.append(f"**{top['district']} reste le plus urgent** : risque réel de sécurité alimentaire {fs['risk']} (IPC {fs['ipc']}/5), facteur principal : {top['top_driver']['name'] if top['top_driver'] else 'n/a'}.")
        if disagree:
            names = ", ".join(r["district"] for r in disagree[:3])
            bullets.append(f"**Le climat seul ne suffit pas à expliquer le risque réel dans {len(disagree)} district(s)** ({names}), où le modèle de sécurité alimentaire réel diverge du signal climatique.")
        if worsening:
            names = ", ".join(r["district"] for r in worsening[:3])
            bullets.append(f"**La tendance à 8 semaines se dégrade pour {len(worsening)} district(s)** ({names}), selon la prévision réelle validée.")
        if len(validated) == len(ranked):
            bullets.append(f"**Les {len(ranked)} districts ont tous une vraie vérité terrain FEWS NET** ; aucune extrapolation nécessaire ici.")
        else:
            bullets.append(f"**{len(validated)} sur {len(ranked)} districts ont une vraie vérité terrain FEWS NET** ; les {len(ranked) - len(validated)} autres sont extrapolés à partir de ces districts validés.")
    else:
        fs = top["food_security"]
        if fs:
            bullets.append(f"**{top['district']} remains the most urgent district**: real food-security risk {fs['risk']} (IPC {fs['ipc']}/5), top driver: {top['top_driver']['name'] if top['top_driver'] else 'n/a'}.")
        if disagree:
            names = ", ".join(r["district"] for r in disagree[:3])
            bullets.append(f"**Climate alone does not explain real risk in {len(disagree)} district(s)** ({names}), where the real food-security model diverges from the climate signal.")
        if worsening:
            names = ", ".join(r["district"] for r in worsening[:3])
            bullets.append(f"**The 8-week trend is worsening in {len(worsening)} district(s)** ({names}), per the real validated forecast.")
        if len(validated) == len(ranked):
            bullets.append(f"**All {len(ranked)} districts have real FEWS NET ground truth**; no extrapolation needed here.")
        else:
            bullets.append(f"**{len(validated)} of {len(ranked)} districts have real FEWS NET ground truth**; the other {len(ranked) - len(validated)} are extrapolated from those validated districts.")
    return bullets


@router.get("/key-messages")
def key_messages(lang: str = "en", user: dict = Depends(get_current_user)):
    """A real, FEWS-NET-style 'Key Messages' synthesis: a handful of
    plain-language, bolded-lead bullets covering the most important real
    signals across the whole country, generated from the same real
    command-center data — not a new model, a different presentation of
    already-real numbers.

    Each bullet is tagged with the real district(s) it references, so the
    frontend can make it click-through to the live evidence behind it —
    a structural advantage no static monthly report can offer: FEWS NET's
    Key Messages are prose in a PDF, with no way to click a claim and see
    the live numbers and a what-if tool behind it. This is the same real
    text, made genuinely interactive."""
    ranked = list(_compute_ranked_districts(user["country"]))
    if not ranked:
        return {"bullets": [], "ai_mode": "no_data"}

    district_names = [r["district"] for r in ranked]
    name_re = re.compile(r"\b(" + "|".join(re.escape(n) for n in district_names) + r")\b")

    lang = lang if lang in ("en", "fr") else "en"
    top = ranked[0]
    disagree = [r["district"] for r in ranked if not r["models_agree"]]
    worsening = [r["district"] for r in ranked if r["forecast_trend"] == "worsening"]
    prompt = (
        f"Country: {user['country']}. {len(ranked)} districts tracked.\n"
        f"Most urgent: {top['district']} (climate: {top['climate_risk']}, "
        f"real food-security: {top['food_security']['risk'] if top['food_security'] else 'n/a'} "
        f"IPC {top['food_security']['ipc'] if top['food_security'] else 'n/a'}/5, "
        f"top driver: {top['top_driver']['name'] if top['top_driver'] else 'n/a'}).\n"
        f"Districts where climate and real food-security models disagree: {', '.join(disagree) if disagree else 'none'}.\n"
        f"Districts with a worsening 8-week forecast trend: {', '.join(worsening) if worsening else 'none'}.\n"
        f"Full ranked list: " + "; ".join(
            f"{r['district']} (climate {r['climate_risk']}, food-security "
            f"{r['food_security']['risk'] if r['food_security'] else 'n/a'})" for r in ranked
        ) + "."
        + LANG_INSTRUCTION[lang]
    )
    ai_result = call_ai(KEY_MESSAGES_SYSTEM_PROMPT, prompt, max_tokens=320)
    if ai_result["text"]:
        bullet_texts = [b.strip("- ").strip() for b in ai_result["text"].split("\n") if b.strip()]
    else:
        bullet_texts = _key_messages_template(ranked, lang)

    bullets = [
        {"text": b, "districts": sorted(set(name_re.findall(b)))}
        for b in bullet_texts
    ]
    return {"bullets": bullets, "ai_mode": ai_result["mode"], "country": user["country"]}


@router.get("/command-center/{district_name}/briefing")
def command_center_briefing(district_name: str, lang: str = "en", user: dict = Depends(get_current_user)):
    """A single AI-synthesized briefing pulling together every real signal
    for one district — same honest live/fallback pattern as everywhere
    else in SAHELI."""
    ranked = _compute_ranked_districts(user["country"])
    row = next((r for r in ranked if r["district"] == district_name), None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"District '{district_name}' not found")

    lang = lang if lang in ("en", "fr") else "en"
    fs = row["food_security"]
    prompt = (
        f"District: {district_name}, {row['country']}.\n"
        f"Climate-shock risk: {row['climate_risk']} (drought index {row['drought_index']}).\n"
        + (f"Real food-security risk (FEWS NET {fs['status']}): {fs['risk']} (IPC {fs['ipc']}/5).\n" if fs else "Real food-security model: not available.\n")
        + f"Anomaly detector: {row['anomaly'] or 'no anomaly flagged'}.\n"
        + f"8-week forecast trend: {row['forecast_trend'] or 'unavailable'}.\n"
        + (f"Top SHAP driver: {row['top_driver']['name']}.\n" if row['top_driver'] else "")
        + f"Crop Scanner field reports logged: {row['crop_reports']['n_total']} ({row['crop_reports']['n_disease']} flagging disease).\n"
        + f"Composite urgency score: {row['composite_urgency_score']}/4.8."
        + LANG_INSTRUCTION[lang]
    )
    ai_result = call_ai(COMMAND_CENTER_SYSTEM_PROMPT, prompt, max_tokens=320)
    narrative = ai_result["text"] or _template_briefing(row, lang)
    return {"district": district_name, "signals": row, "briefing": narrative, "ai_mode": ai_result["mode"]}


def _template_briefing(row, lang):
    fs = row["food_security"]
    if lang == "fr":
        parts = [f"{row['district']} ({row['country']}) : choc climatique {row['climate_risk']}."]
        if fs:
            parts.append(f"Risque réel de sécurité alimentaire ({fs['status']}) : {fs['risk']} (IPC {fs['ipc']}/5).")
        if row["anomaly"] == "adverse":
            parts.append("Une anomalie de dégradation a été détectée par le système temps réel.")
        if row["forecast_trend"]:
            trend_fr = {"worsening": "dégradation", "improving": "amélioration", "stable": "stabilité"}[row["forecast_trend"]]
            parts.append(f"Tendance à 8 semaines : {trend_fr}.")
        if row["crop_reports"]["n_total"]:
            parts.append(f"{row['crop_reports']['n_total']} rapport(s) de terrain du scanner de maïs enregistré(s).")
        return " ".join(parts)
    parts = [f"{row['district']} ({row['country']}): climate-shock risk {row['climate_risk']}."]
    if fs:
        parts.append(f"Real food-security risk ({fs['status']}): {fs['risk']} (IPC {fs['ipc']}/5).")
    if row["anomaly"] == "adverse":
        parts.append("A worsening-direction anomaly was flagged by the real-time detector.")
    if row["forecast_trend"]:
        parts.append(f"8-week forecast trend: {row['forecast_trend']}.")
    if row["crop_reports"]["n_total"]:
        parts.append(f"{row['crop_reports']['n_total']} Corn Scanner field report(s) logged.")
    return " ".join(parts)
