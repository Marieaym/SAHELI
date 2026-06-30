"""
SAHELI Backend — AI Assistant (Agent Explainer, conversational)

Calls the OpenAI API server-side using OPENAI_API_KEY from the environment.
The key never touches the frontend.

If no key is configured, the endpoint falls back to a transparent, rule-based
summary generated from the same district data.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import os
import sys
import json
import re
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import (get_latest_snapshot, get_scored_df, DATA_DIR, assert_district_access,
                          RECOMMENDATIONS, RECOMMENDATIONS_FR, get_model_artifacts,
                          get_v2_model_artifacts, v2_ipc_to_risk_level, V2_VALIDATED_DISTRICTS,
                          get_shap_explainer, get_tft_weights)
from ai_client import call_ai, get_ai_status
from routers.auth import get_current_user
from db import log_activity
import numpy as np

router = APIRouter(prefix="/api", tags=["assistant"])

SEVERITY = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

FEATURE_COLS = [
    "precip_30d", "precip_90d", "et_30d", "temp_30d_avg",
    "water_balance_30d", "drought_index", "consec_dry_days",
    "month", "monsoon_season", "lat", "lon",
    "conflict_events_30d", "conflict_fatalities_30d", "price_anomaly_30d",
    "groundwater_anomaly_cm", "water_point_count_50km", "sentinel2_ndvi",
]

FEATURE_NAMES = {
    "drought_index": "drought index", "water_balance_30d": "30-day water balance",
    "consec_dry_days": "consecutive dry days", "precip_30d": "30-day rainfall",
    "precip_90d": "90-day rainfall", "et_30d": "30-day evapotranspiration",
    "temp_30d_avg": "average temperature", "lat": "latitude", "lon": "longitude",
    "month": "month of year", "monsoon_season": "monsoon timing",
    "conflict_events_30d": "nearby conflict events (30d, real ACLED)",
    "conflict_fatalities_30d": "nearby conflict fatalities (30d, real ACLED)",
    "price_anomaly_30d": "millet price anomaly (real WFP market data)",
    "groundwater_anomaly_cm": "groundwater storage anomaly (real GRACE-FO satellite data)",
    "water_point_count_50km": "mapped water points within 50km (real OpenStreetMap data)",
    "sentinel2_ndvi": "vegetation health index (real Sentinel-2 satellite imagery)",
}

FEATURE_NAMES_FR = {
    "drought_index": "indice de sécheresse", "water_balance_30d": "bilan hydrique sur 30 jours",
    "consec_dry_days": "jours secs consécutifs", "precip_30d": "précipitations sur 30 jours",
    "precip_90d": "précipitations sur 90 jours", "et_30d": "évapotranspiration sur 30 jours",
    "temp_30d_avg": "température moyenne", "lat": "latitude", "lon": "longitude",
    "month": "mois de l'année", "monsoon_season": "période de mousson",
    "conflict_events_30d": "événements de conflit proches (30j, données réelles ACLED)",
    "conflict_fatalities_30d": "victimes de conflit proches (30j, données réelles ACLED)",
    "price_anomaly_30d": "anomalie du prix du mil (données réelles WFP)",
    "groundwater_anomaly_cm": "anomalie de stockage souterrain (données réelles GRACE-FO)",
    "water_point_count_50km": "points d'eau cartographiés dans un rayon de 50km (données réelles OSM)",
    "sentinel2_ndvi": "indice de santé de la végétation (imagerie réelle Sentinel-2)",
}


def _real_shap_explanation(district: str, lang: str = "en") -> str:
    """
    Real per-instance SHAP explanation for this specific district's current
    prediction — not global feature importance, the actual local attribution
    from the saved TreeExplainer, run live on this district's current feature row.
    """
    try:
        explainer = get_shap_explainer()
        if explainer is None:
            return ""
        model, le, _ = get_model_artifacts()
        latest = get_latest_snapshot()
        row = latest[latest["district"] == district]
        if row.empty:
            return ""
        r = row.iloc[0]
        X = r[FEATURE_COLS].to_frame().T.astype(float)
        predicted_class_idx = list(le.classes_).index(r["predicted_risk"])
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            class_shap = shap_values[predicted_class_idx][0]
        else:
            class_shap = shap_values[0, :, predicted_class_idx]
        contributions = sorted(zip(FEATURE_COLS, class_shap), key=lambda x: -abs(x[1]))[:3]
        names = FEATURE_NAMES_FR if lang == "fr" else FEATURE_NAMES
        parts = []
        for feat, val in contributions:
            if lang == "fr":
                direction = "a augmenté" if val > 0 else "a diminué"
                parts.append(f"{names.get(feat, feat)} {direction} le score de risque de {abs(val):.2f}")
            else:
                direction = "increased" if val > 0 else "decreased"
                parts.append(f"{names.get(feat, feat)} {direction} the risk score by {abs(val):.2f}")
        if lang == "fr":
            return f"Pour {district} précisément, l'attribution du modèle montre : " + "; ".join(parts) + "."
        return f"For {district} specifically, the model's own attribution shows: " + "; ".join(parts) + "."
    except Exception:
        return ""


class AssistantQuery(BaseModel):
    question: str
    district: str | None = None
    lang: str = "en"


def _trend_description(district: str, lang: str = "en") -> str:
    """Compare the last 30 days of drought index to the prior 30 days for this district."""
    df = get_scored_df()
    hist = df[df["district"] == district].sort_values("date").tail(60)
    if len(hist) < 60:
        return ""
    recent = hist["drought_index"].tail(30).mean()
    prior = hist["drought_index"].head(30).mean()
    delta = recent - prior
    if lang == "fr":
        if delta < -0.15:
            return f"L'indice de sécheresse s'est aggravé de {abs(delta):.2f} sur les 30 derniers jours par rapport aux 30 précédents."
        elif delta > 0.15:
            return f"L'indice de sécheresse s'est amélioré de {delta:.2f} sur les 30 derniers jours par rapport aux 30 précédents."
        return "L'indice de sécheresse est resté stable sur les 60 derniers jours."
    if delta < -0.15:
        return f"The drought index has worsened by {abs(delta):.2f} over the past 30 days compared to the previous 30."
    elif delta > 0.15:
        return f"The drought index has improved by {delta:.2f} over the past 30 days compared to the previous 30."
    return "The drought index has been stable over the past 60 days."


def _top_drivers() -> str:
    """Real SHAP-derived global feature importance, for grounding explanations in the actual model."""
    try:
        with open(os.path.join(DATA_DIR, "metrics.json")) as f:
            metrics = json.load(f)
        importance = metrics.get("feature_importance", {})
        top3 = sorted(importance.items(), key=lambda x: -x[1])[:3]
        names = {"drought_index": "drought index", "water_balance_30d": "30-day water balance",
                 "consec_dry_days": "consecutive dry days", "lat": "latitude", "lon": "longitude",
                 "precip_30d": "30-day rainfall", "et_30d": "30-day evapotranspiration", "temp_30d_avg": "average temperature"}
        readable = [names.get(k, k) for k, _ in top3]
        return f"Across the full model, the strongest predictors of food-security risk are: {', '.join(readable)}."
    except Exception:
        return ""


_INTENT_CLASSIFIER = None
_INTENT_DISTRICT_RE = None


def _load_intent_classifier():
    """Lazily load the real, locally-trained intent classifier (TF-IDF +
    Logistic Regression, trained by us in models/train_intent_classifier.py).
    No external API involved — this runs even with zero API keys configured."""
    global _INTENT_CLASSIFIER
    if _INTENT_CLASSIFIER is None:
        try:
            _INTENT_CLASSIFIER = joblib.load(os.path.join(DATA_DIR, "intent_classifier.joblib"))
        except Exception:
            _INTENT_CLASSIFIER = False  # mark as "tried and unavailable"
    return _INTENT_CLASSIFIER or None


def classify_intent(question: str, user_country: str) -> tuple[str | None, float]:
    """Returns (intent_label, confidence) using the real trained classifier,
    or (None, 0.0) if the model isn't available — callers must handle that
    gracefully and fall back to the existing district-count-based routing."""
    clf = _load_intent_classifier()
    if not clf or not question:
        return None, 0.0
    latest = get_latest_snapshot()
    country_districts = latest[latest["country"] == user_country]["district"].tolist()
    if country_districts:
        pattern = re.compile(r"\b(" + "|".join(re.escape(d) for d in country_districts) + r")\b", re.IGNORECASE)
        normalized = pattern.sub("<DISTRICT>", question)
    else:
        normalized = question
    try:
        proba = clf.predict_proba([normalized])[0]
        idx = proba.argmax()
        return clf.classes_[idx], float(proba[idx])
    except Exception:
        return None, 0.0


def detect_mentioned_districts(question: str, user_country: str) -> list[str]:
    """
    Scan the question text for any district name belonging to the user's own
    country. Scoped by construction: only that country's district names are
    ever checked, so this can never leak another country's district into context.
    """
    latest = get_latest_snapshot()
    country_districts = latest[latest["country"] == user_country]["district"].tolist()
    q_lower = question.lower()
    return [d for d in country_districts if re.search(rf"\b{re.escape(d.lower())}\b", q_lower)]


def _district_block(r) -> str:
    return (
        f"{r['district']} ({r['zone']} zone): {r['predicted_risk']} risk, "
        f"drought index {r['drought_index']:.2f}, {int(r['consec_dry_days'])} consecutive dry days, "
        f"model confidence {max(r['prob_low'], r['prob_medium'], r['prob_high'], r['prob_critical'])*100:.0f}%."
    )


def _recommendations_block(risk: str, district: str) -> str:
    actions = RECOMMENDATIONS.get(risk, [])
    if not actions:
        return ""
    return f"Recommended actions for {district}: " + "; ".join(actions) + "."


def _ranked_list_block(latest_country) -> str:
    ranked = latest_country.copy()
    ranked["severity"] = ranked["predicted_risk"].map(SEVERITY)
    ranked = ranked.sort_values("severity", ascending=False)
    lines = [f"{r['district']} ({r['predicted_risk']}, drought index {r['drought_index']:.2f})" for _, r in ranked.iterrows()]
    return "Districts ranked by current severity, most urgent first: " + "; ".join(lines) + "."


def build_context(district: str | None, user_country: str, question: str = "") -> str:
    latest = get_latest_snapshot()
    latest_country = latest[latest["country"] == user_country]

    mentioned = detect_mentioned_districts(question, user_country) if question else []
    if district and district not in mentioned:
        mentioned = [district] + mentioned

    # ── Multi-district comparison ──────────────────────────────────────────
    if len(mentioned) >= 2:
        blocks = []
        for d in mentioned[:3]:
            row = latest_country[latest_country["district"] == d]
            if not row.empty:
                blocks.append(_district_block(row.iloc[0]))
        return (
            f"Comparing {len(blocks)} districts in {user_country}: " + " | ".join(blocks)
        )

    # ── Single district ─────────────────────────────────────────────────────
    if mentioned:
        d = mentioned[0]
        row = latest_country[latest_country["district"] == d]
        if not row.empty:
            r = row.iloc[0]
            base = (
                f"District: {r['district']}, {r['country']} ({r['zone']} zone). "
                f"Current predicted risk: {r['predicted_risk']}. "
                f"Drought index: {r['drought_index']:.2f} std. dev. from historical baseline. "
                f"Consecutive dry days: {int(r['consec_dry_days'])}. "
                f"Risk probabilities — Low: {r['prob_low']:.2f}, Medium: {r['prob_medium']:.2f}, "
                f"High: {r['prob_high']:.2f}, Critical: {r['prob_critical']:.2f}. "
            )
            trend = _trend_description(d)
            shap_explanation = _real_shap_explanation(d)
            recs = _recommendations_block(r["predicted_risk"], d)
            return f"{base}{trend} {shap_explanation} {recs}".strip()

    # ── No specific district — country overview, with full ranking ─────────
    n_critical = (latest_country["predicted_risk"] == "Critical").sum()
    n_high = (latest_country["predicted_risk"] == "High").sum()
    critical_districts = latest_country[latest_country["predicted_risk"] == "Critical"]["district"].tolist()
    drivers = _top_drivers()
    ranked = _ranked_list_block(latest_country)
    top_recs = ""
    if not latest_country.empty:
        top_row = latest_country.loc[latest_country["predicted_risk"].map(SEVERITY).idxmax()]
        top_recs = _recommendations_block(top_row["predicted_risk"], top_row["district"])
    return (
        f"{user_country} overview: {len(latest_country)} districts monitored. "
        f"{n_critical} districts at Critical risk: {', '.join(critical_districts) if critical_districts else 'none'}. "
        f"{n_high} districts at High risk. {ranked} {drivers} {top_recs}"
    )


SYSTEM_PROMPT_BASE = (
    "You are the SAHELI Agent Explainer, an AI assistant embedded in a Sahel food-security "
    "early-warning dashboard. You answer questions from government officials, NGO staff, and "
    "researchers about current food-security risk conditions, using ONLY the district data "
    "context provided to you. Handle four kinds of questions: (1) single-district explanation, "
    "(2) side-by-side comparison when multiple districts are in the context, (3) priority "
    "ranking ('which district needs attention') using the provided ranked list, (4) requests "
    "for recommended actions, using the provided recommendation list verbatim where relevant. "
    "Structure substantive answers in clear parts: current situation and key drivers, trend, "
    "and a concrete policy-relevant recommendation. Be specific and analytical, not generic — "
    "reference the actual numbers given. Never invent data not present in the context. If asked "
    "something outside the provided context, say so clearly rather than guessing. Aim for "
    "6-10 sentences for substantive questions, more if comparing multiple districts."
)

LANG_INSTRUCTION = {
    "en": " Respond entirely in English.",
    "fr": " Réponds entièrement en français, y compris les chiffres et les unités, dans un français professionnel et naturel, pas une traduction littérale mot à mot.",
}


def _real_food_security_answer(district: str, user_country: str, lang: str) -> str | None:
    """Real answer using the v2 model (trained directly on real FEWS NET
    ground truth), for the food_security_question intent. Returns None
    if the model artifact or the district row isn't available, so the
    caller can fall back to the existing district summary instead of
    showing a broken answer."""
    v2_model, v2_features = get_v2_model_artifacts()
    if v2_model is None:
        return None
    latest = get_latest_snapshot()
    row = latest[latest["district"] == district]
    if row.empty:
        return None
    r = row.iloc[0]
    x = r[v2_features].values.astype(float).reshape(1, -1)
    ipc = float(np.clip(v2_model.predict(x)[0], 1.0, 5.0))
    fs_risk = v2_ipc_to_risk_level(ipc)
    status = "validated" if district in V2_VALIDATED_DISTRICTS else "extrapolated"
    climate_risk = r["predicted_risk"]
    if lang == "fr":
        status_txt = "validé avec de vraies données FEWS NET" if status == "validated" else "extrapolé, pas encore de vraie donnée FEWS NET locale"
        agree = "Les deux modèles sont d'accord." if fs_risk == climate_risk else f"Cela diffère du signal climatique seul ({climate_risk}), un écart réel qu'on a documenté, pas du bruit."
        return (
            f"Pour {district}, le vrai modèle de sécurité alimentaire (entraîné directement sur "
            f"32 443 vraies observations FEWS NET, corrélation r=0,62 avec la réalité contre r=-0,20 "
            f"pour le modèle climatique seul) indique : IPC {ipc:.2f}/5, risque {fs_risk}. Statut : {status_txt}. {agree}"
        )
    status_txt = "validated with real FEWS NET data" if status == "validated" else "extrapolated, no real local FEWS NET data yet"
    agree = "Both models agree." if fs_risk == climate_risk else f"This differs from the climate-only signal ({climate_risk}), a real, documented gap, not noise."
    return (
        f"For {district}, the real food-security model (trained directly on 32,443 real FEWS NET "
        f"observations, r=0.62 with real outcomes vs r=-0.20 for the climate-only model) shows: "
        f"IPC {ipc:.2f}/5, {fs_risk} risk. Status: {status_txt}. {agree}"
    )


def _real_forecast_answer(district: str, user_country: str, lang: str) -> str | None:
    """Real answer using the TFT-lite temporal attention forecaster, for
    the forecast_question intent. Returns None if weights aren't
    available so the caller can fall back gracefully."""
    w = get_tft_weights()
    if w is None:
        return None
    try:
        df = get_scored_df()
        dist_df = df[df["district"] == district].copy()
        if dist_df.empty:
            return None
        dist_df["week"] = dist_df["date"].dt.to_period("W").apply(lambda p: p.start_time)
        feats = ["drought_index", "water_balance_30d", "sentinel2_ndvi", "price_anomaly_30d", "conflict_events_30d"]
        weekly = dist_df.groupby("week")[feats].mean().reset_index().sort_values("week")
        lookback = int(w["lookback_weeks"])
        if len(weekly) < lookback:
            return None
        window = weekly[feats].values[-lookback:]
        last_value = float(window[-1, 0])
        X = ((window - w["feat_mean"]) / w["feat_std"])[None, :, :]

        def relu(x): return np.maximum(0, x)
        def softmax(x):
            x = x - x.max(axis=-1, keepdims=True); e = np.exp(x); return e / e.sum(axis=-1, keepdims=True)
        n_heads, d_model = int(w["n_heads"]), int(w["d_model"])
        d_head = d_model // n_heads
        embed = relu(X @ w["W_embed"] + w["b_embed"])
        Q, K, V = embed @ w["W_q"], embed @ w["W_k"], embed @ w["W_v"]
        B, T = X.shape[0], X.shape[1]
        Qh = Q.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
        scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(d_head)
        attn = softmax(scores)
        attn_out = (attn @ Vh).transpose(0, 2, 1, 3).reshape(B, T, n_heads * d_head)
        residual = embed + attn_out @ w["W_o"]
        pooled = residual.mean(axis=1)
        ctx = relu(pooled @ w["W_ctx"] + w["b_ctx"])
        delta_preds_s = ctx @ w["W_heads"].T + w["b_heads"]
        deltas = delta_preds_s[0] * float(w["delta_std"]) + float(w["delta_mean"])
        forecasts = deltas + last_value
        horizons = [4, 8, 12]
        if lang == "fr":
            parts = [f"+{h}sem: {f:.2f}" for h, f in zip(horizons, forecasts)]
            return (
                f"Prévision réelle pour {district} (attention temporelle, validée sur 2023-2024 jamais "
                f"vue, 8% meilleure que la persistance à 8 semaines, 21% à 12 semaines) : indice "
                f"sécheresse actuel {last_value:.2f}, projeté { ' / '.join(parts) }. Une valeur plus basse "
                f"indique une aggravation."
            )
        parts = [f"+{h}w: {f:.2f}" for h, f in zip(horizons, forecasts)]
        return (
            f"Real forecast for {district} (temporal attention, validated on 2023-2024 never-seen "
            f"data, 8% better than persistence at 8 weeks, 21% at 12 weeks): current drought index "
            f"{last_value:.2f}, projected { ' / '.join(parts) }. A lower value means worsening conditions."
        )
    except Exception:
        return None


def fallback_response(context: str, district: str | None, question: str, user_country: str, lang: str = "en") -> dict:
    """
    Professional fallback when no OpenAI API key is configured (or the live
    call fails). Generates a clean analytical summary from real district data.
    """
    latest = get_latest_snapshot()
    latest_country = latest[latest["country"] == user_country]
    mentioned = detect_mentioned_districts(question, user_country) if question else []
    if district and district not in mentioned:
        mentioned = [district] + mentioned
    recs_dict = RECOMMENDATIONS_FR if lang == "fr" else RECOMMENDATIONS

    # ── Real, locally-trained intent signal (no API key needed) ────────────
    intent, intent_conf = classify_intent(question, user_country) if question else (None, 0.0)

    # Off-topic, with reasonable confidence and no district named — be honest
    # about scope rather than awkwardly defaulting to a country overview.
    if intent == "off_topic" and intent_conf > 0.55 and not mentioned:
        if lang == "fr":
            return {
                "answer": "Je ne peux répondre qu'aux questions sur le risque de sécurité alimentaire dans le périmètre de SAHELI — districts, climat, conflit, marché, ou recommandations. Pose-moi une question sur l'une de ces vraies données.",
                "mode": "indicator_summary", "context_used": context,
            }
        return {
            "answer": "I can only answer questions within SAHELI's real food-security scope — districts, climate, conflict, markets, or recommendations. Ask me something grounded in that real data.",
            "mode": "indicator_summary", "context_used": context,
        }

    # Explicit recommendation intent with a named district — always give a
    # real action-oriented answer, not gated behind risk level the way the
    # single-district branch below is.
    if intent == "recommendation" and intent_conf > 0.45 and len(mentioned) == 1:
        d = mentioned[0]
        row = latest_country[latest_country["district"] == d]
        if not row.empty:
            r = row.iloc[0]
            actions = recs_dict.get(r["predicted_risk"], [])
            if actions:
                if lang == "fr":
                    answer = f"Pour {d} (risque {r['predicted_risk']}), actions recommandées : {'; '.join(actions)}."
                else:
                    answer = f"For {d} ({r['predicted_risk']} risk), recommended actions: {'; '.join(actions)}."
                return {"answer": answer, "mode": "indicator_summary", "context_used": context}

    # Real food-security model (v2) question, with a named district — a
    # genuinely separate intent from a plain single-district summary,
    # routed to the actual FEWS-NET-validated model rather than the
    # climate-only context. Conservative threshold: if the classifier
    # isn't confident, fall through to the existing behavior below rather
    # than risk a wrong routing.
    if intent == "food_security_question" and intent_conf > 0.40 and mentioned:
        answer = _real_food_security_answer(mentioned[0], user_country, lang)
        if answer:
            return {"answer": answer, "mode": "indicator_summary", "context_used": context}

    # Real multi-horizon forecast question, with a named district.
    if intent == "forecast_question" and intent_conf > 0.40 and mentioned:
        answer = _real_forecast_answer(mentioned[0], user_country, lang)
        if answer:
            return {"answer": answer, "mode": "indicator_summary", "context_used": context}

    # ── Comparison ──────────────────────────────────────────────────────────
    if len(mentioned) >= 2:
        if lang == "fr":
            sentences = [f"Comparaison entre {' et '.join(mentioned[:3])} :"]
        else:
            sentences = [f"Comparing {' and '.join(mentioned[:3])}:"]
        rows = []
        for d in mentioned[:3]:
            row = latest_country[latest_country["district"] == d]
            if not row.empty:
                r = row.iloc[0]
                rows.append(r)
                if lang == "fr":
                    sentences.append(f"{r['district']} est en risque {r['predicted_risk']} (indice de sécheresse {r['drought_index']:.2f}, {int(r['consec_dry_days'])} jours secs).")
                else:
                    sentences.append(f"{r['district']} is at {r['predicted_risk']} risk (drought index {r['drought_index']:.2f}, {int(r['consec_dry_days'])} dry days).")
        if len(rows) == 2:
            worse = rows[0] if SEVERITY[rows[0]["predicted_risk"]] >= SEVERITY[rows[1]["predicted_risk"]] else rows[1]
            if lang == "fr":
                sentences.append(f"{worse['district']} présente actuellement les conditions les plus sévères des deux.")
            else:
                sentences.append(f"{worse['district']} currently shows the more severe conditions of the two.")
        return {"answer": " ".join(sentences), "mode": "indicator_summary", "context_used": context}

    # ── Single district ─────────────────────────────────────────────────────
    if mentioned:
        d = mentioned[0]
        row = latest_country[latest_country["district"] == d]
        if not row.empty:
            r = row.iloc[0]
            if lang == "fr":
                sentences = [f"{r['district']} est actuellement classé en risque {r['predicted_risk']}."]
                if r["drought_index"] < -0.3:
                    sentences.append(
                        f"L'indice de sécheresse est de {abs(r['drought_index']):.2f} écarts-types sous la "
                        f"référence historique du district, avec {int(r['consec_dry_days'])} jours secs consécutifs enregistrés."
                    )
                elif r["drought_index"] > 0.3:
                    sentences.append("Le bilan hydrique récent est supérieur à la référence historique du district.")
                else:
                    sentences.append("Les indicateurs climatiques sont dans la plage historique normale du district.")
            else:
                sentences = [f"{r['district']} is currently classified at {r['predicted_risk']} risk."]
                if r["drought_index"] < -0.3:
                    sentences.append(
                        f"The drought index is {abs(r['drought_index']):.2f} standard deviations below "
                        f"the district's historical baseline, with {int(r['consec_dry_days'])} consecutive dry days recorded."
                    )
                elif r["drought_index"] > 0.3:
                    sentences.append("Recent water balance is above the district's historical baseline.")
                else:
                    sentences.append("Climate indicators are within the district's normal historical range.")
            top_prob = max(
                [("Low", r["prob_low"]), ("Medium", r["prob_medium"]), ("High", r["prob_high"]), ("Critical", r["prob_critical"])],
                key=lambda x: x[1],
            )
            if lang == "fr":
                sentences.append(f"La confiance du modèle pour cette classification est de {top_prob[1]*100:.0f}%.")
            else:
                sentences.append(f"Model confidence for this classification is {top_prob[1]*100:.0f}%.")
            trend = _trend_description(d, lang)
            if trend:
                sentences.append(trend)
            shap_explanation = _real_shap_explanation(d, lang)
            if shap_explanation:
                sentences.append(shap_explanation)
            if r["predicted_risk"] in ("High", "Critical"):
                actions = recs_dict.get(r["predicted_risk"], [])
                if actions:
                    if lang == "fr":
                        sentences.append(f"Actions recommandées : {'; '.join(actions[:3])}.")
                    else:
                        sentences.append(f"Recommended actions: {'; '.join(actions[:3])}.")
            return {"answer": " ".join(sentences), "mode": "indicator_summary", "context_used": context}

    # ── Ranking / country overview ──────────────────────────────────────────
    if not latest_country.empty:
        ranked = latest_country.copy()
        ranked["severity"] = ranked["predicted_risk"].map(SEVERITY)
        ranked = ranked.sort_values("severity", ascending=False)
        top = ranked.iloc[0]
        if lang == "fr":
            sentences = [
                f"Sur {len(latest_country)} districts surveillés au {user_country}, "
                f"{top['district']} nécessite actuellement l'attention la plus urgente, en risque {top['predicted_risk']} "
                f"(indice de sécheresse {top['drought_index']:.2f})."
            ]
        else:
            sentences = [
                f"Of {len(latest_country)} districts monitored in {user_country}, "
                f"{top['district']} currently requires the most urgent attention, at {top['predicted_risk']} risk "
                f"(drought index {top['drought_index']:.2f})."
            ]
        if len(ranked) > 1:
            runner_up = ranked.iloc[1]
            if lang == "fr":
                sentences.append(f"{runner_up['district']} suit en risque {runner_up['predicted_risk']}.")
            else:
                sentences.append(f"{runner_up['district']} follows at {runner_up['predicted_risk']} risk.")
        actions = recs_dict.get(top["predicted_risk"], [])
        if actions:
            if lang == "fr":
                sentences.append(f"Actions recommandées pour {top['district']} : {'; '.join(actions[:3])}.")
            else:
                sentences.append(f"Recommended actions for {top['district']}: {'; '.join(actions[:3])}.")
        return {"answer": " ".join(sentences), "mode": "indicator_summary", "context_used": context}

    return {"answer": context, "mode": "indicator_summary", "context_used": context}


@router.get("/assistant/status")
def assistant_status(user: dict = Depends(get_current_user)):
    return get_ai_status()


@router.post("/assistant/ask")
def ask_assistant(query: AssistantQuery, user: dict = Depends(get_current_user)):
    log_activity(user["id"], "assistant_question")
    if query.district:
        assert_district_access(query.district, user["country"])
    lang = query.lang if query.lang in ("en", "fr") else "en"
    context = build_context(query.district, user["country"], query.question)
    system_prompt = SYSTEM_PROMPT_BASE + LANG_INSTRUCTION[lang]
    result = call_ai(
        system_prompt=system_prompt,
        user_prompt=f"Context data: {context}\n\nQuestion: {query.question}",
        max_tokens=600,
    )
    if result["mode"] == "live_openai_api":
        return {"answer": result["text"], "mode": "live_openai_api"}

    fb = fallback_response(context, query.district, query.question, user["country"], lang)
    if result["mode"] == "fallback_error":
        fb["ai_error"] = result["error"]
        fb["error_code"] = result.get("error_code")
        fb["mode"] = "fallback_error" if result.get("error_code") == "quota_exceeded" else "indicator_summary"
    return fb
