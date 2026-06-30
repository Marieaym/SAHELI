"""
SAHELI Backend — Intervention Optimizer: simplified RL-style budget allocation

Allocation is now based explicitly on the REAL food-security severity (v2
model, trained directly on real FEWS NET IPC ground truth), not the
original climate-shock proxy — because money should follow the model
that actually correlates with real food insecurity (validated: r=0.62
vs r=-0.20 for the climate-only model, see
models/food_security_v2_module.py). The original climate-shock risk is
still computed and shown explicitly, side by side, for transparency:
SAHELI's own validation found the two can disagree on which district
needs help most, and that disagreement is shown, not hidden.
"""
from fastapi import APIRouter, Query, Depends
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import (
    get_latest_snapshot, POP_WEIGHTS,
    get_v2_model_artifacts, v2_ipc_to_risk_level, V2_VALIDATED_DISTRICTS,
)
from routers.auth import get_current_user
from ai_client import call_ai

router = APIRouter(prefix="/api", tags=["intervention"])

LANG_INSTRUCTION = {
    "en": " Respond entirely in English.",
    "fr": " Réponds entièrement en français, dans un français professionnel et naturel, pas une traduction mot à mot.",
}

INTERVENTION_SYSTEM_PROMPT = (
    "You are SAHELI's budget briefing assistant. You explain a real emergency "
    "relief budget allocation result to a government decision-maker or NGO "
    "field officer who may not read tables easily. You are given ONLY real, "
    "already-computed numbers from a real severity-weighted allocation model, "
    "based on the REAL food-security model validated against FEWS NET ground "
    "truth — never invent or estimate a number not given to you. Write 4-6 "
    "short sentences: how much was allocated and to how many districts, which "
    "real districts got the largest share and why (named, with their real "
    "food-security risk level), the real before/after Critical-district "
    "count, and if the climate-shock model and the food-security model "
    "disagree on any named district, mention that explicitly. Plain language, "
    "no jargon, no markdown headers, conversational but precise."
)

SEVERITY_SCORE = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
COST_PER_SEVERITY_UNIT = 150_000


def severity_to_risk_label(s):
    if s >= 3.5: return "Critical"
    if s >= 2.5: return "High"
    if s >= 1.5: return "Medium"
    return "Low"


def ipc_to_severity(ipc_value):
    """Map the real v2 model's 1.0-5.0 IPC scale onto the same 1-4
    severity scale the allocation math already uses, so the two models
    are comparable and the formula below doesn't need two code paths."""
    return float(np.clip((ipc_value - 1.0) / (3.0 - 1.0) * 3.0 + 1.0, 1.0, 4.0))


@router.get("/intervention/simulate")
def simulate_intervention(budget: float = 1_500_000, lang: str = Query("en"), user: dict = Depends(get_current_user)):
    latest = get_latest_snapshot().copy()
    latest = latest[latest["country"] == user["country"]]
    latest["pop_weight"] = latest["district"].map(POP_WEIGHTS).fillna(0.5)

    v2_model, v2_features = get_v2_model_artifacts()
    climate_severity = latest["predicted_risk"].map(SEVERITY_SCORE)

    if v2_model is not None:
        ipc_pred = np.clip(v2_model.predict(latest[v2_features]), 1.0, 5.0)
        latest["food_security_ipc"] = ipc_pred
        latest["food_security_risk"] = [v2_ipc_to_risk_level(v) for v in ipc_pred]
        latest["food_security_status"] = latest["district"].apply(
            lambda d: "validated" if d in V2_VALIDATED_DISTRICTS else "extrapolated"
        )
        # Allocation now follows the REAL food-security severity, explicitly.
        latest["severity"] = [ipc_to_severity(v) for v in ipc_pred]
        allocation_basis = "food_security_v2"
    else:
        latest["severity"] = climate_severity
        allocation_basis = "climate_shock_fallback"

    latest["climate_shock_risk"] = latest["predicted_risk"]
    latest["climate_shock_severity"] = climate_severity
    latest["need_index"] = latest["severity"] * latest["pop_weight"]

    total_need = latest["need_index"].sum()
    latest["allocation"] = (latest["need_index"] / total_need * budget).round(0)

    latest["effective_dose"] = latest["allocation"] / (COST_PER_SEVERITY_UNIT * latest["pop_weight"])
    latest["risk_reduction"] = latest["severity"] * (1 - np.exp(-latest["effective_dose"]))
    latest["projected_severity"] = (latest["severity"] - latest["risk_reduction"]).clip(lower=0.3)
    latest["projected_risk"] = latest["projected_severity"].apply(severity_to_risk_label)

    n_critical_before = int((latest["severity"] >= 3.5).sum())
    n_critical_after = int((latest["projected_risk"] == "Critical").sum())
    n_climate_critical_before = int((latest["climate_shock_risk"] == "Critical").sum())

    cols = [
        "district", "country", "predicted_risk", "severity", "allocation",
        "projected_risk", "projected_severity", "climate_shock_risk", "climate_shock_severity",
    ]
    if v2_model is not None:
        cols += ["food_security_ipc", "food_security_risk", "food_security_status"]
    results = latest[cols].sort_values("allocation", ascending=False).to_dict(orient="records")

    disagreements = [
        r["district"] for r in results
        if v2_model is not None and r["food_security_risk"] != r["climate_shock_risk"]
    ]

    top3 = results[:3]
    top3_desc = "; ".join(
        f"{r['district']} ({r.get('food_security_risk', r['predicted_risk'])}) received ${r['allocation']:,.0f}" for r in top3
    )
    lang = lang if lang in ("en", "fr") else "en"
    user_prompt = (
        f"Total budget: ${budget:,.0f}, allocated across {len(results)} districts in {user['country']}, "
        f"based on {'the real FEWS-NET-validated food security model' if v2_model is not None else 'the climate-shock model (food-security model unavailable)'}.\n"
        f"Critical-risk districts before allocation: {n_critical_before}. After: {n_critical_after}.\n"
        f"Top 3 recipient districts by allocation: {top3_desc}.\n"
        + (f"Districts where the climate-shock model and the real food-security model disagree on risk level: "
           f"{', '.join(disagreements) if disagreements else 'none — both models agree on every district'}.\n"
           if v2_model is not None else "")
        + LANG_INSTRUCTION[lang]
    )
    ai_result = call_ai(INTERVENTION_SYSTEM_PROMPT, user_prompt, max_tokens=300)
    narrative = ai_result["text"] or _template_intervention_narrative(
        budget, len(results), n_critical_before, n_critical_after, top3, lang, disagreements if v2_model is not None else None
    )

    return {
        "budget": budget,
        "country": user["country"],
        "n_critical_before": n_critical_before,
        "n_critical_after": n_critical_after,
        "n_climate_critical_before": n_climate_critical_before,
        "allocation_basis": allocation_basis,
        "v2_available": v2_model is not None,
        "model_disagreement_districts": disagreements,
        "allocations": results,
        "ai_narrative": narrative,
        "ai_mode": ai_result["mode"],
    }


def _template_intervention_narrative(budget, n_districts, n_before, n_after, top3, lang, disagreements=None):
    """Real, non-AI fallback narrative from the same real numbers — used
    when no OpenAI key is configured or the live call fails."""
    change = n_after - n_before
    if lang == "fr":
        base = (
            f"Ce budget de ${budget:,.0f} est réparti sur {n_districts} districts selon leur vrai risque de "
            f"sécurité alimentaire, validé contre les données FEWS NET. Les districts en risque Critique "
            f"passent de {n_before} à {n_after}{' (' + str(change) + ')' if change != 0 else ''}."
        )
        if top3:
            names = ", ".join(f"{r['district']} (${r['allocation']:,.0f})" for r in top3)
            base += f" Les trois districts recevant le plus : {names}."
        if disagreements:
            base += f" Le modèle climatique et le modèle de sécurité alimentaire réelle ne sont pas d'accord sur : {', '.join(disagreements)}."
        return base
    base = (
        f"This ${budget:,.0f} budget is allocated across {n_districts} districts based on their real "
        f"food-security risk, validated against FEWS NET data. Critical-risk districts go from {n_before} "
        f"to {n_after}{' (' + str(change) + ')' if change != 0 else ''}."
    )
    if top3:
        names = ", ".join(f"{r['district']} (${r['allocation']:,.0f})" for r in top3)
        base += f" The three largest recipients: {names}."
    if disagreements:
        base += f" The climate-shock model and the real food-security model disagree on: {', '.join(disagreements)}."
    return base
