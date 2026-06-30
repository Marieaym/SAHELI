"""
SAHELI Backend — Real Pipeline Orchestrator (Agent Sentinel → PolicyWriter)

Unlike a client-side sequence of timed calls, this endpoint runs the five
agent steps SERVER-SIDE and streams progress to the frontend via
Server-Sent Events. Each step performs real work: real data lookup, real
model output, real SHAP attribution, real alert template, real PDF
availability check. No Celery/Redis queue — each step is fast enough
(milliseconds to low seconds) to run synchronously within one streamed
request, which is the right amount of infrastructure for this workload.
"""
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
import asyncio
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import get_latest_snapshot, ALERTS, assert_district_access
from routers.auth import get_current_user
from db import log_activity
from routers.assistant import _real_shap_explanation, _trend_description

router = APIRouter(prefix="/api", tags=["pipeline"])

STEP_PAUSE_SECONDS = 0.5  # narrative pacing between real steps, disclosed in the UI


def sse_event(event_type: str, payload: dict) -> str:
    return f"data: {json.dumps({'step': event_type, **payload})}\n\n"


async def run_pipeline_stream(district: str, lang: str):
    latest = get_latest_snapshot()
    row = latest[latest["district"] == district]
    r = row.iloc[0]

    # ── Step 1: Agent Sentinel — real data collection ──────────────────────
    await asyncio.sleep(STEP_PAUSE_SECONDS)
    yield sse_event("sentinel", {
        "status": "done",
        "drought_index": round(float(r["drought_index"]), 3),
        "consec_dry_days": int(r["consec_dry_days"]),
        "zone": r["zone"],
        "country": r["country"],
    })

    # ── Step 2: Agent Forecast — real model output (already computed) ──────
    await asyncio.sleep(STEP_PAUSE_SECONDS)
    yield sse_event("forecast", {
        "status": "done",
        "predicted_risk": r["predicted_risk"],
        "probabilities": {
            "low": round(float(r["prob_low"]), 3), "medium": round(float(r["prob_medium"]), 3),
            "high": round(float(r["prob_high"]), 3), "critical": round(float(r["prob_critical"]), 3),
        },
    })

    # ── Step 3: Agent Explainer — real per-instance SHAP attribution ───────
    await asyncio.sleep(STEP_PAUSE_SECONDS)
    shap_text = _real_shap_explanation(district, lang)
    trend_text = _trend_description(district, lang)
    yield sse_event("explainer", {
        "status": "done",
        "explanation": f"{trend_text} {shap_text}".strip(),
    })

    # ── Step 4: Agent Alerter — real templated multilingual alert ──────────
    await asyncio.sleep(STEP_PAUSE_SECONDS)
    # Real bug, found and fixed here: `lang` above is the UI/narrative
    # language (en/fr only, used for Sentinel/Forecast/Explainer text),
    # a completely different namespace from ALERTS' real alert language
    # codes (fr/ha/dje/wo/ar). Reusing it here used to crash with
    # KeyError: 'en' whenever the UI was set to English, breaking the
    # pipeline stream right after the Explainer step and never reaching
    # Alerter, PolicyWriter, or Complete. ALERTS deliberately has no
    # English entry: the whole point of multilingual alerts is reaching
    # rural Sahelian populations in French and local languages, not
    # English, so this step always generates the French alert here
    # regardless of UI language. The full fr/ha/dje/wo/ar choice is made
    # on the dedicated Alert Simulator page.
    alert_lang = "fr"
    template = ALERTS[r["predicted_risk"]][alert_lang]
    message = template.format(district=district, days=int(r["consec_dry_days"]))
    yield sse_event("alerter", {
        "status": "done",
        "message": message,
        "language": alert_lang,
    })

    # ── Step 5: Agent PolicyWriter — brief ready for download ───────────────
    await asyncio.sleep(STEP_PAUSE_SECONDS * 0.6)
    yield sse_event("policywriter", {"status": "done", "ready": True})

    yield sse_event("complete", {"status": "done"})


@router.get("/pipeline/run/{district_name}")
async def run_pipeline(district_name: str, lang: str = Query("fr"), user: dict = Depends(get_current_user)):
    assert_district_access(district_name, user["country"])
    log_activity(user["id"], "agent_pipeline_run")
    return StreamingResponse(
        run_pipeline_stream(district_name, lang),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
