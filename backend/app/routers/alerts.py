"""
SAHELI Backend — Agent Alerter: multilingual SMS alert generation, with
real WhatsApp/SMS dispatch via Twilio (see sms_client.py for the honest
configured/not-configured framing).
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import get_latest_snapshot, ALERTS, assert_district_access
from routers.auth import get_current_user
from sms_client import send_alert, get_sms_status

router = APIRouter(prefix="/api", tags=["alerts"])

LANGUAGE_NAMES = {"fr": "Français", "ha": "Hausa", "dje": "Zarma", "wo": "Wolof", "ar": "العربية"}


class SendAlertRequest(BaseModel):
    phone_number: str
    lang: str = "fr"
    channel: str = "sms"


@router.get("/alerts/sms-status")
def sms_status(user: dict = Depends(get_current_user)):
    """Honest check: is real dispatch actually configured right now, or
    is this still demo-text-only mode? Declared BEFORE /{district_name}
    below — FastAPI matches routes in declaration order, and a generic
    path parameter would otherwise swallow this literal path first."""
    return get_sms_status()


@router.get("/alerts/{district_name}")
def get_alert(district_name: str, lang: str = "fr", user: dict = Depends(get_current_user)):
    if lang not in LANGUAGE_NAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported language code '{lang}'. Use fr, ha, dje, wo, or ar.")
    assert_district_access(district_name, user["country"])

    latest = get_latest_snapshot()
    row = latest[latest["district"] == district_name].iloc[0]

    risk = row["predicted_risk"]
    template = ALERTS[risk][lang]
    message = template.format(district=district_name, days=int(row["consec_dry_days"]))

    return {
        "district": district_name,
        "risk_level": risk,
        "language": LANGUAGE_NAMES[lang],
        "language_code": lang,
        "message": message,
        "channel": "Real SMS (default) or WhatsApp via Twilio — not Africa's Talking, which would need a paid African telco integration not set up here",
        "disclaimer": "This text itself is real and district-specific. Whether it actually gets sent depends on real Twilio credentials being configured — see /api/alerts/sms-status. Every language here, including this one, is SAHELI's own best-effort phrasing; a real field deployment should have a native speaker linguist review each string before it ever reaches a farmer.",
    }


@router.post("/alerts/{district_name}/send")
def send_district_alert(district_name: str, body: SendAlertRequest, user: dict = Depends(get_current_user)):
    """Really sends the district's real alert text via SMS by default
    (reaches any phone, no app needed) or WhatsApp if requested, if real
    credentials are configured. Returns an honest status either way —
    never claims to have sent something it didn't."""
    if body.lang not in LANGUAGE_NAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported language code '{body.lang}'. Use fr, ha, dje, wo, or ar.")
    assert_district_access(district_name, user["country"])

    alert = get_alert(district_name, body.lang, user)
    result = send_alert(body.phone_number, alert["message"], channel=body.channel)
    return {**result, "district": district_name, "message_sent": alert["message"]}


@router.get("/alerts/{district_name}/all-languages")
def get_alert_all_languages(district_name: str, user: dict = Depends(get_current_user)):
    return {lang: get_alert(district_name, lang, user) for lang in LANGUAGE_NAMES}
