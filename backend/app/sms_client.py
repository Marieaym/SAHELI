"""
SAHELI — Real SMS/WhatsApp dispatch client (Twilio).

Honest framing: the original Alert Simulator only ever generated message
text — it never sent anything, despite labeling the channel "SMS via
Africa's Talking API". This client makes that real.

Real SMS is the default channel, not WhatsApp, for a deliberate reason:
SAHELI's own mission, since the essay's first draft, has been reaching
farmers on basic feature phones with no internet connection. WhatsApp
needs a smartphone and a data connection, exactly what the last mile
communities this project is for often do not have. Plain SMS works on
any phone. WhatsApp is kept as a real, working second option, not the
primary one.

Credentials are read from the environment at call time, exactly like
ai_client.py, and are never hardcoded or logged. If they are missing,
this returns an honest "not configured" status instead of pretending
to send.

Setup for real SMS (your own machine, not this sandbox, api.twilio.com
is not reachable from here, this code is real but untested live for
that reason):
1. Create a free account at twilio.com, this automatically assigns a
   real Twilio phone number, no extra setup needed.
2. Console, Verified Caller IDs, add the real phone number you want to
   test with (you, or whoever will receive the demo alert). Twilio
   texts a verification code to that number once. Trial accounts can
   only send to numbers verified this way, up to 5, until upgraded.
3. Put these three values in backend/.env:
   TWILIO_ACCOUNT_SID=...
   TWILIO_AUTH_TOKEN=...
   TWILIO_PHONE_NUMBER=...   (the real Twilio number assigned in step 1, E.164 format)

Setup for WhatsApp (optional, second channel): see TWILIO_WHATSAPP_FROM
in .env.example, needs the recipient to send a join code once, from
their own WhatsApp, to Twilio's shared sandbox number.
"""
import os

_last_send: dict = {"ok": False, "error_code": None, "error": None, "channel": None}


def _creds():
    return (
        os.environ.get("TWILIO_ACCOUNT_SID", "").strip(),
        os.environ.get("TWILIO_AUTH_TOKEN", "").strip(),
        os.environ.get("TWILIO_PHONE_NUMBER", "").strip(),
        os.environ.get("TWILIO_WHATSAPP_FROM", "").strip(),
    )


def get_sms_status() -> dict:
    sid, token, sms_from, whatsapp_from = _creds()
    sms_configured = bool(sid and token and sms_from)
    whatsapp_configured = bool(sid and token and whatsapp_from)
    return {
        "configured": sms_configured or whatsapp_configured,
        "sms_configured": sms_configured,
        "whatsapp_configured": whatsapp_configured,
        "sms_from_number": sms_from if sms_configured else None,
        "whatsapp_from_number": whatsapp_from if whatsapp_configured else None,
        "last_send_ok": _last_send["ok"],
        "last_error_code": _last_send["error_code"],
        "last_error": _last_send["error"],
        "last_channel_used": _last_send["channel"],
    }


def _classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "unverified" in msg or "21608" in msg:
        return "recipient_not_verified"
    if "not a valid whatsapp" in msg or "unable to create record" in msg:
        return "invalid_recipient"
    if "63007" in msg or ("channel" in msg and "not found" in msg):
        return "recipient_not_in_sandbox"
    if "authenticate" in msg or "20003" in msg:
        return "invalid_credentials"
    if "63038" in msg or "daily messages limit" in msg:
        return "daily_limit_reached"
    return "send_error"


def send_alert(to_number: str, message: str, channel: str = "sms") -> dict:
    """to_number: a real phone number in E.164 format, e.g. +22790112233.
    channel: "sms" (default, reaches any phone, no app needed) or
    "whatsapp" (needs the recipient to have joined the sandbox first).
    Returns a real Twilio message SID and status on success, never a
    fabricated one. Returns an honest fallback status if credentials are
    missing or the real API call fails."""
    sid, token, sms_from, whatsapp_from = _creds()

    if channel == "whatsapp":
        from_number = whatsapp_from
        to_addr = to_number if to_number.startswith("whatsapp:") else f"whatsapp:{to_number}"
    else:
        channel = "sms"
        from_number = sms_from
        to_addr = to_number

    if not (sid and token and from_number):
        return {"sent": False, "mode": "fallback_no_credentials", "error": None,
                "error_code": "no_credentials", "message_sid": None, "channel": channel}

    try:
        from twilio.rest import Client
        client = Client(sid, token)
        msg = client.messages.create(body=message, from_=from_number, to=to_addr)
        _last_send.update({"ok": True, "error_code": None, "error": None, "channel": channel})
        return {
            "sent": True, "mode": "live_twilio_api", "message_sid": msg.sid,
            "status": msg.status, "error": None, "error_code": None, "channel": channel,
        }
    except Exception as exc:
        code = _classify_error(exc)
        _last_send.update({"ok": False, "error_code": code, "error": str(exc), "channel": channel})
        return {"sent": False, "mode": "fallback_error", "error": str(exc),
                "error_code": code, "message_sid": None, "channel": channel}
