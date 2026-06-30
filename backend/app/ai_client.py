"""
SAHELI — AI client, multi-provider (OpenAI, Google Gemini, and/or DeepSeek).

Keys read at call time from OPENAI_API_KEY, GEMINI_API_KEY, and/or
DEEPSEEK_API_KEY. Any combination, including none, can be configured:
  - Several configured: tries them in order (OpenAI, then Gemini, then
    DeepSeek), and if one specific call fails (quota, invalid key, rate
    limit, network), automatically retries the SAME request on the next
    configured provider before falling back to template mode. This is
    real redundancy, not a cosmetic toggle — if one provider's quota runs
    out mid-demo, the next can carry the request without the page
    breaking.
  - Only one configured: uses that one, template fallback if it fails.
  - None configured: honest template fallback mode everywhere, exactly
    as before.

DeepSeek's API is OpenAI-compatible (same request/response shape, same
official Python SDK, just a different base_url and model name), so it
reuses the OpenAI call path below rather than a separate implementation
— one real code path, two real providers.

Honest limitation: this sandbox cannot reach api.openai.com,
generativelanguage.googleapis.com, or api.deepseek.com (all three return
HTTP 403 through the egress proxy here, tested directly), so none of
these three paths has been exercised against a live API from within
this sandbox. Each one follows its provider's real, documented REST
contract; none has been live-tested end-to-end here for that network
reason, not because it was skipped.
"""
import os
import requests

_last_call: dict = {"live_ok": False, "error_code": None, "error": None, "provider": None}


def _openai_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _deepseek_key() -> str:
    return os.environ.get("DEEPSEEK_API_KEY", "").strip()


def _classify_error(exc: Exception, provider: str) -> str:
    msg = str(exc).lower()
    if "insufficient_quota" in msg or "exceeded your current quota" in msg or "resource_exhausted" in msg or "insufficient balance" in msg:
        return "quota_exceeded"
    if "invalid_api_key" in msg or "incorrect api key" in msg or "api_key_invalid" in msg or "api key not valid" in msg or "authentication" in msg:
        return "invalid_key"
    if "rate_limit" in msg or "429" in msg:
        return "rate_limit"
    return "api_error"


def get_ai_status() -> dict:
    """Key presence only — no probe call (avoids burning quota on every page load)."""
    openai_key, gemini_key, deepseek_key = _openai_key(), _gemini_key(), _deepseek_key()
    configured = [n for n, k in [("openai", openai_key), ("gemini", gemini_key), ("deepseek", deepseek_key)] if k]
    return {
        "openai_configured": bool(openai_key),
        "gemini_configured": bool(gemini_key),
        "deepseek_configured": bool(deepseek_key),
        "active_provider": configured[0] if configured else None,
        "fallback_providers": configured[1:] if len(configured) > 1 else [],
        "openai_model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "gemini_model": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        "deepseek_model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "ready": bool(configured),
        "live_ok": _last_call["live_ok"],
        "error_code": _last_call["error_code"],
        "error": _last_call["error"],
        "last_provider_used": _last_call["provider"],
    }


def _call_openai_compatible(system_prompt: str, user_prompt: str, max_tokens: int, api_key: str, base_url: str, model: str) -> str:
    """Real Chat Completions call against any OpenAI-compatible endpoint —
    used for both OpenAI itself and DeepSeek, which deliberately implements
    the same contract."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _call_openai(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return _call_openai_compatible(system_prompt, user_prompt, max_tokens, _openai_key(), None, model)


def _call_deepseek(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """DeepSeek's real, documented API: OpenAI-compatible, base_url
    https://api.deepseek.com, default model 'deepseek-chat'."""
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    return _call_openai_compatible(system_prompt, user_prompt, max_tokens, _deepseek_key(), "https://api.deepseek.com", model)


def _call_gemini(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """Real Google Gemini call via the documented generateContent REST
    endpoint (no SDK dependency needed — one HTTPS POST, consistent with
    SAHELI's dependency-light pattern elsewhere). Gemini has no separate
    'system' role on this endpoint; the system prompt is passed as a
    systemInstruction block, the documented way to set it."""
    key = _gemini_key()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def call_ai(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> dict:
    openai_key, gemini_key, deepseek_key = _openai_key(), _gemini_key(), _deepseek_key()
    if not openai_key and not gemini_key and not deepseek_key:
        return {"text": None, "mode": "fallback_no_key", "error": None, "error_code": "no_key"}

    # Try every configured provider in order, OpenAI first, then Gemini,
    # then DeepSeek — each one a real automatic retry of the SAME request
    # if the previous provider's call just failed.
    providers = []
    if openai_key:
        providers.append(("openai", _call_openai, "live_openai_api"))
    if gemini_key:
        providers.append(("gemini", _call_gemini, "live_gemini_api"))
    if deepseek_key:
        providers.append(("deepseek", _call_deepseek, "live_deepseek_api"))

    last_error, last_code = None, None
    for name, fn, mode in providers:
        try:
            text = fn(system_prompt, user_prompt, max_tokens)
            if text:
                _last_call.update({"live_ok": True, "error_code": None, "error": None, "provider": name})
                return {"text": text, "mode": mode, "error": None, "error_code": None}
            last_error, last_code = f"Empty response from {name}", "empty_response"
        except Exception as e:
            last_code = _classify_error(e, name)
            last_error = f"[{name}] {e}"
            continue  # real fallback: try the next configured provider before giving up

    _last_call.update({"live_ok": False, "error_code": last_code, "error": last_error, "provider": None})
    return {"text": None, "mode": "fallback_error", "error": last_error, "error_code": last_code}
