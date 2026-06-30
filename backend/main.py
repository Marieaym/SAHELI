"""
SAHELI Backend — FastAPI Application Entry Point
Run with: uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys, os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # python-dotenv not installed — rely on real environment variables (e.g. on Render/Railway)

_ai_providers = [name for name, var in [("OpenAI", "OPENAI_API_KEY"), ("Gemini", "GEMINI_API_KEY"), ("DeepSeek", "DEEPSEEK_API_KEY")] if os.environ.get(var)]
if _ai_providers:
    print(f">>> SAHELI startup: AI provider(s) configured — {', '.join(_ai_providers)}. AI Assistant will use live generation (real fallback between them if more than one).")
else:
    print(">>> SAHELI startup: NO AI provider configured — AI Assistant will run in template fallback mode.")
    print(">>> Add OPENAI_API_KEY, GEMINI_API_KEY, and/or DEEPSEEK_API_KEY to backend/.env (see backend/.env.example).")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.data_access import ensure_model_artifacts
from app.routers import auth, districts, brief, alerts, intervention, assistant, feed, scenario, pipeline, validation, causal, anomaly, forecast, cv_corn, food_security_v2, command_center

try:
    ensure_model_artifacts()
except Exception as exc:
    print(f">>> SAHELI startup: model bootstrap skipped ({exc})")

app = FastAPI(
    title="SAHELI API",
    description=(
        "Sahel Anticipatory Hub for Early-warning, Land Intelligence, and food security.\n\n"
        "Real models behind this API: an XGBoost climate-shock classifier (65k+ real daily "
        "observations), a second XGBoost model trained directly on real FEWS NET IPC ground "
        "truth (food-security-v2, r=0.62 vs real outcomes), a NumPy temporal-attention "
        "forecaster, a NumPy anomaly detector, a from-scratch CNN for corn leaf disease, real "
        "DoWhy causal inference, and a real Flower federated-learning simulation. Every "
        "endpoint below is backed by one of these, not a mock — see each router's module "
        "docstring in the source for the full honest scope and limitations of what it does."
    ),
    version="1.0.0",
    contact={"name": "SAHELI", "url": "https://github.com/Marieaym"},
)

# CORS_ALLOWED_ORIGINS env var: comma-separated list, e.g.
# "https://saheli.vercel.app,https://saheli-yourname.vercel.app"
# Falls back to "*" (any origin) for local development only.
_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(districts.router)
app.include_router(brief.router)
app.include_router(alerts.router)
app.include_router(intervention.router)
app.include_router(assistant.router)
app.include_router(feed.router)
app.include_router(scenario.router)
app.include_router(pipeline.router)
app.include_router(validation.router)
app.include_router(causal.router)
app.include_router(anomaly.router)
app.include_router(forecast.router)
app.include_router(cv_corn.router)
app.include_router(food_security_v2.router)
app.include_router(command_center.router)


@app.get("/")
def root():
    return {
        "service": "SAHELI API",
        "status": "operational",
        "docs": "/docs",
        "endpoints": [
            "/api/districts", "/api/districts/{name}", "/api/districts/{name}/history",
            "/api/zones/summary", "/api/model/metrics",
            "/api/brief/{district_name}",
            "/api/alerts/{district_name}",
            "/api/intervention/simulate",
            "/api/assistant/ask",
        ],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
