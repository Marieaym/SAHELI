# 🌾 SAHELI — Web Application
### Sahel Anticipatory Hub for Early-warning, Land Intelligence, and food security

**Presidential African Youth in Artificial Intelligence and Robotics Competition 2026 — Presidential Award Category**

Built by **Marie Yahaya Abdou** · AI Undergraduate, African Development Universalis (ADU) · Niamey, Niger

---

## What this is

A full-stack web application: a **React frontend** and a **FastAPI backend** serving a trained XGBoost
food-security risk model across 18 Sahel districts, with ten interactive modules:

- **Agent Pipeline** — the centerpiece: watch SAHELI's five agents (Sentinel, Forecast, Explainer, Alerter, PolicyWriter) run live, in sequence, on real data for any district — data collection, model inference, AI-generated explanation, multilingual SMS alert, and a downloadable PDF brief, end to end
- **Risk Map** — live interactive map with agro-ecological zone overlays, district risk ranking, SHAP-style explainability
- **Live Alert Feed** — chronological feed of real historical risk-level transitions (not scripted)
- **Scenario Simulator ("Digital Twin")** — slider-driven rainfall shock simulation that re-runs the real trained model on adjusted climate inputs for all 18 districts
- **Compare Districts** — side-by-side risk and 12-month drought trend comparison
- **Policy Brief Generator** — generates a real downloadable PDF brief, live, per district
- **Multilingual Alert Simulator** — French / Hausa / Zarma SMS alert preview with phone mockup
- **Intervention Simulator** — live budget slider showing projected risk reduction across districts
- **AI Assistant** — conversational interface answering questions about real district data, powered by the **Anthropic Claude API** (server-side key), with a transparent fallback mode when no key is configured
- **Role-based views** — switch between Farmer / NGO Field Agent / Government Minister perspectives, each surfacing different information density, plus a global district search

A parallel **Streamlit prototype** (see `../saheli/`) ships as a guaranteed-working fallback submission.

## Authentication & Country Scoping

SAHELI requires a real account to access the dashboard. This is genuine security,
not a demo gate:

- **Registration**: email, password (bcrypt-hashed), full name, and a country
  (one of the six covered in this deployment). Returns a signed JWT.
- **Every data endpoint requires a valid JWT** (`Authorization: Bearer <token>`).
  Unauthenticated requests get `401`.
- **Every endpoint is scoped to the authenticated user's own country.** A user
  registered to Niger sees only Niger's districts, on the map, in the feed, in
  the policy briefs, in the scenario simulator — everywhere. Requesting a
  district from another country returns `403` with a clear message.
- Tokens are signed with `JWT_SECRET` (set this in production — see `.env.example`)
  and expire after 7 days.
- Persistence is a single SQLite file (`backend/app/saheli.db`, git-ignored) —
  intentionally lightweight for an MVP, real enough for genuine auth without the
  operational overhead of a separate database server.

## Two Kinds of AI in SAHELI

Worth being precise about this, since it comes up often:

1. **The predictive AI (the core of SAHELI)** — a self-trained XGBoost classifier with SHAP explainability.
   No API key, no external dependency. This is what predicts risk and drives the map, the feed, and the
   Scenario Simulator's real model re-inference. Trained entirely in-house, the same way Mama HealthID's
   FastAPI explainable-AI microservice was.
2. **The conversational AI (an optional layer)** — the Agent Explainer / AI Assistant, which uses the
   Anthropic Claude API to generate natural-language answers. This requires `ANTHROPIC_API_KEY`. Without
   it, the system falls back transparently to a rule-based summary rather than failing or pretending.

## Architecture

```
┌─────────────────────┐        ┌──────────────────────────┐
│   React Frontend      │  HTTP  │   FastAPI Backend          │
│   (Vite + Tailwind)   │ <----> │   XGBoost model + SHAP     │
│   5 interactive pages  │        │   PDF generation (ReportLab)│
│                        │        │   Claude API (assistant)   │
└─────────────────────┘        └──────────────────────────┘
```

## Repository Structure

```
saheli-web/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── app/
│   │   ├── data_access.py       # Shared model/data loading
│   │   ├── models_data/         # Trained model artifacts + scored dataset
│   │   └── routers/
│   │       ├── districts.py     # Risk map & district data endpoints
│   │       ├── brief.py         # Live PDF policy brief generation
│   │       ├── alerts.py        # Multilingual alert simulator
│   │       ├── intervention.py  # Budget allocation simulator
│   │       └── assistant.py     # AI Assistant (Claude API + fallback)
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── api/client.js        # Centralized backend API client
    │   ├── components/          # Sidebar, RiskBadge, Card, Metric
    │   └── pages/                # Overview, RiskMap, PolicyBrief, AlertSimulator,
    │                              # InterventionSimulator, Assistant
    ├── tailwind.config.js        # Sahel-rooted design tokens
    └── .env.example
```

## Run Locally

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env          # optional: add your ANTHROPIC_API_KEY to enable live AI assistant
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env.local    # defaults to localhost:8000, edit if backend is elsewhere
npm run dev
```

Visit `http://localhost:5173`.

## Deployment

**Backend → Render (or Railway):**
1. Push this repo to GitHub
2. Create a new Web Service on [render.com](https://render.com), point it at `backend/`
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable `ANTHROPIC_API_KEY` (optional — enables live AI assistant)

**Frontend → Vercel:**
1. Import the repo on [vercel.com](https://vercel.com), set root directory to `frontend/`
2. Add environment variable `VITE_API_URL` = your deployed Render backend URL
3. Deploy

## Design System

Palette and typography are deliberately rooted in the Sahel rather than generic "AI dashboard" defaults:
deep indigo night-sky background, millet-gold signature accent, terracotta/amber/acacia risk semantics,
Space Grotesk display type, Inter body type, IBM Plex Mono for data readouts.

## How SAHELI Compares to Existing Systems

SAHELI does not compete with FEWS NET or WFP HungerMap Live — both are real, valuable,
institutionally-run systems. SAHELI's differentiation is specific and verifiable:

| Capability | FEWS NET | WFP HungerMap | SAHELI |
|---|---|---|---|
| Resolution | Country/region | Country | District |
| Agro-ecological zone adaptation | No | No | Yes (4 zones) |
| Last-mile local-language alerts | No | No | Yes (Hausa, Zarma, French SMS) |
| Decision-ready PDF brief, auto-generated | No | No | Yes |
| Interactive "what-if" scenario simulation | No | No | Yes (real model re-inference) |
| Per-country data sovereignty (own login, own scope) | N/A | N/A | Yes |

SAHELI is positioned as complementary infrastructure — the layer between institutional
forecasting and the local action it should trigger.

## Sustainability Beyond the Competition

This is a roadmap, not a claim of existing partnerships:

- **Data costs**: all current data sources (Open-Meteo/ERA5) are free and open; no
  recurring licensing cost at this scale.
- **Hosting**: the current footprint (SQLite + single backend instance) runs on
  free/low-cost tiers (Render, Vercel) well beyond the competition.
- **Path to real-world use**: the realistic next step is a pilot conversation with an
  existing operational partner — for SAHELI, that is Protra Niger, already a real
  contact through the author's prior work, not a fabricated endorsement.
- **Model honesty**: the current model's headline accuracy reflects internal
  consistency against a rule-derived label (see note above), not validation against
  real historical IPC/FEWS NET crisis declarations. That validation is the explicit
  next milestone, not a claim made today.



## Important Notes

- **Probability columns bug fix:** an earlier version of the model-scoring script mislabeled risk
  probability columns due to `LabelEncoder`'s alphabetical class sorting. This has been corrected —
  probability columns now map explicitly via `le.classes_`. See `models/train_model.py` in the
  Streamlit prototype repo for the fix and explanation.
- **AI Assistant fallback:** if `ANTHROPIC_API_KEY` is not set, `/api/assistant/ask` returns a clearly
  labeled rule-based summary instead of silently failing or pretending to be the LLM.
- **Map tiles & fonts:** loaded from CartoDB and Google Fonts CDNs at runtime — require a normal
  internet connection in the browser (not an issue in production deployment).

## License

MIT License — see [LICENSE](../saheli/LICENSE)

---

*"Africa does not need to wait for external solutions. SAHELI is the proof."*
