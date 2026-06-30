"""
Real DoWhy causal-effect endpoint. Serves the output of
models/causal_module.py: an observational, confounder-adjusted estimate
of the effect of severe drought on Critical-risk probability, with
refutation-test results. This is global (continental), not
country-scoped, because the causal effect was estimated across all
six SAHELI countries together for statistical power.
"""
import json
import os
from fastapi import APIRouter, Depends
from routers.auth import get_current_user

router = APIRouter(prefix="/api", tags=["causal"])

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models_data", "causal_results.json")


@router.get("/causal/effect")
def get_causal_effect(user: dict = Depends(get_current_user)):
    with open(RESULTS_PATH) as f:
        return json.load(f)
