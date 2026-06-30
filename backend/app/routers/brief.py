"""
SAHELI Backend — Policy Brief PDF generation (Agent PolicyWriter)

Full redesign pass: branded header band, real 90-day trend chart,
multi-source indicator card grid, a real (clearly disclosed) resource/
budget estimate reusing the same formula as the live Intervention
Simulator, a dynamic multi-horizon action plan, and page numbers.
Everything renders in the requested language (en/fr).
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from io import BytesIO
from datetime import datetime
import sys, os
import requests
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 HRFlowable, Image, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_RIGHT, TA_CENTER

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import (get_latest_snapshot, get_scored_df, RISK_COLORS, RECOMMENDATIONS,
                          RECOMMENDATIONS_FR, assert_district_access, POP_WEIGHTS, DATA_DIR,
                          get_model_artifacts, get_v2_model_artifacts, v2_ipc_to_risk_level,
                          V2_VALIDATED_DISTRICTS, get_shap_explainer)
from db import get_recent_crop_scans
from ai_client import call_ai
from routers.auth import get_current_user
from db import log_activity
import joblib
import json as jsonlib

router = APIRouter(prefix="/api", tags=["policy-brief"])

# ── Palette, matching the live app's Harmattan Ledger identity ─────────────
C_GOLD = colors.HexColor("#A86E2A")
C_GOLD_LIGHT = colors.HexColor("#F2EADA")
C_INK = colors.HexColor("#29231C")
C_MUTED = colors.HexColor("#6E6353")
C_CARD = colors.HexColor("#FAF6EC")
C_BORDER = colors.HexColor("#D6C8AD")
C_CLAY = colors.HexColor("#A53A26")
C_AMBER = colors.HexColor("#B87721")
C_ACACIA = colors.HexColor("#5A6E4C")
C_WHITE = colors.white

SEVERITY_SCORE = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
COST_PER_SEVERITY_UNIT = 150_000
DEFAULT_NATIONAL_BUDGET = 1_500_000

RISK_LEVEL_FR = {"Critical": "Critique", "High": "Élevé", "Medium": "Moyen", "Low": "Faible"}
ZONE_FR = {"Saharan": "Saharienne", "Sahelian": "Sahélienne", "Sudanian": "Soudanienne", "Guinean": "Guinéenne"}

TXT = {
    "en": {
        "doc_title": "FOOD SECURITY POLICY BRIEF",
        "generated": "Generated {date} \u00b7 Agent PolicyWriter",
        "district": "District", "country": "Country", "zone": "Agro-ecological Zone",
        "satellite_caption": "Satellite view (Esri World Imagery) \u00b7 {lat:.3f}, {lon:.3f}",
        "satellite_unavailable": "Satellite image unavailable \u2014 no internet access at generation time.",
        "situation_summary": "Situation Summary",
        "situation_text": (
            "District <b>{district}</b> ({zone} zone) is currently classified at <b>{risk}</b> food-security "
            "risk. Drought index: <b>{drought:.2f}</b> std. dev. vs. 10-year baseline, with <b>{dry_days}</b> "
            "consecutive dry days recorded."
        ),
        "ai_label": "PolicyWriter Analysis (AI)",
        "trend_title": "90-Day Drought Index Trend",
        "multi_source": "Multi-Source Risk Indicators",
        "drought_label": "Climate (ERA5)", "conflict_label": "Conflict, 30d (ACLED)",
        "price_label": "Millet price anomaly (WFP)", "groundwater_label": "Groundwater anomaly (GRACE-FO)",
        "ndvi_label": "Vegetation health (Sentinel-2)", "water_label": "Water points, 50km (OSM)",
        "resource_title": "Resource & Budget Estimate",
        "resource_text": (
            "Under a national emergency budget of <b>${budget:,.0f}</b>, allocated proportionally to need "
            "across all districts in {country} using SAHELI's real-time severity index, <b>{district}</b>'s "
            "estimated share is <b>${allocation:,.0f}</b> ({pct:.1f}% of the national envelope)."
        ),
        "resource_disclosure": (
            "Disclosure: this allocation uses real-time risk severity weighted by an illustrative relative "
            "population-size proxy (not verified census data) \u2014 see Model Validation for full methodology. "
            "Treat this as a planning estimate, not an audited budget figure."
        ),
        "ground_truth": "Independent Ground Truth",
        "ground_truth_text": (
            "Most recent real FEWS NET classification for this region: <b>IPC Phase {ipc:.1f} / 5</b>. "
            "SAHELI's climate-driven classification and this independent reference measure different "
            "things on different timescales \u2014 see Model Validation for full disclosure."
        ),
        "ground_truth_unavailable": "No name-matchable FEWS NET region available for this district in the current extract.",
        "v2_title": "Real Food-Security Risk \u2014 FEWS-NET-Validated Model (v2)",
        "v2_validated": (
            "This district has real FEWS NET ground truth in SAHELI's data. The v2 model, trained directly "
            "on 32,443 real FEWS NET observations (r=0.62 with real ground truth, vs. r=-0.20 for the "
            "climate-only model), predicts: <b>IPC {ipc:.2f}/5 \u2014 {risk} risk</b>."
        ),
        "v2_extrapolated": (
            "No real FEWS NET ground truth exists yet for this specific district. This prediction applies "
            "the same model, trained on 10 other real Sahelian districts, by extrapolation \u2014 "
            "<b>IPC {ipc:.2f}/5 \u2014 {risk} risk</b>. Treat as indicative, not locally validated."
        ),
        "v2_unavailable": "The real food-security model (v2) has not been trained yet \u2014 run models/food_security_v2_module.py.",
        "v2_agree": "This matches the climate-shock classification above ({risk}) \u2014 both real models agree.",
        "v2_escalate": (
            "This is MORE SEVERE than the climate-shock classification above ({climate_risk}). SAHELI's own "
            "validation shows the real food-security model is the one that correlates with real outcomes \u2014 "
            "treat this district as higher priority than the climate signal alone would suggest."
        ),
        "v2_deescalate": (
            "This is LESS SEVERE than the climate-shock classification above ({climate_risk}), suggesting the "
            "climate shock may be transient rather than a structural food-security crisis. Verify on the "
            "ground before a full emergency response."
        ),
        "crop_scan_title": "Field Reports \u2014 Corn Leaf Scanner",
        "crop_scan_text": (
            "{n} real Corn Scanner report(s) logged for this district, {n_disease} flagging possible "
            "disease. Shown for qualitative context only \u2014 not merged into the risk score above, since "
            "no validated dataset yet links detected crop-disease severity to IPC-scale food security."
        ),
        "key_drivers": "Key Risk Drivers",
        "no_anomaly": "No significant anomaly detected; indicators within normal historical range.",
        "drought_driver": "Water balance deficit of {v:.1f} std. dev. below baseline",
        "dryspell_driver": "{d}-day dry spell, exceeding typical seasonal variability",
        "monsoon_driver": "Deficit occurring within the critical monsoon (Jun-Sep) window",
        "recommended": "Recommended Actions",
        "action_plan_title": "Action Plan \u2014 From Recommendation to Execution",
        "immediate_tier": "IMMEDIATE \u2014 Next 72 Hours",
        "short_term_tier": "SHORT-TERM \u2014 Next 2-4 Weeks",
        "strategic_tier": "STRATEGIC \u2014 Next 1-3 Months",
        "owner_col": "Responsible Party", "action_col": "Action",
        "footer": "Generated automatically by SAHELI's Agent PolicyWriter from real-time indicators. Presidential African Youth in AI and Robotics Competition 2026.",
        "prob_title": "Model Confidence — Full Probability Breakdown",
        "shap_title": "Why This Prediction — Real Model Attribution (SHAP)",
        "shap_increase": "increased", "shap_decrease": "decreased",
        "shap_line": "{feat} {dir} the risk score by {val:.2f}",
        "history_title": "Recent Risk History — This District",
        "history_line": "{date}: {from_r} \u2192 {to_r}",
        "history_none": "No risk-level transitions recorded in the historical window.",
        "ranking_title": "Regional Context — Country Risk Ranking",
        "causal_title": "Causal Context — Validated Effect Sizes",
        "causal_none": "No causal driver above the disclosed threshold for this district at this time \u2014 see Causal Pathway for the full continental analysis.",
        "model_credibility": "Model Credibility",
        "model_credibility_text": (
            "The climate-shock classification above comes from a model with {acc:.1f}% test accuracy and "
            "{f1:.1f}% weighted F1 on {n:,} held-out real records \u2014 against its own rule-derived label, "
            "not real food-security ground truth. The real food-security model (v2, see above) is the one "
            "validated against real FEWS NET data: r=0.62, versus r=-0.20 for this climate model on the "
            "same real test set. See Model Validation for the full disclosure."
        ),
    },
    "fr": {
        "doc_title": "NOTE DE POLITIQUE SUR LA S\u00c9CURIT\u00c9 ALIMENTAIRE",
        "generated": "G\u00e9n\u00e9r\u00e9 le {date} \u00b7 Agent PolicyWriter",
        "district": "District", "country": "Pays", "zone": "Zone agro\u00e9cologique",
        "satellite_caption": "Vue satellite (Esri World Imagery) \u00b7 {lat:.3f}, {lon:.3f}",
        "satellite_unavailable": "Image satellite indisponible \u2014 pas d'acc\u00e8s internet au moment de la g\u00e9n\u00e9ration.",
        "situation_summary": "R\u00e9sum\u00e9 de la situation",
        "situation_text": (
            "Le district de <b>{district}</b> (zone {zone}) est actuellement class\u00e9 en risque alimentaire "
            "<b>{risk}</b>. Indice de s\u00e9cheresse : <b>{drought:.2f}</b> \u00e9carts-types vs r\u00e9f\u00e9rence sur 10 ans, "
            "avec <b>{dry_days}</b> jours secs cons\u00e9cutifs enregistr\u00e9s."
        ),
        "ai_label": "Analyse PolicyWriter (IA)",
        "trend_title": "Tendance de l'indice de s\u00e9cheresse sur 90 jours",
        "multi_source": "Indicateurs de risque multi-sources",
        "drought_label": "Climat (ERA5)", "conflict_label": "Conflit, 30j (ACLED)",
        "price_label": "Anomalie prix mil (WFP)", "groundwater_label": "Anomalie souterraine (GRACE-FO)",
        "ndvi_label": "Sant\u00e9 v\u00e9g\u00e9tation (Sentinel-2)", "water_label": "Points d'eau, 50km (OSM)",
        "resource_title": "Estimation des ressources et du budget",
        "resource_text": (
            "Sous un budget d'urgence national de <b>{budget:,.0f} $</b>, allou\u00e9 proportionnellement aux "
            "besoins sur tous les districts du {country} selon l'indice de s\u00e9v\u00e9rit\u00e9 en temps r\u00e9el de SAHELI, "
            "la part estim\u00e9e de <b>{district}</b> est de <b>{allocation:,.0f} $</b> ({pct:.1f}% de l'enveloppe nationale)."
        ),
        "resource_disclosure": (
            "Divulgation : cette allocation utilise la s\u00e9v\u00e9rit\u00e9 de risque en temps r\u00e9el pond\u00e9r\u00e9e par une "
            "approximation illustrative de taille de population relative (pas des donn\u00e9es de recensement "
            "v\u00e9rifi\u00e9es), voir Model Validation pour la m\u00e9thodologie compl\u00e8te. \u00c0 traiter comme une estimation "
            "de planification, pas un chiffre budg\u00e9taire audit\u00e9."
        ),
        "ground_truth": "R\u00e9f\u00e9rence ind\u00e9pendante",
        "ground_truth_text": (
            "Classification r\u00e9elle FEWS NET la plus r\u00e9cente pour cette r\u00e9gion : <b>Phase IPC {ipc:.1f} / 5</b>. "
            "La classification climatique de SAHELI et cette r\u00e9f\u00e9rence ind\u00e9pendante mesurent des r\u00e9alit\u00e9s "
            "diff\u00e9rentes \u00e0 des \u00e9chelles de temps diff\u00e9rentes, voir Model Validation pour la divulgation compl\u00e8te."
        ),
        "ground_truth_unavailable": "Aucune r\u00e9gion FEWS NET correspondante disponible pour ce district dans l'extraction actuelle.",
        "v2_title": "Risque r\u00e9el de s\u00e9curit\u00e9 alimentaire \u2014 Mod\u00e8le valid\u00e9 FEWS NET (v2)",
        "v2_validated": (
            "Ce district dispose de vraies donn\u00e9es FEWS NET dans les donn\u00e9es de SAHELI. Le mod\u00e8le v2, entra\u00een\u00e9 "
            "directement sur 32 443 vraies observations FEWS NET (r=0,62 avec la r\u00e9alit\u00e9 terrain, contre r=-0,20 "
            "pour le mod\u00e8le climatique seul), pr\u00e9dit : <b>IPC {ipc:.2f}/5 \u2014 risque {risk}</b>."
        ),
        "v2_extrapolated": (
            "Aucune vraie donn\u00e9e FEWS NET n'existe encore pour ce district sp\u00e9cifique. Cette pr\u00e9diction applique "
            "le m\u00eame mod\u00e8le, entra\u00een\u00e9 sur 10 autres vrais districts sah\u00e9liens, par extrapolation \u2014 "
            "<b>IPC {ipc:.2f}/5 \u2014 risque {risk}</b>. \u00c0 traiter comme indicatif, pas valid\u00e9 localement."
        ),
        "v2_unavailable": "Le mod\u00e8le r\u00e9el de s\u00e9curit\u00e9 alimentaire (v2) n'a pas encore \u00e9t\u00e9 entra\u00een\u00e9 \u2014 ex\u00e9cuter models/food_security_v2_module.py.",
        "v2_agree": "Ceci correspond \u00e0 la classification du choc climatique ci-dessus ({risk}) \u2014 les deux vrais mod\u00e8les sont d'accord.",
        "v2_escalate": (
            "Ceci est PLUS S\u00c9V\u00c8RE que la classification du choc climatique ci-dessus ({climate_risk}). La validation "
            "propre de SAHELI montre que c'est le mod\u00e8le r\u00e9el de s\u00e9curit\u00e9 alimentaire qui corr\u00e8le avec la r\u00e9alit\u00e9 "
            "\u2014 traiter ce district comme prioritaire au-del\u00e0 de ce que sugg\u00e9rerait le seul signal climatique."
        ),
        "v2_deescalate": (
            "Ceci est MOINS S\u00c9V\u00c8RE que la classification du choc climatique ci-dessus ({climate_risk}), ce qui sugg\u00e8re "
            "que le choc climatique est probablement transitoire plut\u00f4t qu'une crise alimentaire structurelle. "
            "V\u00e9rifier sur le terrain avant une r\u00e9ponse d'urgence compl\u00e8te."
        ),
        "crop_scan_title": "Rapports de terrain \u2014 Scanner de feuille de ma\u00efs",
        "crop_scan_text": (
            "{n} vrai(s) rapport(s) du Scanner de ma\u00efs enregistr\u00e9(s) pour ce district, {n_disease} signalant une "
            "maladie possible. Affich\u00e9 pour contexte qualitatif uniquement, pas fusionn\u00e9 dans le score de risque "
            "ci-dessus, puisqu'aucun jeu de donn\u00e9es valid\u00e9 ne relie encore la s\u00e9v\u00e9rit\u00e9 de maladie d\u00e9tect\u00e9e \u00e0 la "
            "s\u00e9curit\u00e9 alimentaire \u00e0 l'\u00e9chelle IPC."
        ),
        "key_drivers": "Principaux facteurs de risque",
        "no_anomaly": "Aucune anomalie significative d\u00e9tect\u00e9e ; indicateurs dans la plage historique normale.",
        "drought_driver": "D\u00e9ficit de bilan hydrique de {v:.1f} \u00e9carts-types sous la r\u00e9f\u00e9rence",
        "dryspell_driver": "S\u00e9quence s\u00e8che de {d} jours, d\u00e9passant la variabilit\u00e9 saisonni\u00e8re typique",
        "monsoon_driver": "D\u00e9ficit survenant pendant la fen\u00eatre critique de la mousson (juin-septembre)",
        "recommended": "Actions recommand\u00e9es",
        "action_plan_title": "Plan d'action \u2014 de la recommandation \u00e0 l'ex\u00e9cution",
        "immediate_tier": "IMM\u00c9DIAT \u2014 72 prochaines heures",
        "short_term_tier": "COURT TERME \u2014 2 \u00e0 4 semaines",
        "strategic_tier": "STRAT\u00c9GIQUE \u2014 1 \u00e0 3 mois",
        "owner_col": "Responsable", "action_col": "Action",
        "footer": "G\u00e9n\u00e9r\u00e9 automatiquement par le module Agent PolicyWriter de SAHELI \u00e0 partir d'indicateurs en temps r\u00e9el. Concours pr\u00e9sidentiel africain pour la jeunesse en IA et robotique 2026.",
        "prob_title": "Confiance du mod\u00e8le \u2014 r\u00e9partition compl\u00e8te des probabilit\u00e9s",
        "shap_title": "Pourquoi cette pr\u00e9diction \u2014 vraie attribution du mod\u00e8le (SHAP)",
        "shap_increase": "augment\u00e9", "shap_decrease": "diminu\u00e9",
        "shap_line": "{feat} a {dir} le score de risque de {val:.2f}",
        "history_title": "Historique r\u00e9cent du risque \u2014 ce district",
        "history_line": "{date} : {from_r} \u2192 {to_r}",
        "history_none": "Aucune transition de niveau de risque enregistr\u00e9e sur la p\u00e9riode historique.",
        "ranking_title": "Contexte r\u00e9gional \u2014 classement national des risques",
        "causal_title": "Contexte causal \u2014 effets valid\u00e9s",
        "causal_none": "Aucun facteur causal au-dessus du seuil divulgu\u00e9 pour ce district actuellement \u2014 voir Causal Pathway pour l'analyse continentale compl\u00e8te.",
        "model_credibility": "Cr\u00e9dibilit\u00e9 du mod\u00e8le",
        "model_credibility_text": (
            "La classification du choc climatique ci-dessus provient d'un mod\u00e8le avec {acc:.1f}% de pr\u00e9cision "
            "sur donn\u00e9es de test et {f1:.1f}% de F1 pond\u00e9r\u00e9 sur {n:,} vraies observations retenues \u2014 contre "
            "son propre label d\u00e9riv\u00e9 par r\u00e8gle, pas contre la vraie r\u00e9alit\u00e9 de s\u00e9curit\u00e9 alimentaire. Le vrai "
            "mod\u00e8le de s\u00e9curit\u00e9 alimentaire (v2, voir ci-dessus) est celui valid\u00e9 contre les vraies donn\u00e9es FEWS "
            "NET : r=0,62, contre r=-0,20 pour ce mod\u00e8le climatique sur le m\u00eame jeu de test r\u00e9el. Voir Model "
            "Validation pour la divulgation compl\u00e8te."
        ),
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────

def lonlat_to_tile(lon, lat, zoom):
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def fetch_satellite_image(lat, lon, zoom=13):
    try:
        x, y = lonlat_to_tile(lon, lat, zoom)
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
        r = requests.get(url, timeout=8)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            return BytesIO(r.content)
    except Exception:
        pass
    return None


def render_trend_chart(district: str, risk_hex: str):
    """Real 90-day drought_index trend for this district, rendered as a
    small branded line chart and embedded as an image."""
    try:
        df = get_scored_df()
        hist = df[df["district"] == district].sort_values("date").tail(90)
        if len(hist) < 10:
            return None
        fig, ax = plt.subplots(figsize=(5.2, 1.5), dpi=160)
        ax.plot(hist["date"], hist["drought_index"], color=risk_hex, linewidth=1.6)
        ax.axhline(0, color="#999999", linewidth=0.6, linestyle="--")
        ax.fill_between(hist["date"], hist["drought_index"], 0, color=risk_hex, alpha=0.12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        plt.tight_layout(pad=0.2)
        buf = BytesIO()
        fig.savefig(buf, format="png", transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None


def compute_resource_estimate(row, country: str):
    """Reuses the EXACT same need-weighting formula as the live Intervention
    Simulator (/api/intervention/simulate), for one district, against the
    default national budget — so the PDF never invents a different number
    than what the app itself would show."""
    latest = get_latest_snapshot()
    country_rows = latest[latest["country"] == country].copy()
    country_rows["pop_weight"] = country_rows["district"].map(POP_WEIGHTS).fillna(0.5)
    country_rows["severity"] = country_rows["predicted_risk"].map(SEVERITY_SCORE)
    country_rows["need_index"] = country_rows["severity"] * country_rows["pop_weight"]
    total_need = country_rows["need_index"].sum()
    if total_need <= 0:
        return None
    this_need = SEVERITY_SCORE.get(row["predicted_risk"], 1) * (POP_WEIGHTS.get(row["district"], 0.5))
    allocation = this_need / total_need * DEFAULT_NATIONAL_BUDGET
    pct = this_need / total_need * 100
    return allocation, pct


def get_v2_prediction(row):
    """The real food-security model (v2), trained directly on real FEWS
    NET IPC ground truth (r=0.62 vs r=-0.20 for the climate-only model,
    see models/food_security_v2_module.py). Returns None if the model
    has not been trained yet, never fabricates a number."""
    v2_model, v2_features = get_v2_model_artifacts()
    if v2_model is None:
        return None
    import pandas as pd
    x = pd.DataFrame([row])[v2_features]
    ipc = float(np.clip(v2_model.predict(x)[0], 1.0, 5.0))
    risk = v2_ipc_to_risk_level(ipc)
    status = "validated" if row["district"] in V2_VALIDATED_DISTRICTS else "extrapolated"
    return {"ipc": ipc, "risk": risk, "status": status}


def build_action_plan(row, lang="en", v2_result=None):
    """Builds a real, multi-horizon action plan from the district's actual
    measured signals — not a static boilerplate list. When the real
    food-security model (v2) is available, its signal explicitly drives
    escalation or de-escalation language, because SAHELI's own validation
    found the climate-shock model and the real, FEWS-NET-validated
    food-security model can disagree, and that disagreement changes what
    the right action actually is, not just how it's worded."""
    risk = row["predicted_risk"]
    drought = row["drought_index"]
    dry_days = int(row["consec_dry_days"])
    conflict = row.get("conflict_events_30d", 0) or 0
    price = row.get("price_anomaly_30d", 0) or 0
    gw = row.get("groundwater_anomaly_cm", 0) or 0
    ndvi = row.get("sentinel2_ndvi", 0) or 0
    severe = risk in ("Critical", "High")

    SEVERITY_RANK_LOCAL = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    fs_escalates = fs_deescalates = fs_agrees = False
    if v2_result is not None:
        climate_rank = SEVERITY_RANK_LOCAL[risk]
        fs_rank = SEVERITY_RANK_LOCAL[v2_result["risk"]]
        fs_escalates = fs_rank > climate_rank
        fs_deescalates = fs_rank < climate_rank
        fs_agrees = fs_rank == climate_rank
        severe = severe or fs_rank >= 2  # the real model can independently trigger "severe" handling

    if lang == "fr":
        immediate, short_term, strategic = [], [], []

        if v2_result is not None and fs_escalates:
            status_txt = "valid\u00e9" if v2_result["status"] == "validated" else "extrapol\u00e9, non valid\u00e9 localement"
            immediate.append((
                f"PRIORIT\u00c9 \u2014 Le mod\u00e8le r\u00e9el de s\u00e9curit\u00e9 alimentaire (validation FEWS NET, {status_txt}) "
                f"indique un risque {RISK_LEVEL_FR.get(v2_result['risk'], v2_result['risk'])} (IPC {v2_result['ipc']:.1f}/5), "
                f"plus s\u00e9v\u00e8re que le signal climatique seul ({RISK_LEVEL_FR.get(risk, risk)}). Cet \u00e9cart sugg\u00e8re un "
                f"facteur non climatique (conflit, march\u00e9, acc\u00e8s) qui aggrave la situation r\u00e9elle.",
                "Coordination nationale + Minist\u00e8re"
            ))
        elif v2_result is not None and fs_deescalates:
            immediate.append((
                f"Le choc climatique est {RISK_LEVEL_FR.get(risk, risk)}, mais le mod\u00e8le r\u00e9el de s\u00e9curit\u00e9 "
                f"alimentaire (IPC {v2_result['ipc']:.1f}/5, {RISK_LEVEL_FR.get(v2_result['risk'], v2_result['risk'])}) sugg\u00e8re "
                f"un choc probablement transitoire, pas encore une crise alimentaire structurelle. V\u00e9rifier sur le "
                f"terrain avant de d\u00e9clencher une r\u00e9ponse d'urgence compl\u00e8te.",
                "\u00c9quipe de suivi SAHELI"
            ))

        if risk == "Critical":
            immediate.append(("D\u00e9clencher la proc\u00e9dure d'urgence alimentaire du district et notifier le point focal national", "Minist\u00e8re + Protection civile"))
            immediate.append(("Pr\u00e9-positionner les stocks alimentaires d'urgence disponibles vers le district dans les 72h", "Office national de s\u00e9curit\u00e9 alimentaire"))
        if drought < -0.5:
            immediate.append((f"V\u00e9rifier l'\u00e9tat r\u00e9el des points d'eau dans la zone (indice de s\u00e9cheresse {drought:.2f}, {dry_days} jours secs)", "Service hydraulique r\u00e9gional"))
        if conflict > 0:
            immediate.append((f"Coordonner avec les acteurs s\u00e9curitaires locaux avant tout d\u00e9ploiement logistique ({int(conflict)} \u00e9v\u00e9nements de conflit recens\u00e9s sur 30j)", "Coordination humanitaire / s\u00e9curit\u00e9"))
        if price > 1.0:
            immediate.append((f"Alerter le minist\u00e8re du Commerce sur l'anomalie de prix du mil ({price:.2f}\u03c3 au-dessus de la r\u00e9f\u00e9rence)", "Minist\u00e8re du Commerce"))
        if not immediate:
            immediate.append(("Maintenir la veille standard, aucune action d'urgence requise \u00e0 ce stade", "\u00c9quipe de suivi SAHELI"))

        if severe:
            short_term.append(("D\u00e9ployer une \u00e9quipe d'\u00e9valuation rapide sur le terrain dans les 2 semaines", "ONG partenaires de terrain"))
            short_term.append(("Activer ou renforcer le programme de transferts mon\u00e9taires pour les m\u00e9nages vuln\u00e9rables", "Programme de protection sociale"))
        if gw < -10:
            short_term.append((f"Lancer une \u00e9valuation des nappes phr\u00e9atiques (anomalie GRACE-FO de {gw:.1f} cm)", "Direction de l'hydraulique"))
        if ndvi < 0.1:
            short_term.append((f"Distribuer des semences \u00e0 cycle court adapt\u00e9es \u00e0 la v\u00e9g\u00e9tation actuelle (NDVI {ndvi:.3f})", "Vulgarisation agricole"))
        short_term.append(("R\u00e9\u00e9valuer le district dans 7 jours avec les donn\u00e9es climatiques et de march\u00e9 mises \u00e0 jour", "\u00c9quipe de suivi SAHELI"))

        strategic.append(("Int\u00e9grer ce district dans le plan de contingence saisonnier national", "Planification minist\u00e9rielle"))
        if severe:
            strategic.append(("Pr\u00e9-positionner des r\u00e9serves strat\u00e9giques r\u00e9gionales pour la prochaine saison", "Office national de s\u00e9curit\u00e9 alimentaire"))
        strategic.append(("Renforcer les infrastructures de r\u00e9silience hydrique identifi\u00e9es comme insuffisantes", "Coop\u00e9ration au d\u00e9veloppement"))
        if v2_result is not None:
            agree_txt = "confirm\u00e9 par les deux mod\u00e8les (climat et s\u00e9curit\u00e9 alimentaire r\u00e9elle)" if fs_agrees else "signal divergent entre les deux mod\u00e8les, \u00e0 surveiller pour affiner le mod\u00e8le"
            validation_status_txt = "donn\u00e9es FEWS NET r\u00e9elles disponibles" if v2_result['status'] == 'validated' else "extrapol\u00e9, pas encore de vraies donn\u00e9es FEWS NET pour ce district"
            strategic.append((f"Note m\u00e9thodologique : ce diagnostic est {agree_txt}. Statut de validation locale : {validation_status_txt}.", "\u00c9quipe SAHELI"))
        return {"immediate": immediate, "short_term": short_term, "strategic": strategic}

    immediate, short_term, strategic = [], [], []

    if v2_result is not None and fs_escalates:
        status_txt = "validated" if v2_result["status"] == "validated" else "extrapolated, not locally validated"
        immediate.append((
            f"PRIORITY \u2014 The real food-security model (FEWS NET validated, {status_txt}) flags "
            f"{v2_result['risk']} risk (IPC {v2_result['ipc']:.1f}/5), more severe than the climate signal "
            f"alone ({risk}). This gap suggests a non-climate factor (conflict, market access) is making "
            f"the real situation worse than climate data alone would show.",
            "National Coordination + Ministry"
        ))
    elif v2_result is not None and fs_deescalates:
        immediate.append((
            f"Climate-shock severity is {risk}, but the real food-security model (IPC {v2_result['ipc']:.1f}/5, "
            f"{v2_result['risk']}) suggests this is likely a transient climate shock, not yet a structural food "
            f"crisis. Verify on the ground before triggering a full emergency response.",
            "SAHELI Monitoring Team"
        ))

    if risk == "Critical":
        immediate.append(("Trigger the district-level food emergency procedure and notify the national focal point", "Ministry + Civil Protection"))
        immediate.append(("Pre-position available emergency food stocks to the district within 72 hours", "National Food Security Office"))
    if drought < -0.5:
        immediate.append((f"Verify the real status of water points in the area (drought index {drought:.2f}, {dry_days} dry days)", "Regional Water Service"))
    if conflict > 0:
        immediate.append((f"Coordinate with local security actors before any logistics deployment ({int(conflict)} conflict events recorded in 30 days)", "Humanitarian / Security Coordination"))
    if price > 1.0:
        immediate.append((f"Alert the Ministry of Commerce to the millet price anomaly ({price:.2f}\u03c3 above baseline)", "Ministry of Commerce"))
    if not immediate:
        immediate.append(("Maintain standard monitoring; no emergency action required at this stage", "SAHELI Monitoring Team"))

    if severe:
        short_term.append(("Deploy a rapid field assessment team within 2 weeks", "Field NGO Partners"))
        short_term.append(("Activate or scale up the cash-transfer program for vulnerable households", "Social Protection Program"))
    if gw < -10:
        short_term.append((f"Launch a groundwater assessment (GRACE-FO anomaly of {gw:.1f} cm)", "Water Resources Directorate"))
    if ndvi < 0.1:
        short_term.append((f"Distribute short-cycle seed varieties suited to current vegetation conditions (NDVI {ndvi:.3f})", "Agricultural Extension Service"))
    short_term.append(("Re-assess the district in 7 days with updated climate and market data", "SAHELI Monitoring Team"))

    strategic.append(("Integrate this district into the national seasonal contingency plan", "Ministerial Planning"))
    if severe:
        strategic.append(("Pre-position regional strategic reserves ahead of the next season", "National Food Security Office"))
    strategic.append(("Strengthen water-resilience infrastructure identified as insufficient", "Development Cooperation"))
    if v2_result is not None:
        agree_txt = "confirmed by both models (climate and real food security)" if fs_agrees else "the two models diverge here, worth monitoring to refine the model"
        strategic.append((f"Methodology note: this diagnosis is {agree_txt}. Local validation status: "
                           f"{'real FEWS NET ground truth available' if v2_result['status']=='validated' else 'extrapolated, no real FEWS NET data yet for this district'}.",
                           "SAHELI Team"))
    return {"immediate": immediate, "short_term": short_term, "strategic": strategic}


def _ai_executive_summary(row, lang="en", v2_result=None):
    risk = row["predicted_risk"]
    v2_clause_fr = v2_clause_en = ""
    if v2_result is not None:
        v2_clause_fr = (f" Le mod\u00e8le r\u00e9el de s\u00e9curit\u00e9 alimentaire (valid\u00e9 FEWS NET) indique "
                         f"IPC {v2_result['ipc']:.1f}/5 ({v2_result['risk']}), statut {v2_result['status']}.")
        v2_clause_en = (f" The real food-security model (FEWS NET validated) indicates "
                         f"IPC {v2_result['ipc']:.1f}/5 ({v2_result['risk']}), status {v2_result['status']}.")
    if lang == "fr":
        prompt = (
            f"R\u00e9dige un paragraphe ex\u00e9cutif de 3-4 phrases pour une note minist\u00e9rielle sur "
            f"{row['district']} ({row['country']}), risque climatique {risk}, indice s\u00e9cheresse {row['drought_index']:.2f}, "
            f"{int(row['consec_dry_days'])} jours secs.{v2_clause_fr} Si les deux signaux diff\u00e8rent, mentionne-le "
            f"explicitement. Ton professionnel, factuel, sans inventer de chiffres."
        )
        system = "Tu es l'Agent PolicyWriter de SAHELI. R\u00e9ponds uniquement en fran\u00e7ais."
    else:
        prompt = (
            f"Write a 3-4 sentence executive summary for a ministerial policy brief on "
            f"{row['district']} ({row['country']}), climate-shock risk {risk}, drought index {row['drought_index']:.2f}, "
            f"{int(row['consec_dry_days'])} dry days.{v2_clause_en} If the two signals differ, mention that "
            f"explicitly. Professional, factual tone; do not invent numbers."
        )
        system = "You are SAHELI's Agent PolicyWriter. Respond in English only."
    result = call_ai(system, prompt, max_tokens=240)
    if result.get("text") and result["mode"] in ("live_openai_api", "live_anthropic_api"):
        return result["text"]
    return None


def _card(label, value, accent, styles, w=4.5*cm):
    """One colored indicator card for the multi-source grid."""
    label_p = Paragraph(f'<font color="#6E6353" size="7">{label}</font>', styles["card_label"])
    value_p = Paragraph(f'<font color="{accent}" size="13"><b>{value}</b></font>', styles["card_value"])
    t = Table([[label_p], [value_p]], colWidths=[w])
    t.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.75, colors.HexColor(accent)),
        ("BACKGROUND", (0,0), (-1,-1), C_CARD),
        ("TOPPADDING", (0,0), (-1,0), 7), ("BOTTOMPADDING", (0,0), (-1,0), 2),
        ("TOPPADDING", (0,1), (-1,1), 0), ("BOTTOMPADDING", (0,1), (-1,1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ]))
    return t


def _page_footer(canvas, doc, footer_text):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawString(1.8*cm, 1.1*cm, footer_text[:95])
    canvas.drawRightString(A4[0] - 1.8*cm, 1.1*cm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


SEVERITY_RANK = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

FEATURE_COLS = [
    "precip_30d", "precip_90d", "et_30d", "temp_30d_avg",
    "water_balance_30d", "drought_index", "consec_dry_days",
    "month", "monsoon_season", "lat", "lon",
    "conflict_events_30d", "conflict_fatalities_30d", "price_anomaly_30d",
    "groundwater_anomaly_cm", "water_point_count_50km", "sentinel2_ndvi",
]

FEATURE_NAMES_EN = {
    "drought_index": "drought index", "water_balance_30d": "30-day water balance",
    "consec_dry_days": "consecutive dry days", "precip_30d": "30-day rainfall",
    "precip_90d": "90-day rainfall", "et_30d": "30-day evapotranspiration",
    "temp_30d_avg": "average temperature", "lat": "latitude", "lon": "longitude",
    "month": "month of year", "monsoon_season": "monsoon timing",
    "conflict_events_30d": "nearby conflict events (ACLED)",
    "conflict_fatalities_30d": "nearby conflict fatalities (ACLED)",
    "price_anomaly_30d": "millet price anomaly (WFP)",
    "groundwater_anomaly_cm": "groundwater storage anomaly (GRACE-FO)",
    "water_point_count_50km": "mapped water points (OSM)",
    "sentinel2_ndvi": "vegetation health (Sentinel-2)",
}
FEATURE_NAMES_FR = {
    "drought_index": "indice de s\u00e9cheresse", "water_balance_30d": "bilan hydrique sur 30 jours",
    "consec_dry_days": "jours secs cons\u00e9cutifs", "precip_30d": "pr\u00e9cipitations sur 30 jours",
    "precip_90d": "pr\u00e9cipitations sur 90 jours", "et_30d": "\u00e9vapotranspiration sur 30 jours",
    "temp_30d_avg": "temp\u00e9rature moyenne", "lat": "latitude", "lon": "longitude",
    "month": "mois de l'ann\u00e9e", "monsoon_season": "p\u00e9riode de mousson",
    "conflict_events_30d": "\u00e9v\u00e9nements de conflit proches (ACLED)",
    "conflict_fatalities_30d": "victimes de conflit proches (ACLED)",
    "price_anomaly_30d": "anomalie du prix du mil (WFP)",
    "groundwater_anomaly_cm": "anomalie de stockage souterrain (GRACE-FO)",
    "water_point_count_50km": "points d'eau cartographi\u00e9s (OSM)",
    "sentinel2_ndvi": "sant\u00e9 de la v\u00e9g\u00e9tation (Sentinel-2)",
}


def real_shap_top_drivers(row, lang="en", top_n=4):
    """Real per-instance SHAP attribution for this exact district's current
    prediction, loaded from the saved TreeExplainer — the same mechanism
    used by the live AI Assistant, not a re-derived approximation."""
    try:
        explainer = get_shap_explainer()
        if explainer is None:
            return []
        model, le, _ = get_model_artifacts()
        X = row[FEATURE_COLS].to_frame().T.astype(float)
        predicted_class_idx = list(le.classes_).index(row["predicted_risk"])
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            class_shap = shap_values[predicted_class_idx][0]
        else:
            class_shap = shap_values[0, :, predicted_class_idx]
        contributions = sorted(zip(FEATURE_COLS, class_shap), key=lambda x: -abs(x[1]))[:top_n]
        names = FEATURE_NAMES_FR if lang == "fr" else FEATURE_NAMES_EN
        return [(names.get(f, f), float(v)) for f, v in contributions]
    except Exception:
        return []


def district_history(district, country, lang="en", limit=4):
    """Real risk-level transitions for this specific district, derived
    from the historical scored dataset — the same logic as the live Feed."""
    try:
        df = get_scored_df()
        d = df[df["district"] == district].sort_values("date").copy()
        d["prev_risk"] = d["predicted_risk"].shift(1)
        transitions = d[d["prev_risk"].notna() & (d["predicted_risk"] != d["prev_risk"])]
        transitions = transitions.sort_values("date", ascending=False).head(limit)
        out = []
        for _, r in transitions.iterrows():
            out.append((r["date"].strftime("%d %b %Y"), r["prev_risk"], r["predicted_risk"]))
        return out
    except Exception:
        return []


def country_ranking(country, this_district, lang="en", top_n=6):
    """Real comparison of every district in the country, ranked by current
    severity, so this district's relative position is shown in context."""
    latest = get_latest_snapshot()
    rows = latest[latest["country"] == country].copy()
    rows["severity"] = rows["predicted_risk"].map(SEVERITY_RANK)
    rows = rows.sort_values("severity", ascending=False).head(top_n)
    risk_map = RISK_LEVEL_FR if lang == "fr" else {}
    out = []
    for _, r in rows.iterrows():
        label = risk_map.get(r["predicted_risk"], r["predicted_risk"])
        out.append((r["district"], label, r["predicted_risk"] == this_district or r["district"] == this_district))
    return out


def load_model_metrics():
    try:
        with open(os.path.join(DATA_DIR, "metrics.json")) as f:
            return jsonlib.load(f)
    except Exception:
        return None


def load_causal_context(row, lang="en"):
    """Pulls the real DoWhy-estimated effect sizes and contextualizes them
    against this district's own current measured signals."""
    try:
        with open(os.path.join(DATA_DIR, "causal_results.json")) as f:
            causal = jsonlib.load(f)
    except Exception:
        return []
    lines = []
    drought_ate = causal.get("drought_effect", {}).get("average_treatment_effect")
    if drought_ate is not None and row["drought_index"] < -0.5:
        if lang == "fr":
            lines.append(f"Ce district est en s\u00e9cheresse s\u00e9v\u00e8re. Sur l'ensemble des districts, ce facteur est associ\u00e9 \u00e0 une hausse de {drought_ate*100:.1f} points de la probabilit\u00e9 de risque Critique (estimation DoWhy, ajust\u00e9e zone/saison).")
        else:
            lines.append(f"This district is in severe drought. Across all districts, this factor is associated with a {drought_ate*100:.1f}-point increase in Critical-risk probability (DoWhy estimate, zone/season-adjusted).")
    price_ate = causal.get("price_effect", {}).get("average_treatment_effect")
    if price_ate is not None and (row.get("price_anomaly_30d") or 0) > 1.0:
        if lang == "fr":
            lines.append(f"Une anomalie de prix significative est aussi mesur\u00e9e ici. Ce facteur est associ\u00e9 \u00e0 {price_ate*100:.2f} points de probabilit\u00e9 de risque Critique en plus, \u00e0 l'\u00e9chelle de tous les districts.")
        else:
            lines.append(f"A significant price anomaly is also measured here. This factor is associated with {price_ate*100:.2f} additional points of Critical-risk probability, across all districts.")
    return lines


def generate_pdf(row, district_name, lang="en"):
    L = TXT.get(lang, TXT["en"])
    recs = RECOMMENDATIONS_FR.get(row["predicted_risk"], []) if lang == "fr" else RECOMMENDATIONS.get(row["predicted_risk"], [])
    risk_display = RISK_LEVEL_FR.get(row["predicted_risk"], row["predicted_risk"]) if lang == "fr" else row["predicted_risk"]
    zone_display = ZONE_FR.get(row["zone"], row["zone"]) if lang == "fr" else row["zone"]
    risk_hex = RISK_COLORS.get(row["predicted_risk"], "#888888")
    v2_result = get_v2_prediction(row)  # computed once, reused for AI summary, the explicit section below, and the action plan

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=0.6*cm, bottomMargin=1.8*cm, leftMargin=1.8*cm, rightMargin=1.8*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("card_label", parent=styles["Normal"], fontSize=7, leading=9, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle("card_value", parent=styles["Normal"], fontSize=13, leading=16, fontName="Helvetica-Bold"))

    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=C_INK, spaceBefore=14, spaceAfter=6, fontSize=13)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14.5, textColor=C_INK)
    caption = ParagraphStyle("caption", parent=styles["Normal"], fontSize=7.5, textColor=colors.grey, spaceAfter=10)
    quote = ParagraphStyle("quote", parent=body, backColor=C_GOLD_LIGHT, borderPadding=8, leftIndent=4)

    elements = []

    # ── Branded header band ────────────────────────────────────────────────
    title_p = Paragraph(f'<font color="white" size="17"><b>SAHELI</b></font> '
                         f'<font color="white" size="10">\u2014 {L["doc_title"]}</font>', styles["Normal"])
    meta_p = Paragraph(f'<font color="#F2EADA" size="8">{L["generated"].format(date=datetime.now().strftime("%d %B %Y"))}</font>', styles["Normal"])
    risk_badge_p = Paragraph(
        f'<font color="white" size="11"><b>{risk_display.upper()}</b></font>',
        ParagraphStyle("badge", parent=styles["Normal"], alignment=TA_CENTER)
    )
    header_table = Table([[title_p, risk_badge_p], [meta_p, ""]], colWidths=[13*cm, 5.2*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_INK),
        ("BACKGROUND", (1,0), (1,1), colors.HexColor(risk_hex)),
        ("SPAN", (1,0), (1,1)),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,0), (1,0), "CENTER"),
        ("TOPPADDING", (0,0), (0,0), 16), ("BOTTOMPADDING", (0,1), (0,1), 16),
        ("LEFTPADDING", (0,0), (0,-1), 14), ("RIGHTPADDING", (1,0), (1,-1), 0),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 14))

    # ── District identity row ─────────────────────────────────────────────
    id_table = Table([[
        Paragraph(f'<font size="8" color="#6E6353">{L["district"]}</font><br/><font size="13"><b>{row["district"]}</b></font>', body),
        Paragraph(f'<font size="8" color="#6E6353">{L["country"]}</font><br/><font size="13"><b>{row["country"]}</b></font>', body),
        Paragraph(f'<font size="8" color="#6E6353">{L["zone"]}</font><br/><font size="13"><b>{zone_display}</b></font>', body),
    ]], colWidths=[6.06*cm, 6.06*cm, 6.06*cm])
    id_table.setStyle(TableStyle([("BOTTOMPADDING", (0,0), (-1,-1), 10)]))
    elements.append(id_table)
    elements.append(HRFlowable(width="100%", color=C_BORDER, thickness=1))
    elements.append(Spacer(1, 10))

    # ── Satellite + situation summary side by side ────────────────────────
    img_buf = fetch_satellite_image(row["lat"], row["lon"])
    sat_cell = Image(img_buf, width=5.4*cm, height=5.4*cm) if img_buf else Paragraph(L["satellite_unavailable"], caption)
    situation_cell = [
        Paragraph(L["situation_summary"], h2),
        Paragraph(L["situation_text"].format(
            district=row["district"], zone=zone_display, risk=risk_display,
            drought=row["drought_index"], dry_days=int(row["consec_dry_days"]),
        ), body),
    ]
    sit_table = Table([[sat_cell, situation_cell]], colWidths=[5.8*cm, 12.4*cm])
    sit_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (1,0), (1,0), 12)]))
    elements.append(sit_table)
    if img_buf:
        elements.append(Paragraph(L["satellite_caption"].format(lat=row["lat"], lon=row["lon"]), caption))

    ai_summary = _ai_executive_summary(row, lang, v2_result)
    if ai_summary:
        elements.append(Paragraph(L["ai_label"], h2))
        elements.append(Paragraph(ai_summary, quote))
    elements.append(Spacer(1, 4))

    # ── Real 90-day trend chart ───────────────────────────────────────────
    chart_buf = render_trend_chart(district_name, risk_hex)
    if chart_buf:
        elements.append(Paragraph(L["trend_title"], h2))
        elements.append(Image(chart_buf, width=16.2*cm, height=4.7*cm))
        elements.append(Spacer(1, 4))

    # ── Multi-source indicator card grid (2 rows x 3) ─────────────────────
    elements.append(Paragraph(L["multi_source"], h2))
    cards_row1 = [
        _card(L["drought_label"], f"{row['drought_index']:.2f} idx", risk_hex, styles),
        _card(L["conflict_label"], f"{int(row.get('conflict_events_30d', 0) or 0)}", "#A53A26", styles),
        _card(L["price_label"], f"{row.get('price_anomaly_30d', 0) or 0:.2f}\u03c3", "#B87721", styles),
    ]
    cards_row2 = [
        _card(L["groundwater_label"], f"{row.get('groundwater_anomaly_cm', 0) or 0:.1f} cm", "#5A6E4C", styles),
        _card(L["ndvi_label"], f"{row.get('sentinel2_ndvi', 0) or 0:.3f}", "#5A6E4C", styles),
        _card(L["water_label"], f"{int(row.get('water_point_count_50km', 0) or 0)}", "#A86E2A", styles),
    ]
    grid = Table([cards_row1, cards_row2], colWidths=[5.4*cm, 5.4*cm, 5.4*cm])
    grid.setStyle(TableStyle([("LEFTPADDING", (0,0), (-1,-1), 3), ("RIGHTPADDING", (0,0), (-1,-1), 3),
                               ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
    elements.append(grid)
    elements.append(Spacer(1, 6))

    # ── Full probability breakdown (real model output) ─────────────────────
    elements.append(Paragraph(L["prob_title"], h2))
    prob_rows = [
        ("Low", row.get("prob_low", 0) or 0, "#5A6E4C"),
        ("Medium", row.get("prob_medium", 0) or 0, "#B87721"),
        ("High", row.get("prob_high", 0) or 0, "#A8642B"),
        ("Critical", row.get("prob_critical", 0) or 0, "#A53A26"),
    ]
    prob_label_map = RISK_LEVEL_FR if lang == "fr" else {}
    prob_cells = []
    for level, p, c in prob_rows:
        label = prob_label_map.get(level, level)
        bar_w = max(0.15, p * 4.5)
        bar = Table([[""]], colWidths=[bar_w*cm], rowHeights=[0.32*cm])
        bar.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), colors.HexColor(c))]))
        prob_cells.append([Paragraph(f'<font size="8">{label}</font>', body), bar,
                            Paragraph(f'<font size="8"><b>{p*100:.1f}%</b></font>', body)])
    prob_table = Table(prob_cells, colWidths=[2.4*cm, 5.0*cm, 1.6*cm])
    prob_table.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3)]))
    elements.append(prob_table)
    elements.append(Spacer(1, 8))

    # ── Real per-instance SHAP attribution ──────────────────────────────────
    shap_drivers = real_shap_top_drivers(row, lang)
    if shap_drivers:
        elements.append(Paragraph(L["shap_title"], h2))
        for feat, val in shap_drivers:
            direction = L["shap_increase"] if val > 0 else L["shap_decrease"]
            elements.append(Paragraph(f"\u2022 " + L["shap_line"].format(feat=feat, dir=direction, val=abs(val)), body))
        elements.append(Spacer(1, 6))

    # ── Causal context (real DoWhy effects) ─────────────────────────────────
    causal_lines = load_causal_context(row, lang)
    elements.append(Paragraph(L["causal_title"], h2))
    if causal_lines:
        for line in causal_lines:
            elements.append(Paragraph(f"\u2022 {line}", body))
    else:
        elements.append(Paragraph(L["causal_none"], caption))
    elements.append(Spacer(1, 6))

    # ── Recent risk history for this district ───────────────────────────────
    elements.append(Paragraph(L["history_title"], h2))
    hist = district_history(district_name, row["country"], lang)
    if hist:
        for date_str, from_r, to_r in hist:
            from_label = RISK_LEVEL_FR.get(from_r, from_r) if lang == "fr" else from_r
            to_label = RISK_LEVEL_FR.get(to_r, to_r) if lang == "fr" else to_r
            elements.append(Paragraph(f"\u2022 " + L["history_line"].format(date=date_str, from_r=from_label, to_r=to_label), body))
    else:
        elements.append(Paragraph(L["history_none"], caption))
    elements.append(Spacer(1, 6))

    # ── Regional ranking context ─────────────────────────────────────────────
    elements.append(Paragraph(L["ranking_title"], h2))
    ranking = country_ranking(row["country"], district_name, lang)
    rank_rows = [[Paragraph(f'<font size="8"><b>#</b></font>', body),
                  Paragraph(f'<font size="8"><b>{"District" if lang=="en" else "District"}</b></font>', body),
                  Paragraph(f'<font size="8"><b>{"Risk" if lang=="en" else "Risque"}</b></font>', body)]]
    current_row_idx = None
    for i, (d_name, risk_label, is_current) in enumerate(ranking, 1):
        weight = "Helvetica-Bold" if d_name == district_name else "Helvetica"
        if d_name == district_name:
            current_row_idx = i
        marker = " &lt;--" if d_name == district_name else ""
        rank_rows.append([
            Paragraph(f'<font size="9"><b>{i}</b></font>', body),
            Paragraph(f'<font size="9" name="{weight}">{d_name}{marker}</font>', body),
            Paragraph(f'<font size="9">{risk_label}</font>', body),
        ])
    rank_table = Table(rank_rows, colWidths=[1*cm, 9.6*cm, 5.4*cm])
    rank_style = [
        ("BACKGROUND", (0,0), (-1,0), C_GOLD_LIGHT), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("LINEBELOW", (0,0), (-1,-1), 0.3, C_BORDER), ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]
    if current_row_idx:
        rank_style.append(("BACKGROUND", (0, current_row_idx), (-1, current_row_idx), C_GOLD_LIGHT))
    rank_table.setStyle(TableStyle(rank_style))
    elements.append(rank_table)
    elements.append(Spacer(1, 6))

    # ── Resource & Budget Estimate (real formula, disclosed) ──────────────
    estimate = compute_resource_estimate(row, row["country"])
    elements.append(Paragraph(L["resource_title"], h2))
    if estimate:
        allocation, pct = estimate
        elements.append(Paragraph(
            L["resource_text"].format(budget=DEFAULT_NATIONAL_BUDGET, country=row["country"],
                                       district=row["district"], allocation=allocation, pct=pct), body
        ))
        elements.append(Paragraph(f'<i>{L["resource_disclosure"]}</i>',
                                   ParagraphStyle("disc", parent=caption, fontSize=7.5)))
    elements.append(Spacer(1, 6))

    # ── Ground truth ───────────────────────────────────────────────────────
    elements.append(Paragraph(L["ground_truth"], h2))
    ipc = row.get("ipc_phase_observed")
    if ipc is not None and not (isinstance(ipc, float) and ipc != ipc):
        elements.append(Paragraph(L["ground_truth_text"].format(ipc=float(ipc)), body))
    else:
        elements.append(Paragraph(L["ground_truth_unavailable"], body))
    elements.append(Spacer(1, 6))

    # ── Real food-security model (v2), explicit, compared against the
    # climate-shock classification above ─────────────────────────────────
    elements.append(Paragraph(L["v2_title"], h2))
    if v2_result is None:
        elements.append(Paragraph(L["v2_unavailable"], caption))
    else:
        risk_label_v2 = RISK_LEVEL_FR.get(v2_result["risk"], v2_result["risk"]) if lang == "fr" else v2_result["risk"]
        body_key = "v2_validated" if v2_result["status"] == "validated" else "v2_extrapolated"
        elements.append(Paragraph(L[body_key].format(ipc=v2_result["ipc"], risk=risk_label_v2), body))
        if v2_result["risk"] == row["predicted_risk"]:
            elements.append(Paragraph(L["v2_agree"].format(risk=risk_display), quote))
        elif SEVERITY_RANK[v2_result["risk"]] > SEVERITY_RANK[row["predicted_risk"]]:
            elements.append(Paragraph(L["v2_escalate"].format(climate_risk=risk_display), quote))
        else:
            elements.append(Paragraph(L["v2_deescalate"].format(climate_risk=risk_display), quote))
    elements.append(Spacer(1, 6))

    # ── Crop Scanner field reports (qualitative, real, not fused into the score) ──
    crop_scans = get_recent_crop_scans(row["district"], limit=5)
    if crop_scans:
        n_disease = sum(1 for s in crop_scans if s["predicted_class"] != "Healthy")
        elements.append(Paragraph(L["crop_scan_title"], h2))
        elements.append(Paragraph(
            L["crop_scan_text"].format(n=len(crop_scans), n_disease=n_disease), body
        ))
        for s in crop_scans[:3]:
            elements.append(Paragraph(
                f"&bull; {s['created_at']} \u2014 {s['predicted_class']} ({s['confidence']*100:.0f}%)", caption
            ))
        elements.append(Spacer(1, 6))

    # ── Key drivers ────────────────────────────────────────────────────────
    elements.append(Paragraph(L["key_drivers"], h2))
    drivers = []
    if row["drought_index"] < -0.5:
        drivers.append(L["drought_driver"].format(v=abs(row["drought_index"])))
    if row["consec_dry_days"] >= 15:
        drivers.append(L["dryspell_driver"].format(d=int(row["consec_dry_days"])))
    if row["monsoon_season"] == 1:
        drivers.append(L["monsoon_driver"])
    if not drivers:
        drivers.append(L["no_anomaly"])
    for d in drivers:
        elements.append(Paragraph(f"\u2022 {d}", body))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph(L["recommended"], h2))
    for rec in recs:
        elements.append(Paragraph(f"\u2022 {rec}", body))
    elements.append(Spacer(1, 10))

    # ── Full multi-horizon action plan ─────────────────────────────────────
    elements.append(HRFlowable(width="100%", color=C_GOLD, thickness=1))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(L["action_plan_title"], h2))
    plan = build_action_plan(row, lang, v2_result)

    tier_colors = {"immediate": C_CLAY, "short_term": C_AMBER, "strategic": C_ACACIA}
    action_cell_style = ParagraphStyle("actioncell", parent=styles["Normal"], fontSize=9, leading=12.5)
    owner_cell_style = ParagraphStyle("ownercell", parent=styles["Normal"], fontSize=8, leading=11, textColor=colors.HexColor("#555555"))

    for tier_key, tier_label_key in [("immediate", "immediate_tier"), ("short_term", "short_term_tier"), ("strategic", "strategic_tier")]:
        tier_style = ParagraphStyle(f"tier_{tier_key}", parent=styles["Normal"], fontSize=9.5, textColor=colors.white,
                                     backColor=tier_colors[tier_key], spaceBefore=8, spaceAfter=4,
                                     leftIndent=6, leading=16)
        block = [Paragraph(f"&nbsp;{L[tier_label_key]}", tier_style)]
        rows_data = [[Paragraph(f"<b>{L['action_col']}</b>", action_cell_style), Paragraph(f"<b>{L['owner_col']}</b>", action_cell_style)]]
        for action_text, owner in plan[tier_key]:
            rows_data.append([Paragraph(f"\u2022 {action_text}", action_cell_style), Paragraph(owner, owner_cell_style)])
        tier_table = Table(rows_data, colWidths=[11*cm, 4.6*cm])
        tier_table.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("BACKGROUND", (0,0), (-1,0), C_GOLD_LIGHT),
            ("LINEBELOW", (0,0), (-1,-1), 0.4, C_BORDER),
            ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        block.append(tier_table)
        elements.append(KeepTogether(block))
        elements.append(Spacer(1, 6))

    metrics = load_model_metrics()
    if metrics:
        elements.append(Spacer(1, 8))
        elements.append(HRFlowable(width="100%", color=C_BORDER, thickness=0.5))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(L["model_credibility"], h2))
        elements.append(Paragraph(
            L["model_credibility_text"].format(
                acc=metrics.get("accuracy", 0) * 100, f1=metrics.get("weighted_f1", 0) * 100,
                n=metrics.get("n_test", 0)
            ), caption
        ))

    footer_text = L["footer"]
    doc.build(
        elements,
        onFirstPage=lambda c, d: _page_footer(c, d, footer_text),
        onLaterPages=lambda c, d: _page_footer(c, d, footer_text),
    )
    buf.seek(0)
    return buf


@router.get("/brief/{district_name}")
def get_policy_brief(district_name: str, lang: str = Query("en", pattern="^(en|fr)$"), user: dict = Depends(get_current_user)):
    log_activity(user["id"], "policy_brief_generated")
    assert_district_access(district_name, user["country"])
    latest = get_latest_snapshot()
    match = latest[latest["district"] == district_name]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"District '{district_name}' not found")
    row = match.iloc[0]
    pdf_buf = generate_pdf(row, district_name, lang)
    filename = f"SAHELI_Brief_{district_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return StreamingResponse(
        pdf_buf, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )