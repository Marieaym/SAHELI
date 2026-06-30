"""
SAHELI Backend — shared data/model loading.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "app", "models_data")

RISK_COLORS = {"Low": "#6B9080", "Medium": "#B89B4A", "High": "#D9822B", "Critical": "#B83227"}

POP_WEIGHTS = {
    "Niamey": 1.4, "Zinder": 1.1, "Maradi": 1.0, "Tahoua": 0.9, "Agadez": 0.5, "Diffa": 0.6,
    "Bamako": 1.5, "Mopti": 0.8, "Timbuktu": 0.4, "Gao": 0.5,
    "Ouagadougou": 1.4, "Dori": 0.5, "Djibo": 0.4,
    "NDjamena": 1.3, "Abeche": 0.5,
    "Nouakchott": 0.9, "Kiffa": 0.4,
    "Matam": 0.5,
}

RECOMMENDATIONS = {
    "Critical": [
        "Immediate release of emergency food reserves to the district",
        "Deploy mobile health and nutrition screening units within 7 days",
        "Activate cash-transfer program for the most vulnerable households",
        "Pre-position water trucking capacity for pastoral corridors",
    ],
    "High": [
        "Place district on elevated monitoring status with weekly re-assessment",
        "Pre-position seed and fodder reserves at regional depots",
        "Issue early advisory to local agricultural extension officers",
    ],
    "Medium": [
        "Maintain standard monitoring cadence",
        "Verify market price stability through next reporting cycle",
    ],
    "Low": [
        "No intervention required at this time",
        "Continue routine seasonal monitoring",
    ],
}

RECOMMENDATIONS_FR = {
    "Critical": [
        "Libération immédiate des réserves alimentaires d'urgence vers le district",
        "Déploiement d'unités mobiles de santé et de dépistage nutritionnel dans les 7 jours",
        "Activation du programme de transferts monétaires pour les ménages les plus vulnérables",
        "Pré-positionnement de capacités de transport d'eau pour les couloirs pastoraux",
    ],
    "High": [
        "Placement du district sous surveillance renforcée avec réévaluation hebdomadaire",
        "Pré-positionnement de réserves de semences et de fourrage dans les dépôts régionaux",
        "Émission d'un avis précoce aux agents de vulgarisation agricole locaux",
    ],
    "Medium": [
        "Maintien de la cadence de surveillance standard",
        "Vérification de la stabilité des prix du marché lors du prochain cycle de rapport",
    ],
    "Low": [
        "Aucune intervention requise pour le moment",
        "Poursuite de la surveillance saisonnière de routine",
    ],
}

ALERTS = {
    # Honest confidence note, carried through from the original fr/ha/dje set:
    # these are SAHELI's own best-effort phrasing, not reviewed by a native
    # speaker linguist. The existing disclaimer in routers/alerts.py already
    # says production deployment should validate every localized string with
    # native speakers before field use — that applies to every language here,
    # old and new alike. Wolof and Arabic are added now because confidence in
    # basic structure and vocabulary is reasonably high; Bambara and Fulfulde/
    # Pulaar, also widely spoken across SAHELI's six countries, are
    # deliberately NOT included yet — confidence there is too low to
    # generate reliably without a native speaker, and guessing would risk
    # shipping wrong text dressed up as real coverage, exactly what SAHELI
    # has tried not to do anywhere else.
    "Critical": {
        "fr": "ALERTE SAHELI: Risque alimentaire CRITIQUE detecte a {district}. Secheresse severe depuis {days} jours. Contactez votre agent communautaire.",
        "ha": "GARGADI SAHELI: An gano hadarin abinci mai TSANANI a {district}. Fari mai tsanani na kwana {days}. Tuntubi wakilin al'umma.",
        "dje": "GAARI SAHELI: Riski hima-koyne no i gar {district} ra. Jaw kankam na han {days}. Ma ce ni kunda boro.",
        "wo": "ARTU SAHELI: Njarin lekk gu METTI lool feeñ ci {district}. Coono bu metti li {days} fan. Jokkoo ak sa jawriñ dëkk bi.",
        "ar": "\u062a\u062d\u0630\u064a\u0631 \u0633\u0627\u0647\u064a\u0644\u064a: \u062e\u0637\u0631 \u0641\u064a \u0627\u0644\u0623\u0645\u0646 \u0627\u0644\u0639\u0630\u0627\u0626\u064a \u0628\u0644\u0644 \u062f\u0631\u062c\u0629 \u062d\u0631\u062c\u0629 \u0641\u064a {district}. \u062c\u0641\u0627\u0641 \u0634\u062f\u064a\u062f \u0645\u0646\u0630 {days} \u064a\u0648\u0645\u0627. \u0627\u062a\u0635\u0644 \u0628\u0627\u0644\u0645\u0633\u0624\u0648\u0644 \u0627\u0644\u0645\u062d\u0644\u064a.",
    },
    "High": {
        "fr": "AVIS SAHELI: Risque alimentaire ELEVE a {district}. Surveillez vos reserves d'eau et de semences.",
        "ha": "SANARWA SAHELI: Hadarin abinci MAI YAWA a {district}. Lura da ruwa da iri.",
        "dje": "BAARO SAHELI: Riski bambata no i gar {district} ra. Ma haggoy hari nda dumi.",
        "wo": "XIBAAR SAHELI: Njarin lekk gu YOKKU ci {district}. Wottu sa ndox ak sa pepp yi.",
        "ar": "\u062a\u0646\u0628\u064a\u0647 \u0633\u0627\u0647\u064a\u0644\u064a: \u062e\u0637\u0631 \u0645\u0631\u062a\u0641\u0639 \u0641\u064a \u0627\u0644\u0623\u0645\u0646 \u0627\u0644\u0639\u0630\u0627\u0626\u064a \u0641\u064a {district}. \u0631\u0627\u0642\u0628 \u0645\u062e\u0632\u0648\u0646 \u0627\u0644\u0645\u0627\u0621 \u0648\u0627\u0644\u0628\u0630\u0648\u0631.",
    },
    "Medium": {
        "fr": "INFO SAHELI: Niveau de risque MODERE a {district}. Situation stable.",
        "ha": "BAYANI SAHELI: Matsakaicin hadari a {district}. Yanayi yana da kwanciyar hankali.",
        "dje": "BAARO SAHELI: Riski daabu no i gar {district} ra. Hala go kosey.",
        "wo": "XIBAAR SAHELI: Njarin lekk gu DIGGANTE ci {district}. Mukk dafa stabil.",
        "ar": "\u0645\u0639\u0644\u0648\u0645\u0629 \u0633\u0627\u0647\u064a\u0644\u064a: \u0645\u0633\u062a\u0648\u0649 \u062e\u0637\u0631 \u0645\u062a\u0648\u0633\u0637 \u0641\u064a {district}. \u0627\u0644\u0648\u0636\u0639 \u0645\u0633\u062a\u0642\u0631.",
    },
    "Low": {
        "fr": "INFO SAHELI: Risque FAIBLE a {district}. Aucune action requise.",
        "ha": "BAYANI SAHELI: Karamin hadari a {district}. Babu wani mataki da ake bukata.",
        "dje": "BAARO SAHELI: Riski kayna no i gar {district} ra. Sohõ kala si tilas.",
        "wo": "XIBAAR SAHELI: Njarin lekk gu TUUTI ci {district}. Amul liggéey bu war def.",
        "ar": "\u0645\u0639\u0644\u0648\u0645\u0629 \u0633\u0627\u0647\u064a\u0644\u064a: \u062e\u0637\u0631 \u0645\u0646\u062e\u0641\u0636 \u0641\u064a {district}. \u0644\u0627 \u062a\u0648\u062c\u062f \u0625\u062c\u0631\u0627\u0621\u0627\u062a \u0645\u0637\u0644\u0648\u0628\u0629.",
    },
}


@lru_cache()
def get_scored_df():
    path = os.path.join(DATA_DIR, "scored_dataset.csv")
    return pd.read_csv(path, parse_dates=["date"])


FEATURE_COLS = [
    "precip_30d", "precip_90d", "et_30d", "temp_30d_avg", "water_balance_30d",
    "drought_index", "consec_dry_days", "month", "monsoon_season", "lat", "lon",
    "conflict_events_30d", "conflict_fatalities_30d", "price_anomaly_30d",
    "groundwater_anomaly_cm", "water_point_count_50km", "sentinel2_ndvi",
]

_MODEL_PATH = os.path.join(DATA_DIR, "saheli_xgb_model.joblib")
_ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.joblib")
_EXPLAINER_PATH = os.path.join(DATA_DIR, "shap_explainer.joblib")
_METRICS_PATH = os.path.join(DATA_DIR, "metrics.json")


@lru_cache()
def get_metrics():
    with open(_METRICS_PATH) as f:
        return json.load(f)


def _train_and_save_artifacts():
    """Generate missing .joblib files from scored_dataset.csv (first-run bootstrap)."""
    import numpy as np
    import xgboost as xgb
    import shap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    csv_path = os.path.join(DATA_DIR, "scored_dataset.csv")
    df = pd.read_csv(csv_path).dropna(subset=FEATURE_COLS + ["risk_level"])
    X = df[FEATURE_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df["risk_level"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
    else:
        mean_abs_shap = (
            np.abs(shap_values).mean(axis=(0, 2))
            if shap_values.ndim == 3
            else np.abs(shap_values).mean(axis=0)
        )
    importance = dict(sorted(zip(FEATURE_COLS, mean_abs_shap.tolist()), key=lambda x: -x[1]))
    full_probs = model.predict_proba(df[FEATURE_COLS])
    for i, cls in enumerate(le.classes_):
        df[f"prob_{cls.lower()}"] = full_probs[:, i]
    df["predicted_risk"] = le.inverse_transform(model.predict(df[FEATURE_COLS]))
    df.to_csv(csv_path, index=False)
    joblib.dump(model, _MODEL_PATH)
    joblib.dump(le, _ENCODER_PATH)
    joblib.dump(explainer, _EXPLAINER_PATH)
    existing_note = None
    if os.path.exists(_METRICS_PATH):
        with open(_METRICS_PATH) as f:
            existing_note = json.load(f).get("validation_note")
    metrics = {
        "accuracy": acc,
        "weighted_f1": f1,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importance": importance,
        "risk_levels": list(le.classes_),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(le.classes_),
        "validation_note": existing_note or (
            "Metrics reflect performance against a rule-derived risk label."
        ),
    }
    with open(_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    get_metrics.cache_clear()
    get_model_artifacts.cache_clear()
    print(f">>> SAHELI: trained and saved model artifacts to {DATA_DIR}")


def ensure_model_artifacts():
    """Call on startup — creates .joblib files if missing (fresh clone / deploy)."""
    if not all(os.path.exists(p) for p in (_MODEL_PATH, _ENCODER_PATH, _EXPLAINER_PATH)):
        print(">>> SAHELI: model artifacts missing — training from scored_dataset.csv...")
        _train_and_save_artifacts()


@lru_cache()
def get_model_artifacts():
    ensure_model_artifacts()
    model = joblib.load(_MODEL_PATH)
    le = joblib.load(_ENCODER_PATH)
    metrics = get_metrics()
    return model, le, metrics


_V2_MODEL_PATH = os.path.join(DATA_DIR, "food_security_v2_model.json")
_V2_FEATURES_PATH = os.path.join(DATA_DIR, "food_security_v2_features.joblib")

V2_VALIDATED_DISTRICTS = {"Agadez", "Bamako", "Diffa", "Gao", "Maradi", "Mopti",
                          "Niamey", "Nouakchott", "Tahoua", "Zinder"}


def v2_ipc_to_risk_level(ipc_value):
    if ipc_value >= 2.5:
        return "Critical"
    if ipc_value >= 2.0:
        return "High"
    if ipc_value >= 1.5:
        return "Medium"
    return "Low"


@lru_cache()
def get_v2_model_artifacts():
    """The real food-security-targeted model (v2), trained directly on
    real FEWS NET IPC ground truth. Returns (model, features) or
    (None, None) if it has not been trained yet (run
    models/food_security_v2_module.py first)."""
    if not (os.path.exists(_V2_MODEL_PATH) and os.path.exists(_V2_FEATURES_PATH)):
        return None, None
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(_V2_MODEL_PATH)
    features = joblib.load(_V2_FEATURES_PATH)
    return model, features


@lru_cache()
def get_shap_explainer():
    """The real, saved SHAP TreeExplainer for the main climate model.
    Centralized here, cached, because it was being reloaded from disk on
    every single call from THREE different places (assistant.py's live
    Q&A, brief.py's PDF generation, and brief.py's real_shap_top_drivers
    used by Command Center for every district) before this fix — a real,
    confirmed source of slow Command Center load times, not a guess."""
    path = os.path.join(DATA_DIR, "shap_explainer.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@lru_cache()
def get_tft_weights():
    """The real, saved temporal attention forecaster weights. Centralized
    here, cached, for the same reason as get_shap_explainer above — this
    exact .npz was being reloaded from disk separately by forecast.py's
    live endpoint AND assistant.py's forecast_question answer path."""
    path = os.path.join(DATA_DIR, "tft_lite_weights.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)


@lru_cache()
def get_latest_snapshot():
    """Cached: this was being re-derived (sort + groupby + tail over the
    full 65k+ row dataset) on every single call before this fix — and
    it's called from 21 different places across nearly every router
    (districts, brief, scenario, intervention, food_security_v2,
    command_center, pipeline, anomaly, assistant, alerts). This single
    fix is the highest-leverage one in this pass: it speeds up every
    page that touches district data, not just Command Center."""
    df = get_scored_df()
    return df.sort_values("date").groupby("district").tail(1).reset_index(drop=True)


def get_district_country(district_name: str) -> str | None:
    """Look up which country a district belongs to, for access-control checks."""
    latest = get_latest_snapshot()
    row = latest[latest["district"] == district_name]
    return row.iloc[0]["country"] if not row.empty else None


def assert_district_access(district_name: str, user_country: str):
    """Raise 403 if the requested district does not belong to the user's own country."""
    from fastapi import HTTPException
    actual_country = get_district_country(district_name)
    if actual_country is None:
        raise HTTPException(status_code=404, detail=f"District '{district_name}' not found")
    if actual_country != user_country:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: '{district_name}' belongs to {actual_country}, your account is scoped to {user_country}.",
        )
