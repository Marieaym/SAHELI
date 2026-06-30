"""
SAHELI — Retrain XGBoost including real ACLED conflict features.

Honest note up front: risk_level (the prediction target) is a rule-derived
label built purely from climate variables (see metrics.json validation_note,
unchanged by this script). Adding conflict_events_30d / conflict_fatalities_30d
as INPUT features to predict that same label will show low SHAP importance
for conflict almost by construction — the label simply doesn't encode
conflict information. That is reported honestly below, not hidden. The
substantively meaningful test of conflict's relationship to risk is the
separate causal estimate in causal_module.py, not this classifier.

Methodology fix, disclosed: earlier versions of this script used a random
80/20 train_test_split. On daily time series, that lets adjacent days from
the same district sit on both sides of the split — they are highly
correlated, so the model can partly "see its own neighbors," inflating
the reported test accuracy. This version uses the same chronological
split already used by tft_lite_module.py and food_security_v2_module.py
(train through 2021, test on 2022-2024, genuinely never seen), for
consistency and because it is the methodologically honest choice for
time series. Both numbers are reported below so the change itself is
visible, not silently swapped in.

A plain logistic regression baseline is also reported alongside XGBoost,
since "our model beats a simple linear model" is a real, useful sanity
check that was missing before.
"""
import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import shap

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "app", "models_data", "scored_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "app", "models_data")
TRAIN_END = "2021-12-31"

FEATURES = ["precip_30d", "precip_90d", "et_30d", "temp_30d_avg", "water_balance_30d",
            "drought_index", "consec_dry_days", "month", "monsoon_season", "lat", "lon",
            "conflict_events_30d", "conflict_fatalities_30d", "price_anomaly_30d",
            "groundwater_anomaly_cm", "water_point_count_50km", "sentinel2_ndvi"]


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).dropna(subset=FEATURES + ["risk_level"])
    print(f"{len(df)} rows after feature cleaning")

    le = LabelEncoder()
    df["_y"] = le.fit_transform(df["risk_level"])
    print("Classes:", list(le.classes_))

    # ── OLD methodology, kept only to report honestly what changed ────────
    X_all, y_all = df[FEATURES], df["_y"]
    X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    old_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   objective="multi:softprob", random_state=42)
    old_model.fit(X_train_random, y_train_random)
    old_acc = accuracy_score(y_test_random, old_model.predict(X_test_random))

    # ── NEW methodology: chronological split, consistent with the rest of
    # SAHELI's models, and the methodologically honest choice for time series ──
    train_mask = df["date"] <= pd.Timestamp(TRAIN_END)
    X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, "_y"]
    X_test, y_test = df.loc[~train_mask, FEATURES], df.loc[~train_mask, "_y"]
    print(f"Chronological split — train: {len(X_train)} (2015-2021)  test: {len(X_test)} (2022-2024, never seen)")

    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               objective="multi:softprob", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Chronological test Accuracy: {acc:.4f}  |  Weighted F1: {f1:.4f}  (old random-split accuracy was {old_acc:.4f})")

    # ── Linear baseline: does XGBoost actually beat a plain linear model? ──
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    linear_model = LogisticRegression(max_iter=2000)
    linear_model.fit(X_train_s, y_train)
    linear_acc = accuracy_score(y_test, linear_model.predict(X_test_s))
    linear_f1 = f1_score(y_test, linear_model.predict(X_test_s), average="weighted")
    print(f"Linear (logistic regression) baseline, same chronological split: Accuracy={linear_acc:.4f}  F1={linear_f1:.4f}")

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)
    importance = dict(sorted(zip(FEATURES, mean_abs_shap.tolist()), key=lambda x: -x[1]))
    print("\nFeature importance (mean |SHAP|), including new conflict features:")
    for k, v in importance.items():
        print(f"  {k}: {v:.4f}")

    conflict_rank = list(importance.keys()).index("conflict_events_30d") + 1
    price_rank = list(importance.keys()).index("price_anomaly_30d") + 1
    gw_rank = list(importance.keys()).index("groundwater_anomaly_cm") + 1
    print(f"\nconflict_events_30d ranks #{conflict_rank}, price_anomaly_30d ranks #{price_rank}, "
          f"groundwater_anomaly_cm ranks #{gw_rank} of {len(FEATURES)} features.")

    # Score the full dataset (used by the live app's district views)
    df = df.drop(columns=["_y"])
    full_probs = model.predict_proba(df[FEATURES])
    for i, cls in enumerate(le.classes_):
        df[f"prob_{cls.lower()}"] = full_probs[:, i]
    df["predicted_risk"] = le.inverse_transform(model.predict(df[FEATURES]))
    df.to_csv(DATA_PATH, index=False)

    joblib.dump(model, os.path.join(MODEL_DIR, "saheli_xgb_model.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.joblib"))

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
        "validation_methodology": {
            "current": "Chronological split: train on 2015-2021, test on 2022-2024 (never seen during training).",
            "old_random_split_accuracy_for_comparison": round(float(old_acc), 4),
            "why_changed": (
                "A random 80/20 split lets adjacent days from the same district appear on both "
                "sides of the split; since consecutive days are highly correlated, this inflates "
                "reported accuracy. The chronological split above is the honest choice for time "
                "series and is now used consistently across every SAHELI model."
            ),
        },
        "linear_baseline_comparison": {
            "method": "Logistic regression (multinomial, standardized features), same chronological split.",
            "linear_accuracy": round(float(linear_acc), 4),
            "linear_weighted_f1": round(float(linear_f1), 4),
            "xgboost_accuracy": round(float(acc), 4),
            "xgboost_weighted_f1": round(float(f1), 4),
            "interpretation": (
                f"XGBoost {'beats' if acc > linear_acc else 'does not clearly beat'} a plain linear "
                f"model on the same real, chronological test set ({acc:.1%} vs {linear_acc:.1%} "
                f"accuracy) — reported honestly either way, not assumed."
            ),
        },
        "validation_note": (
            "Metrics reflect performance against a rule-derived risk label "
            "(constructed from drought index and dry-day streaks per district), "
            "not independent ground-truth IPC/FEWS NET crisis classifications "
            "(see the separate real ground-truth comparison on this page, which "
            "shows a weak negative correlation — an important, honestly disclosed "
            "limitation, now addressed by food_security_v2_module.py). Real ACLED "
            "conflict, WFP price, and GRACE-FO groundwater features are now included "
            "as model inputs; their modest SHAP importance is expected since the "
            "label itself is climate-only by construction."
        ),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved model, encoder, explainer, and metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
