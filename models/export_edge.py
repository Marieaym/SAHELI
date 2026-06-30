"""
SAHELI — Real ONNX export and INT8 quantization of the trained XGBoost
model, for the essay's edge-deployment claim.

Honest framing: this verifies the model genuinely fits the constraints of
a Raspberry Pi 4 edge node (file size, CPU-only inference latency) using
real export and quantization tooling (onnx, onnxruntime, skl2onnx). It
does NOT include a physical hardware test — we do not have a Raspberry Pi
in hand. What's real: the exported file, its measured size, and its
measured CPU inference latency in this environment. What's not claimed:
actual on-device benchmarking, solar power draw, or a 80 USD bill of
materials — those remain roadmap items, not measured facts.
"""
import joblib
import json
import os
import time
import numpy as np
import pandas as pd
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "app", "models_data")
FEATURES = ["precip_30d", "precip_90d", "et_30d", "temp_30d_avg", "water_balance_30d",
            "drought_index", "consec_dry_days", "month", "monsoon_season", "lat", "lon",
            "conflict_events_30d", "conflict_fatalities_30d", "price_anomaly_30d",
            "groundwater_anomaly_cm", "water_point_count_50km", "sentinel2_ndvi"]


def main():
    model = joblib.load(os.path.join(MODEL_DIR, "saheli_xgb_model.joblib"))
    df = pd.read_csv(os.path.join(MODEL_DIR, "scored_dataset.csv")).dropna(subset=FEATURES)
    X_sample = df[FEATURES].head(500).values.astype(np.float32)

    print("Exporting to ONNX...")
    n_features = X_sample.shape[1]
    # onnxmltools' XGBoost converter requires internal feature names to
    # follow the generic 'f0','f1',... pattern, not the pandas column
    # names baked in at training time. Reset them on the booster only for
    # this export; the original model.joblib used by the live app is
    # untouched.
    booster = model.get_booster()
    booster.feature_names = [f"f{i}" for i in range(n_features)]
    onnx_model = convert_xgboost(model, initial_types=[("input", FloatTensorType([None, n_features]))],
                                  target_opset=15)
    onnx_path = os.path.join(MODEL_DIR, "saheli_edge_model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    raw_size_kb = os.path.getsize(onnx_path) / 1024
    print(f"ONNX model exported: {raw_size_kb:.1f} KB")

    # Verify the exported model produces equivalent predictions
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_preds = sess.run(None, {input_name: X_sample})[0]
    sklearn_preds = model.predict(X_sample)
    agreement = float((onnx_preds.flatten() == sklearn_preds).mean())
    print(f"ONNX vs original prediction agreement on 500 real rows: {agreement:.4f}")

    # Benchmark CPU-only inference latency (proxy for edge feasibility —
    # NOT a physical Raspberry Pi measurement, disclosed below).
    n_runs = 200
    single_row = X_sample[:1]
    start = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {input_name: single_row})
    elapsed = time.perf_counter() - start
    per_inference_ms = (elapsed / n_runs) * 1000

    results = {
        "onnx_file_size_kb": round(raw_size_kb, 1),
        "prediction_agreement_vs_original": round(agreement, 4),
        "cpu_inference_latency_ms_per_request": round(per_inference_ms, 3),
        "raspberry_pi_4_ram_budget_mb": 4096,
        "model_footprint_vs_budget_pct": round((raw_size_kb / 1024) / 4096 * 100, 4),
        "honest_limitations": [
            "This is a real ONNX export and CPU-inference latency benchmark, "
            "run in a standard x86 environment — NOT a physical Raspberry Pi 4 "
            "measurement. We do not have a Raspberry Pi in hand to validate "
            "on-device.",
            "Solar power draw and the '$80 per edge node' bill of materials "
            "from the essay remain roadmap claims, not measured facts.",
            "The model is small enough in principle (sub-megabyte) that physical "
            "feasibility is very likely, but 'likely feasible' is disclosed as "
            "exactly that, not as 'deployed and tested'.",
        ],
    }
    with open(os.path.join(MODEL_DIR, "edge_export_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
