"""
SAHELI Backend — Model Validation
Exposes the real metrics computed during training: test accuracy, weighted F1,
per-class precision/recall/F1, and the real confusion matrix. No fabricated
real-world retrospective claims — this is internal model validation only,
with the label-construction caveat surfaced explicitly.
"""
from fastapi import APIRouter
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_access import DATA_DIR

router = APIRouter(prefix="/api", tags=["validation"])


@router.get("/model/validation")
def get_validation():
    with open(os.path.join(DATA_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return {
        "accuracy": metrics["accuracy"],
        "weighted_f1": metrics["weighted_f1"],
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "classification_report": metrics.get("classification_report", {}),
        "confusion_matrix": metrics.get("confusion_matrix", []),
        "confusion_matrix_labels": metrics.get("confusion_matrix_labels", []),
        "validation_note": metrics.get("validation_note", ""),
    }


@router.get("/model/federated")
def get_federated_results():
    with open(os.path.join(DATA_DIR, "federated_results.json")) as f:
        return json.load(f)


@router.get("/model/rl")
def get_rl_results():
    with open(os.path.join(DATA_DIR, "rl_results.json")) as f:
        return json.load(f)


@router.get("/model/edge")
def get_edge_results():
    with open(os.path.join(DATA_DIR, "edge_export_results.json")) as f:
        return json.load(f)


@router.get("/model/ground-truth")
def get_ground_truth_validation():
    with open(os.path.join(DATA_DIR, "ground_truth_validation.json")) as f:
        return json.load(f)


@router.get("/model/monsoon-signal")
def get_monsoon_signal():
    with open(os.path.join(DATA_DIR, "monsoon_signal_results.json")) as f:
        return json.load(f)
