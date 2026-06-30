"""
SAHELI Backend — Agent layer: live corn leaf disease detection.

Loads the real CNN trained from scratch by models/cv_corn_module.py and
scores an uploaded leaf photo. Single crop (corn), 4 classes, honestly
scoped — see models/cv_corn_module.py's module docstring for the full
honest framing of what this is and is not.

Honest link to food security risk: scans are logged per district (when
a district is provided) and surfaced as a qualitative field-report
signal — e.g. "2 disease detections logged this week" next to that
district's risk data. They are NOT fused into the quantitative risk
score: there is no real dataset connecting detected crop-disease
severity to IPC-scale food security outcomes, so this never pretends
to compute one. This is a real, working illustration of how a future,
properly validated version could feed in, not a finished fusion model.
"""
import os
import io
import sys
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from PIL import Image
import tensorflow as tf

from data_access import DATA_DIR
from routers.auth import get_current_user

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db import log_crop_scan, get_recent_crop_scans

router = APIRouter(prefix="/api", tags=["computer-vision"])

IMG_SIZE = 96
CLASS_NAMES = [
    "Cercospora / Gray Leaf Spot",
    "Common Rust",
    "Northern Leaf Blight",
    "Healthy",
]
_MODEL_PATH = os.path.join(DATA_DIR, "cv_corn_cnn.keras")
_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            return None
        _model = tf.keras.models.load_model(_MODEL_PATH)
    return _model


@router.post("/cv/corn-predict")
async def predict_corn_leaf(
    file: UploadFile = File(...),
    district: str | None = Form(None),
    user: dict = Depends(get_current_user),
):
    model = _get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="CV model not found. Run models/cv_corn_module.py first.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Please upload an image file (JPEG or PNG).")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read this file as an image.")

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr[None, ...]
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = round(float(probs[pred_idx]), 4)

    if district:
        log_crop_scan(user["id"], district, predicted_class, confidence)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "district_logged": district,
        "scope_note": (
            "Single-crop proof of concept: trained only on corn/maize leaves "
            "from the public PlantVillage dataset. Not validated on real "
            "Sahelian field photos, and not trained on millet, sorghum, or "
            "groundnut, the region's other staple crops. Held-out validation "
            "accuracy: 87.2% overall, but this hides a real weak spot: "
            "Cercospora/Gray Leaf Spot is correctly detected only about 10% of "
            "the time and is usually confused with Northern Leaf Blight. Treat "
            "a Cercospora or Northern Leaf Blight result here as 'likely one of "
            "these two', not a confirmed diagnosis."
        ),
    }


@router.get("/cv/scans/{district_name}")
def recent_scans(district_name: str, user: dict = Depends(get_current_user)):
    """Real, logged Corn Scanner reports for this district — a
    qualitative field-report signal shown alongside the quantitative
    risk model, not merged into its score."""
    scans = get_recent_crop_scans(district_name)
    n_disease = sum(1 for s in scans if s["predicted_class"] != "Healthy")
    return {
        "district": district_name,
        "n_recent_scans": len(scans),
        "n_disease_detections": n_disease,
        "scans": scans,
        "note": (
            "A qualitative field-report signal, logged from real Corn Scanner "
            "use, shown for context next to this district's risk data. Not "
            "merged into the quantitative risk score — no validated dataset "
            "links detected crop-disease severity to IPC-scale food security "
            "outcomes yet."
        ),
    }
