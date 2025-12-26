from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .schemas import PredictRequest, PredictResponse, ItemPrediction, HealthResponse
from .backends import OnnxTextClassifier, SklearnTextClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="Spam Classifier API (B3)", version="0.1.0")


def _load_backend():
    backend = os.getenv("B3_BACKEND", "onnx").strip().lower()

    if backend == "onnx":
        model_dir = Path(os.getenv("B3_MODEL_DIR", str(PROJECT_ROOT / "models" / "b3" / "onnx_distilbert_ft")))
        model = OnnxTextClassifier(model_dir)
        return backend, model_dir, model

    if backend == "sklearn":
        model_dir = Path(os.getenv("B3_MODEL_DIR", str(PROJECT_ROOT / "models" / "b3" / "sklearn_tfidf_svm")))
        model = SklearnTextClassifier(model_dir)
        return backend, model_dir, model

    raise ValueError(f"Unknown B3_BACKEND: {backend}")


BACKEND_NAME, MODEL_DIR, MODEL = _load_backend()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", backend=BACKEND_NAME, model_dir=str(MODEL_DIR))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    texts = [t.strip() for t in req.texts if t and t.strip()]
    if not texts:
        return JSONResponse(status_code=400, content={"error": "texts are empty"})

    threshold = float(req.threshold) if req.threshold is not None else float(MODEL.default_threshold)

    probs = MODEL.predict_proba(texts)

    preds: List[ItemPrediction] = []
    for p in probs.tolist():
        is_spam = p >= threshold
        preds.append(
            ItemPrediction(
                label="spam" if is_spam else "ham",
                spam_prob=float(p),
                is_spam=bool(is_spam),
            )
        )

    return PredictResponse(
        backend=BACKEND_NAME,
        model_dir=str(MODEL_DIR),
        threshold=threshold,
        predictions=preds,
    )
