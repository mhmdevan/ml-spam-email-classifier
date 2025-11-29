# src/api.py

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .predict_spam import predict_single

app = FastAPI(
    title="Spam Classifier API",
    version="1.0.0",
    description=(
        "Production-like spam classifier service built on top of "
        "TF-IDF + multiple models (LogReg, RF, LinearSVM, NaiveBayes)."
    ),
)


class SpamRequest(BaseModel):
    text: str


class SpamPrediction(BaseModel):
    label_int: int
    label_str: str
    spam_probability: Optional[float] = None
    clean_text: str


@app.get("/health")
def health():
    """
    Simple health check.

    We call predict_single() on a tiny dummy text to verify that:
      - vectorizer + model can be loaded
      - the pipeline runs end-to-end

    If anything goes wrong, we report 'model_not_loaded' + error detail.
    """
    try:
        _ = predict_single("healthcheck email")
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        # do NOT raise HTTP error here; we just report status
        return {
            "status": "model_not_loaded",
            "detail": str(exc),
        }


@app.post("/predict", response_model=SpamPrediction)
def predict(req: SpamRequest):
    """
    Predict whether the given email text is SPAM or HAM.

    This is a thin wrapper around src.predict_spam.predict_single(),
    so that the CLI, tests, and API all share the same logic.
    """
    try:
        result = predict_single(req.text)
    except FileNotFoundError as exc:
        # Model artifacts missing → 503 (service unavailable)
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train the classifier first.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        # Any unexpected error → 500
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    # result is expected to be something like:
    # {
    #   "label_str": "spam" / "ham",
    #   "label_int": 1 or 0 (optional),
    #   "spam_probability": float or None,
    #   "clean_text": "...",
    #   ...
    # }

    raw_label_str = (result.get("label_str") or "").strip()
    spam_prob = result.get("spam_probability", None)
    clean_text = result.get("clean_text", "")

    # Normalise label to upper SPAM/HAM for API
    label_lower = raw_label_str.lower()
    if label_lower == "spam":
        label_int = 1
        label_str_norm = "SPAM"
    elif label_lower == "ham":
        label_int = 0
        label_str_norm = "HAM"
    else:
        # Fallback: try numeric label if provided
        raw_int = int(result.get("label_int", 0))
        label_int = 1 if raw_int == 1 else 0
        label_str_norm = "SPAM" if label_int == 1 else "HAM"

    return SpamPrediction(
        label_int=label_int,
        label_str=label_str_norm,
        spam_probability=spam_prob,
        clean_text=clean_text,
    )
