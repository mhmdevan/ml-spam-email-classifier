# src/predict_spam.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from joblib import load

from .text_preprocessing import basic_clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def load_model_and_vectorizer():
    """
    Load the trained TF-IDF vectorizer and spam classifier model from disk.
    """
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    vectorizer = load(VECTORIZER_PATH)
    model = load(MODEL_PATH)

    print(f"[LOAD] Loaded vectorizer from {VECTORIZER_PATH}")
    print(f"[LOAD] Loaded classifier from {MODEL_PATH}")
    return vectorizer, model


def predict_single(raw_text: str) -> Dict[str, Any]:
    """
    Predict whether a single email/text is spam or ham.

    Returns a dict:
      {
        "label_int": 0 or 1,
        "label_str": "ham" or "spam",
        "spam_probability": float or None,
        "cleaned_text": str,
      }
    """
    vectorizer, model = load_model_and_vectorizer()

    cleaned = basic_clean_text(raw_text)
    X_vec = vectorizer.transform([cleaned])

    label_int = int(model.predict(X_vec)[0])
    label_str = "spam" if label_int == 1 else "ham"

    spam_probability = None
    if hasattr(model, "predict_proba"):
        spam_probability = float(model.predict_proba(X_vec)[0, 1])

    return {
        "label_int": label_int,
        "label_str": label_str,
        "spam_probability": spam_probability,
        "cleaned_text": cleaned,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an email/text is spam or ham."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="Raw email/text content to classify. "
             "If omitted, the script will read from stdin.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON instead of human-readable text.",
    )

    args = parser.parse_args()

    if args.text:
        raw_text = args.text
    else:
        print("[INPUT] Please paste your email text. End with Ctrl+D (Unix) or Ctrl+Z (Windows):")
        raw_text = ""
        try:
            for line in iter(input, ""):
                raw_text += line + "\n"
        except EOFError:
            pass

    if not raw_text.strip():
        print("[ERROR] Empty input text. Nothing to classify.")
        return

    result = predict_single(raw_text)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n[RESULT]")
        print(f"Predicted label: {result['label_str'].upper()} (int={result['label_int']})")
        if result["spam_probability"] is not None:
            print(f"Estimated spam probability: {result['spam_probability']:.3f}")
        print("\n[DEBUG] Cleaned text preview:")
        print(result["cleaned_text"][:400])


if __name__ == "__main__":
    main()
