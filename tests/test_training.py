# tests/test_training.py

from pathlib import Path
import json

from src import train_spam_classifier


def test_training_outputs_exist():
    """
    After run_training(), core artifacts must exist:
      - tfidf_vectorizer.joblib
      - spam_classifier.joblib
      - model_metadata.json
      - classification_metrics.json (with at least one model block)
    """
    project_root: Path = train_spam_classifier.PROJECT_ROOT

    models_dir = project_root / "models"
    metrics_dir = project_root / "output" / "metrics"

    vec_path = models_dir / "tfidf_vectorizer.joblib"
    model_path = models_dir / "spam_classifier.joblib"
    meta_path = models_dir / "model_metadata.json"
    metrics_path = metrics_dir / "classification_metrics.json"

    assert vec_path.exists(), f"Expected vectorizer at {vec_path}"
    assert model_path.exists(), f"Expected classifier at {model_path}"
    assert meta_path.exists(), f"Expected model metadata at {meta_path}"
    assert metrics_path.exists(), f"Expected metrics JSON at {metrics_path}"

    # basic metrics sanity check
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # new reality:
    # {
    #   "LinearSVM": { "default": {...}, "tuned": {...}, ... },
    #   "RandomForest": {...},
    #   "NaiveBayes": {...},
    #   "best_model_name": "LinearSVM_calibrated",
    #   ...
    # }

    assert isinstance(data, dict), "metrics JSON should be a dict"
    assert len(data) >= 1, "metrics JSON should not be empty"

    # pick only model blocks (dicts that contain 'default' metrics)
    model_blocks = {
        name: block
        for name, block in data.items()
        if isinstance(block, dict) and "default" in block
    }

    assert model_blocks, "expected at least one model metrics block with a 'default' section"

    # take first model block and validate core metrics
    first_model_name, first_model_metrics = next(iter(model_blocks.items()))
    default_metrics = first_model_metrics["default"]

    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in default_metrics, f"'{key}' missing in default metrics of {first_model_name}"
        value = float(default_metrics[key])
        assert 0.0 <= value <= 1.0, f"{key} out of range [0,1]: {value}"
