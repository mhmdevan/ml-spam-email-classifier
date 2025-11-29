# tests/conftest.py

from pathlib import Path

import pytest

from src import train_spam_classifier


@pytest.fixture(scope="session", autouse=True)
def ensure_trained_model():
    """
    Train the spam classifier once per test session.

    This will:
      - load data
      - clean + split
      - fit vectorizer + models
      - save:
          models/tfidf_vectorizer.joblib
          models/spam_classifier.joblib
          models/model_metadata.json
          output/metrics/classification_metrics.json
    """
    print("\n[TEST] Running training once for the whole test session...")
    train_spam_classifier.run_training()
    project_root = train_spam_classifier.PROJECT_ROOT

    # sanity check: models directory exists
    models_dir = project_root / "models"
    metrics_dir = project_root / "output" / "metrics"
    assert models_dir.exists(), "models/ directory should exist after training"
    assert metrics_dir.exists(), "output/metrics directory should exist after training"

    yield
