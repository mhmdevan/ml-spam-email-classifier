# tests/test_predict.py

from src.predict_spam import predict_single
from src.train_spam_classifier import run_training


def test_predict_single_after_training():
    """
    Ensure that after training we can load the model and
    classify a simple spam-like and ham-like text without errors.
    """
    # Make sure model/vectorizer exist
    run_training(sample_size=800)

    spam_like = "Congratulations! You have won 1,000,000 dollars. Click this URL to claim your prize."
    ham_like = "Hi John, just wanted to confirm our meeting tomorrow at 10am. Thanks."

    spam_result = predict_single(spam_like)
    ham_result = predict_single(ham_like)

    assert spam_result["label_str"] in {"spam", "ham"}
    assert 0.0 <= (spam_result["spam_probability"] or 0.5) <= 1.0

    assert ham_result["label_str"] in {"spam", "ham"}
    assert 0.0 <= (ham_result["spam_probability"] or 0.5) <= 1.0
