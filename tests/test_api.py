# tests/test_api.py

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    # In normal pipeline (with training fixture), we expect 'ok'
    # If in some environment model is missing, status will be 'model_not_loaded'.
    assert data["status"] in {"ok", "model_not_loaded"}

    if data["status"] == "ok":
        # then no detail is strictly required
        pass
    else:
        # if model_not_loaded, we at least expect some detail
        assert "detail" in data


def test_predict_spam_via_api():
    payload = {
        "text": "Congratulations! You have been selected for a FREE prize. Click here now!"
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["label_str"] == "SPAM"
    assert data["label_int"] == 1

    if data["spam_probability"] is not None:
        assert 0.7 <= data["spam_probability"] <= 1.0


def test_predict_ham_via_api():
    payload = {
        "text": "Hello, just sending you the notes from today's meeting. Let me know if you have questions."
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["label_str"] == "HAM"
    assert data["label_int"] == 0

    if data["spam_probability"] is not None:
        assert 0.0 <= data["spam_probability"] <= 0.5
