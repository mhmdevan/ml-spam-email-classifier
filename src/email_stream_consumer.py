# src/email_stream_consumer.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any

import pika
from joblib import load

from .text_preprocessing import basic_clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"


def load_model_bundle() -> Dict[str, Any]:
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Vectorizer/model not found. Run training first "
            "(python -m src.train_spam_classifier)."
        )
    vectorizer = load(VECTORIZER_PATH)
    model = load(MODEL_PATH)
    return {"vectorizer": vectorizer, "model": model}


def predict_spam_text(bundle: Dict[str, Any], text: str) -> Dict[str, Any]:
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    cleaned = basic_clean_text(text)
    X_vec = vectorizer.transform([cleaned])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vec)[:, 1]
        spam_prob = float(probs[0])
        label_int = int(spam_prob >= 0.5)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_vec)
        spam_prob = None
        label_int = int(scores[0] >= 0.0)
    else:
        preds = model.predict(X_vec)
        label_int = int(preds[0])
        spam_prob = None

    return {
        "label_int": label_int,
        "label_str": "spam" if label_int == 1 else "ham",
        "spam_probability": spam_prob,
        "cleaned_text": cleaned,
    }


def main() -> None:
    """
    Simple RabbitMQ consumer:
      - listens on input queue (raw emails)
      - runs spam classifier
      - publishes result to output queue
    """

    bundle = load_model_bundle()

    rabbit_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    queue_in = os.getenv("SPAM_INPUT_QUEUE", "emails_raw")
    queue_out = os.getenv("SPAM_OUTPUT_QUEUE", "emails_scored")

    params = pika.URLParameters(rabbit_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.queue_declare(queue=queue_in, durable=True)
    channel.queue_declare(queue=queue_out, durable=True)

    print(f"[STREAM] Connected to RabbitMQ at {rabbit_url}")
    print(f"[STREAM] Consuming from queue: {queue_in}")
    print(f"[STREAM] Publishing results to queue: {queue_out}")

    def callback(ch, method, properties, body: bytes):
        try:
            msg = json.loads(body.decode("utf-8"))
            text = msg.get("text", "")
            result = predict_spam_text(bundle, text)

            out_msg = {
                "id": msg.get("id"),
                "subject": msg.get("subject"),
                "text": text,
                "spam_label": result["label_str"],
                "spam_probability": result["spam_probability"],
            }
            channel.basic_publish(
                exchange="",
                routing_key=queue_out,
                body=json.dumps(out_msg).encode("utf-8"),
                properties=pika.BasicProperties(delivery_mode=2),
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"[STREAM] Processed message id={msg.get('id')} → {out_msg['spam_label']}")
        except Exception as exc:
            print(f"[STREAM][ERROR] {exc}")
            # Decide policy: ack or not. For now, ack to avoid poison-message loops.
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=10)
    channel.basic_consume(queue=queue_in, on_message_callback=callback)

    print("[STREAM] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()


if __name__ == "__main__":
    main()
