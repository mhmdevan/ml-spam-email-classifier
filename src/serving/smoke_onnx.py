from __future__ import annotations

import json
from pathlib import Path
import argparse
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="Path to ONNX model dir (contains model.onnx + tokenizer files)")
    p.add_argument("--max-len", type=int, default=256)
    args = p.parse_args()

    model_dir = Path(args.model_dir).resolve()
    onnx_path = model_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"model.onnx not found in: {model_dir}")

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}

    texts = [
        "free money click now!!!",
        "hey can we reschedule the meeting?",
        "URGENT!!! Your account will be closed. Click this link to verify immediately. " * 30,
        "Hi",
    ]

    enc = tok(texts, return_tensors="np", padding=True, truncation=True, max_length=args.max_len)

    feed = {}
    if "input_ids" in input_names:
        feed["input_ids"] = enc["input_ids"].astype(np.int64)
    if "attention_mask" in input_names and "attention_mask" in enc:
        feed["attention_mask"] = enc["attention_mask"].astype(np.int64)
    if "token_type_ids" in input_names and "token_type_ids" in enc:
        feed["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

    logits = sess.run(None, feed)[0]
    probs = softmax(logits, axis=-1)[:, 1].tolist()

    out = {"model_dir": str(model_dir), "texts": texts, "spam_prob": probs}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
