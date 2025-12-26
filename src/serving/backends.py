from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from transformers import AutoTokenizer
import onnxruntime as ort


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class OnnxTextClassifier:
    """
    ONNXRuntime backend.
    Expects model_dir containing:
      - model.onnx
      - tokenizer files (tokenizer.json / vocab.txt / etc)
      - optional meta.json with {"threshold": ...}
    """
    def __init__(self, model_dir: Path, max_length: int = 256):
        self.model_dir = Path(model_dir).resolve()
        self.onnx_path = self.model_dir / "model.onnx"
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"model.onnx not found in: {self.model_dir}")

        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        # ORT threading knobs (optional)
        so = ort.SessionOptions()
        intra = int(os.getenv("B3_ORT_INTRA_OP_NUM_THREADS", "0"))
        inter = int(os.getenv("B3_ORT_INTER_OP_NUM_THREADS", "0"))
        if intra > 0:
            so.intra_op_num_threads = intra
        if inter > 0:
            so.inter_op_num_threads = inter

        # Providers: CPUExecutionProvider is safest cross-platform.
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        self.input_names = {i.name for i in self.session.get_inputs()}

        meta_path = self.model_dir / "meta.json"
        self.meta: Dict[str, Any] = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
        self.default_threshold: float = float(self.meta.get("threshold", 0.5))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        feed: Dict[str, np.ndarray] = {}

        if "input_ids" in self.input_names:
            feed["input_ids"] = enc["input_ids"].astype(np.int64)

        if "attention_mask" in self.input_names and "attention_mask" in enc:
            feed["attention_mask"] = enc["attention_mask"].astype(np.int64)

        # DistilBERT typically doesn't need token_type_ids, but keep it for compatibility
        if "token_type_ids" in self.input_names and "token_type_ids" in enc:
            feed["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

        outputs = self.session.run(None, feed)
        logits = outputs[0]
        probs = _softmax_np(logits, axis=-1)[:, 1]
        return probs.astype(float)


class SklearnTextClassifier:
    """
    sklearn/joblib backend.
    Expects artifact_dir containing:
      - model.joblib (must implement predict_proba)
      - optional meta.json with {"threshold": ...}
    """
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir).resolve()
        import joblib  # local import to keep import cost minimal

        model_path = self.artifact_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"model.joblib not found in: {self.artifact_dir}")

        self.bundle = joblib.load(model_path)

        meta_path = self.artifact_dir / "meta.json"
        self.meta: Dict[str, Any] = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
        self.default_threshold: float = float(self.meta.get("threshold", 0.5))

        if not hasattr(self.bundle, "predict_proba"):
            raise TypeError("Loaded sklearn artifact does not implement predict_proba().")

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        probs = self.bundle.predict_proba(texts)[:, 1]
        return probs.astype(float)


class TorchTextClassifier:
    """
    Pure PyTorch Transformers backend (for benchmarking vs ONNX).
    This is NOT for production speed on CPU — it’s for comparing Torch vs ONNX.
    Expects model_dir containing HF weights/config/tokenizer.

    env:
      - B3_TORCH_DEVICE=cpu|mps|cuda (default cpu)
      - B3_TORCH_THREADS=<int> (default 4)
    """
    def __init__(self, model_dir: Path, max_length: int = 256):
        self.model_dir = Path(model_dir).resolve()
        self.max_length = int(max_length)

        # Lazy imports so sklearn/onnx users don't pay torch import cost
        import torch
        from transformers import AutoModelForSequenceClassification

        self._torch = torch

        threads = int(os.getenv("B3_TORCH_THREADS", "4"))
        if threads > 0:
            torch.set_num_threads(threads)

        self.device = os.getenv("B3_TORCH_DEVICE", "cpu").strip().lower()
        if self.device not in {"cpu", "mps", "cuda"}:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        self.model.to(self.device)

        meta_path = self.model_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text("utf-8"))
            self.default_threshold = float(meta.get("threshold", 0.5))
        else:
            self.default_threshold = 0.5

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        torch = self._torch

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.inference_mode():
            out = self.model(**enc)
            logits = out.logits.detach().float().cpu().numpy()

        probs = _softmax_np(logits, axis=-1)[:, 1]
        return probs.astype(float)
