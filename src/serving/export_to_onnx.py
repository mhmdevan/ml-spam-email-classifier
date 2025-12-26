from __future__ import annotations

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from peft import PeftConfig, PeftModel  # LoRA merge

try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError("onnxruntime is required for sanity check. Install: pip install onnxruntime") from e


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "models" / "b3" / "onnx_export"


def _is_peft_adapter_dir(p: Path) -> bool:
    return (p / "adapter_config.json").exists() or (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _run(cmd: List[str]) -> None:
    print("[B3] RUN:", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}")


def export_to_onnx(
    source_dir: Path,
    out_dir: Path,
    threshold: float = 0.5,
    opset: int = 13,
    task: str = "text-classification",
    device: str = "cpu",
) -> Path:
    """
    Exports a HF SequenceClassification model to ONNX using optimum-cli (stable, supports --opset).
    If source_dir is a PEFT adapter dir, it merges LoRA weights into base model first, then exports.
    """
    source_dir = source_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "source_dir": str(source_dir),
        "threshold": float(threshold),
        "opset": int(opset),
        "task": task,
        "device": device,
        "exporter": "optimum-cli export onnx",
    }

    tmpdir_obj: Optional[tempfile.TemporaryDirectory] = None
    export_source = source_dir

    # ---- Merge LoRA if needed ----
    if _is_peft_adapter_dir(source_dir):
        peft_cfg = PeftConfig.from_pretrained(str(source_dir))
        base_name = peft_cfg.base_model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(str(source_dir))
        base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)

        model = PeftModel.from_pretrained(base, str(source_dir))
        merged = model.merge_and_unload()

        tmpdir_obj = tempfile.TemporaryDirectory(prefix="b3_merged_")
        tmpdir = Path(tmpdir_obj.name)

        merged.save_pretrained(str(tmpdir))
        tokenizer.save_pretrained(str(tmpdir))

        export_source = tmpdir
        meta["peft"] = {"type": "lora", "base_model": base_name, "merged": True}
        meta["tokenizer_from"] = "adapter_dir"
    else:
        meta["peft"] = None
        meta["tokenizer_from"] = "source_dir"

    # ---- Export using optimum-cli (supports --opset, --task) ----
    # Docs show: optimum-cli export onnx -m MODEL [--task TASK] [--opset OPSET] output :contentReference[oaicite:2]{index=2}
    cmd = [
        "optimum-cli", "export", "onnx",
        "--model", str(export_source),
        "--task", task,
        "--opset", str(opset),
        "--device", device,
        str(out_dir),
    ]
    _run(cmd)

    # ---- Ensure tokenizer is present in out_dir (safe even if already exported) ----
    tok = AutoTokenizer.from_pretrained(str(export_source))
    tok.save_pretrained(str(out_dir))

    # ---- Write meta ----
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Sanity check: run ORT on 2 texts ----
    onnx_path = out_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Expected model.onnx in {out_dir}, but not found. Export likely failed.")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}

    texts = ["Free money!!! click here", "Hi team, meeting tomorrow at 10am"]
    enc = tok(texts, return_tensors="np", padding=True, truncation=True, max_length=256)

    feed = {}
    if "input_ids" in input_names:
        feed["input_ids"] = enc["input_ids"].astype(np.int64)
    if "attention_mask" in input_names and "attention_mask" in enc:
        feed["attention_mask"] = enc["attention_mask"].astype(np.int64)
    if "token_type_ids" in input_names and "token_type_ids" in enc:
        feed["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

    logits = sess.run(None, feed)[0]
    probs = _softmax(logits, axis=-1)[:, 1]

    (out_dir / "sanity_check.json").write_text(
        json.dumps({"texts": texts, "spam_prob": probs.tolist()}, indent=2),
        encoding="utf-8",
    )

    if tmpdir_obj is not None:
        tmpdir_obj.cleanup()

    return out_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--task", default="text-classification")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    out = export_to_onnx(
        source_dir=Path(args.source),
        out_dir=Path(args.out),
        threshold=args.threshold,
        opset=args.opset,
        task=args.task,
        device=args.device,
    )
    print(f"[B3] Exported ONNX to: {out}")


if __name__ == "__main__":
    main()
