# src/monitoring_report.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output" / "monitoring"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_logs() -> pd.DataFrame:
    """
    Load all prediction_*.jsonl files into a single DataFrame.
    """
    if not LOGS_DIR.exists():
        raise FileNotFoundError(f"Logs directory not found: {LOGS_DIR}")

    frames: List[pd.DataFrame] = []
    for path in LOGS_DIR.glob("predictions_*.jsonl"):
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        if records:
            df_part = pd.DataFrame(records)
            df_part["log_file"] = path.name
            frames.append(df_part)

    if not frames:
        raise RuntimeError("No prediction logs found in logs/ directory.")

    df = pd.concat(frames, ignore_index=True)
    # Derive date from timestamp_utc
    df["date"] = pd.to_datetime(df["timestamp_utc"]).dt.date
    return df


def compute_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-day aggregates:
      - total predictions
      - spam count
      - ham count
      - spam ratio
      - avg text length
      - std text length
    """
    grouped = (
        df.groupby("date")
        .agg(
            total_predictions=("label_int", "count"),
            spam_count=("label_int", lambda x: int((x == 1).sum())),
            ham_count=("label_int", lambda x: int((x == 0).sum())),
            spam_ratio=("label_int", "mean"),
            avg_length=("raw_text_length", "mean"),
            std_length=("raw_text_length", "std"),
        )
        .reset_index()
    )

    # Drift vs first day as naive baseline
    baseline = grouped.iloc[0]
    grouped["spam_ratio_delta_vs_first"] = grouped["spam_ratio"] - baseline["spam_ratio"]
    grouped["avg_length_delta_vs_first"] = grouped["avg_length"] - baseline["avg_length"]

    return grouped


def main() -> None:
    df_logs = load_all_logs()
    daily_stats = compute_daily_stats(df_logs)

    out_csv = OUTPUT_DIR / "daily_stats.csv"
    daily_stats.to_csv(out_csv, index=False)
    print(f"[MON] Saved daily stats to {out_csv}")

    print("\n[MON] Daily stats preview:")
    print(daily_stats.tail())


if __name__ == "__main__":
    main()
