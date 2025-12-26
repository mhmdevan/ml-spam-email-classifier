from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .text_normalization import normalize_text_regex


@dataclass
class DatasetSplit:
    x_train_raw: np.ndarray
    x_test_raw: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_spam_csv(csv_path: Path) -> pd.DataFrame:
    """
    Compatible with your v1 loader logic (schemas):
      - v1/v2  -> label/text
      - text/target -> label
      - text/label
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    cols = set(df.columns)

    if {"v1", "v2"}.issubset(cols):
        df = df.rename(columns={"v1": "label", "v2": "text"})
    elif {"text", "target"}.issubset(cols):
        df = df.rename(columns={"target": "label"})
    elif {"text", "label"}.issubset(cols):
        pass
    else:
        raise ValueError(f"Unsupported schema. Columns found: {list(df.columns)}")

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str)

    # Normalize labels to 0/1
    if df["label"].dtype == object:
        df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(lambda x: 1 if x == "spam" else 0)
        )
    else:
        df["label"] = df["label"].astype(int)

    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    # Keep a “raw_norm” column for Transformers
    df["text_raw_norm"] = df["text"].map(normalize_text_regex)

    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    x = df["text_raw_norm"].values
    y = df["label"].values

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DatasetSplit(x_train, x_test, y_train, y_test)
