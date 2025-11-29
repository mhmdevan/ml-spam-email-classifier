# src/text_preprocessing.py

from __future__ import annotations

import re


def basic_clean_text(text: str) -> str:
    """
    Basic text normalization for email / SMS spam classification.
    This is intentionally simple and fast; it does NOT do heavy NLP.

    Steps:
      - lowercase
      - replace URLs with a token
      - replace email addresses with a token
      - replace numbers with a token
      - collapse multiple spaces
    """
    if not isinstance(text, str):
        text = str(text)

    # lowercasing
    text = text.lower()

    # replace URLs
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)

    # replace email addresses
    text = re.sub(r"\S+@\S+", " EMAIL ", text)

    # replace numbers
    text = re.sub(r"\d+", " NUM ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
