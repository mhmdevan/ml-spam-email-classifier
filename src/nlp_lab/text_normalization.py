from __future__ import annotations

import re

URL_TOKEN = "__url__"
EMAIL_TOKEN = "__email__"
NUM_TOKEN = "__num__"


_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NUM_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


def normalize_text_regex(text: str) -> str:
    """
    Minimal, safe normalization for downstream NLP/Transformers:
      - lower
      - strip html tags
      - replace url/email/num with stable tokens
      - collapse whitespace
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    text = _HTML_RE.sub(" ", text)
    text = text.lower()

    text = _URL_RE.sub(f" {URL_TOKEN} ", text)
    text = _EMAIL_RE.sub(f" {EMAIL_TOKEN} ", text)
    text = _NUM_RE.sub(f" {NUM_TOKEN} ", text)

    text = _WS_RE.sub(" ", text).strip()
    return text
