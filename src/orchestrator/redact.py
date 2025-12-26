from __future__ import annotations

import re


_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_RE_PHONE = re.compile(r"\b(\+?\d[\d\s().-]{7,}\d)\b")
_RE_CC = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def redact_pii(text: str) -> str:
    text = _RE_EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _RE_PHONE.sub("[REDACTED_PHONE]", text)
    # Very naive CC redaction (can over-redact, acceptable for demo)
    text = _RE_CC.sub("[REDACTED_CARD]", text)
    return text


def normalize(text: str) -> str:
    # Remove excessive whitespace
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def hard_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"
