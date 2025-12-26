from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class EmailIn(BaseModel):
    source: str = Field(default="n8n", description="where this email came from")
    message_id: str | None = None

    from_addr: str | None = None
    subject: str | None = None
    received_at: str | None = None  # ISO string (n8n gives one)

    text: str = Field(..., description="plain-ish body text (already extracted)")
    spam_prob: float | None = None
    is_spam: bool | None = None
    spam_model: str | None = None  # e.g. 'onnx_distilbert_ft_opset18'


Category = Literal["scam", "promo", "phishing", "adult", "malware", "social", "receipt", "job", "unknown"]


class TriageOut(BaseModel):
    category: Category
    risk_score: int = Field(ge=0, le=100)
    short_summary: str
    recommended_action: str
    notes: str | None = None


class TriageResponse(BaseModel):
    stored: bool
    record_id: str
    triage: TriageOut
