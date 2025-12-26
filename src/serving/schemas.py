from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=200)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ItemPrediction(BaseModel):
    label: str
    spam_prob: float
    is_spam: bool


class PredictResponse(BaseModel):
    backend: str
    model_dir: str
    threshold: float
    predictions: List[ItemPrediction]


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_dir: str
