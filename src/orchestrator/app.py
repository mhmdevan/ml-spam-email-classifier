from __future__ import annotations

import uuid
from fastapi import FastAPI, HTTPException

from .settings import settings
from .schemas import EmailIn, TriageResponse, TriageOut
from .redact import normalize, redact_pii, hard_truncate
from .db import init_db, insert_event
from .llm_graph import build_graph


app = FastAPI(title=settings.service_name, version="0.1.0")
graph = build_graph()


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok", "service": settings.service_name, "env": settings.environment}


@app.post("/v1/triage", response_model=TriageResponse)
def triage_email(payload: EmailIn):
    # Normalize + truncate + redact
    text = normalize(payload.text)
    text = hard_truncate(text, settings.max_chars)
    if settings.redact_pii:
        text = redact_pii(text)

    # Run LangGraph flow
    try:
        state = {
            "text": text,
            "subject": payload.subject,
            "from_addr": payload.from_addr,
            "spam_prob": payload.spam_prob,
            "is_spam": payload.is_spam,
            "spam_model": payload.spam_model,
        }
        out = graph.invoke(state)
        triage_dict = out.get("triage")
        if not triage_dict:
            raise RuntimeError("LLM triage returned empty output")

        triage_obj = TriageOut(**triage_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"triage_failed: {e}")

    # Persist
    record_id = str(uuid.uuid4())
    db_payload = {
        "id": record_id,
        "source": payload.source,
        "message_id": payload.message_id,
        "from_addr": payload.from_addr,
        "subject": payload.subject,
        "received_at": payload.received_at,
        "text": text,
        "spam_prob": payload.spam_prob,
        "is_spam": payload.is_spam,
        "spam_model": payload.spam_model,
        "llm_category": triage_obj.category,
        "llm_risk_score": triage_obj.risk_score,
        "llm_summary": triage_obj.short_summary,
        "llm_action": triage_obj.recommended_action,
        "llm_notes": triage_obj.notes,
    }
    insert_event(db_payload)

    return TriageResponse(stored=True, record_id=record_id, triage=triage_obj)
