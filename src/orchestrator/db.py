from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import create_engine, String, Text, Float, Integer, Boolean, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .settings import settings


class Base(DeclarativeBase):
    pass


class EmailEvent(Base):
    __tablename__ = "email_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    source: Mapped[str] = mapped_column(String(50), default="n8n")
    message_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    from_addr: Mapped[str | None] = mapped_column(String(255), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(500), nullable=True)
    received_at: Mapped[str | None] = mapped_column(String(50), nullable=True)

    text: Mapped[str] = mapped_column(Text)

    spam_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_spam: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    spam_model: Mapped[str | None] = mapped_column(String(200), nullable=True)

    llm_category: Mapped[str] = mapped_column(String(50), default="unknown")
    llm_risk_score: Mapped[int] = mapped_column(Integer, default=0)
    llm_summary: Mapped[str] = mapped_column(Text, default="")
    llm_action: Mapped[str] = mapped_column(Text, default="")
    llm_notes: Mapped[str | None] = mapped_column(Text, nullable=True)


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


def insert_event(payload: Dict[str, Any]) -> str:
    with SessionLocal() as db:
        ev = EmailEvent(**payload)
        db.add(ev)
        db.commit()
        db.refresh(ev)
        return ev.id
