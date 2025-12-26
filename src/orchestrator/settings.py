from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    openai_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"  # change if you want
    llm_temperature: float = 0.0

    # Safety / privacy
    redact_pii: bool = True
    max_chars: int = 6000  # hard truncate before LLM

    # DB
    database_url: str = "postgresql+psycopg://spam:spam@localhost:5432/spamdb"

    # Service metadata
    service_name: str = "nlp-orchestrator"
    environment: str = "dev"


settings = Settings()
