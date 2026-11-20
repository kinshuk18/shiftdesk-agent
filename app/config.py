"""
app/config.py — all configuration read from environment.
Uses pydantic-settings; values can come from .env or real env vars.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://shiftdesk:shiftdesk@db:5432/shiftdesk"
    jwt_secret: str = "change-me-in-production"

    # Frozen clock; empty string means "use real time".
    app_now: str = "2026-11-20T18:45:00+05:30"

    llm_provider: str = "fake"
    enable_dev_issuer: bool = True
    approval_ttl_hours: int = 24

    # LLM model names (overridden per provider; verify free-tier names before use).
    # TODO: verify groq free-tier model at https://console.groq.com/docs/models
    groq_model: str = ""
    # TODO: verify gemini free-tier model at https://ai.google.dev/gemini-api/docs/models/gemini
    gemini_model: str = ""

    # Fallback transport desk phone shown in failure messages when the company
    # phone is unavailable (e.g. DB unreachable).
    desk_phone: str = "the transport desk"


# Module-level singleton — import this everywhere.
settings = Settings()
