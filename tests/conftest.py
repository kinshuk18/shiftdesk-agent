"""
tests/conftest.py — shared pytest fixtures.

The DB fixtures require a live Postgres instance (from docker compose).
Tests that don't need the DB are marked with no special markers and run anywhere.
"""

import os
os.environ["LLM_PROVIDER"] = "fake"

import pytest
import psycopg
from datetime import datetime, timezone, timedelta

import jwt

from app.auth import Principal
from app.config import settings


# ---------------------------------------------------------------------------
# DB fixture — skip if DATABASE_URL is unreachable.
# ---------------------------------------------------------------------------

def _db_available() -> bool:
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def db_conn():
    """
    Session-scoped raw psycopg connection to the test database.
    Skips the whole session if the DB is not reachable.
    """
    if not _db_available():
        pytest.skip("Postgres not reachable; skipping DB tests")
    conn = psycopg.connect(settings.database_url)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def clean_chat_checkpoints_and_reset_llm():
    """
    Clear chat checkpoints and reset the FakeLLM singleton before each test.
    Prevents cross-test state leakage in PostgresSaver and FakeLLM.
    """
    if _db_available():
        try:
            with psycopg.connect(settings.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM checkpoint_writes WHERE thread_id LIKE 'chat:%'")
                    cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id LIKE 'chat:%'")
                    cur.execute("DELETE FROM checkpoints WHERE thread_id LIKE 'chat:%'")
                conn.commit()
        except Exception:
            pass
    try:
        import app.graph.chat as _chat_mod
        from app.llm.fake import FakeLLM
        _chat_mod._llm_singleton = FakeLLM()
    except Exception:
        pass
    yield


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def make_token(
    company_id: str,
    employee_id: str,
    role: str = "employee",
    secret: str | None = None,
    exp_delta_seconds: int = 3600,
    extra_claims: dict | None = None,
) -> str:
    """Build a signed JWT for tests."""
    if secret is None:
        secret = settings.jwt_secret
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": employee_id,
        "company_id": company_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(seconds=exp_delta_seconds),
        **(extra_claims or {}),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture
def acme_employee_token() -> str:
    return make_token("acme", "E101", "employee")


@pytest.fixture
def zenith_employee_token() -> str:
    return make_token("zenith", "E101", "employee")
