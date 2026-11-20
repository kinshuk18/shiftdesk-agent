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


@pytest.fixture
def stage3_data(db_conn):
    """Unique tenant data for Stage 3 tests; audit rows intentionally remain append-only."""
    import uuid
    from datetime import date, time

    suffix = uuid.uuid4().hex[:12]
    employees = [f"T{suffix}{i}" for i in range(12)]
    manager = f"M{suffix}"
    zenith_manager = f"Z{suffix}"
    zenith_employee = f"ZE{suffix}"
    cab = f"CAB{suffix}"
    zenith_cab = f"ZC{suffix}"
    entries = [f"ENT{suffix}{i}" for i in range(12)]
    approval_entry = f"APR{suffix}"
    fixture_conn = psycopg.connect(settings.database_url)
    with fixture_conn.transaction():
        with fixture_conn.cursor() as cur:
            cur.executemany("INSERT INTO employees(company_id,employee_id,name) VALUES('acme',%s,%s)", [(e, e) for e in employees] + [(manager, manager)])
            cur.executemany("INSERT INTO employees(company_id,employee_id,name) VALUES('zenith',%s,%s)", [(zenith_manager, zenith_manager), (zenith_employee, zenith_employee)])
            cur.execute("INSERT INTO cabs(company_id,cab_id,route_id,direction,departure_time,capacity) VALUES('acme',%s,'R1','drop','21:00',1)", (cab,))
            cur.execute("INSERT INTO cabs(company_id,cab_id,route_id,direction,departure_time,capacity) VALUES('zenith',%s,'R1','drop','21:00',1)", (zenith_cab,))
            cur.executemany("INSERT INTO roster_entries(company_id,entry_id,employee_id,service_date,direction,cab_id,scheduled_time,status) VALUES('acme',%s,%s,%s,'drop','C_AC_D01',%s,'active')", [(entries[i], employees[i], date(2026, 11, 20), time(22, 0)) for i in range(10)])
            cur.execute("INSERT INTO roster_entries(company_id,entry_id,employee_id,service_date,direction,cab_id,scheduled_time,status) VALUES('acme',%s,%s,%s,'drop','C_AC_D01','20:30','active')", (approval_entry, employees[10], date(2026, 11, 20)))
            cur.execute("INSERT INTO roster_entries(company_id,entry_id,employee_id,service_date,direction,cab_id,scheduled_time,status) VALUES('zenith',%s,%s,%s,'drop',%s,'21:00','active')", (f"ZENT{suffix}", zenith_employee, date(2026, 11, 20), zenith_cab))
    data = {"suffix": suffix, "employees": employees, "manager": manager, "zenith_manager": zenith_manager, "zenith_employee": zenith_employee, "cab": cab, "zenith_cab": zenith_cab, "entries": entries, "approval_entry": approval_entry}
    yield data
    with fixture_conn.transaction():
        with fixture_conn.cursor() as cur:
            cur.execute("DELETE FROM notifications WHERE company_id='acme' AND employee_id=ANY(%s)", (employees + [manager],))
            cur.execute("DELETE FROM shift_change_requests WHERE company_id='acme' AND employee_id=ANY(%s)", (employees + [manager],))
            cur.execute("DELETE FROM roster_entries WHERE company_id='acme' AND employee_id=ANY(%s)", (employees + [manager],))
            cur.execute("DELETE FROM employees WHERE company_id='acme' AND employee_id=ANY(%s)", (employees + [manager],))
            cur.execute("DELETE FROM roster_entries WHERE company_id='zenith' AND employee_id=%s", (zenith_employee,))
            cur.execute("DELETE FROM employees WHERE company_id='zenith' AND employee_id=ANY(%s)", ([zenith_manager, zenith_employee],))
            cur.execute("DELETE FROM cabs WHERE company_id='acme' AND cab_id=%s", (cab,))
            cur.execute("DELETE FROM cabs WHERE company_id='zenith' AND cab_id=%s", (zenith_cab,))
    fixture_conn.close()
