"""
tests/test_auth.py — unit tests for JWT verification in app/auth.py

Tests run without a database connection (pure JWT logic).
"""

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from app.auth import _decode, verify_token, Principal
from app.config import settings
from app.main import app

SECRET = settings.jwt_secret
client = TestClient(app, raise_server_exceptions=False)


def _make_token(
    sub: str = "E101",
    company_id: str = "acme",
    role: str = "employee",
    secret: str = SECRET,
    exp_delta: int = 3600,
    omit_claims: list[str] | None = None,
) -> str:
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": sub,
        "company_id": company_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(seconds=exp_delta),
    }
    for claim in (omit_claims or []):
        payload.pop(claim, None)
    return jwt.encode(payload, secret, algorithm="HS256")


# ---------------------------------------------------------------------------
# _decode() unit tests (no FastAPI stack)
# ---------------------------------------------------------------------------

class TestDecodeUnit:
    def test_valid_token_returns_claims(self):
        token = _make_token()
        claims = _decode(token)
        assert claims["sub"] == "E101"
        assert claims["company_id"] == "acme"
        assert claims["role"] == "employee"

    def test_forged_signature_raises_401(self):
        token = _make_token(secret="wrong-secret")
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _decode(token)
        assert exc_info.value.status_code == 401
        assert "not valid" in exc_info.value.detail.lower()

    def test_expired_token_raises_401(self):
        token = _make_token(exp_delta=-1)   # already expired
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _decode(token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_missing_company_id_raises_401(self):
        token = _make_token(omit_claims=["company_id"])
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _decode(token)
        assert exc_info.value.status_code == 401

    def test_missing_sub_raises_401(self):
        token = _make_token(omit_claims=["sub"])
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _decode(token)
        assert exc_info.value.status_code == 401

    def test_invalid_role_raises_401(self):
        token = _make_token(role="admin")   # not in ("employee", "manager")
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            _decode(token)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# HTTP-level tests via TestClient (uses FastAPI middleware stack)
# ---------------------------------------------------------------------------

class TestAuthHTTP:
    def test_missing_auth_header_returns_401(self):
        resp = client.get("/me")
        assert resp.status_code == 401
        assert "not valid" in resp.json()["detail"].lower()

    def test_valid_token_returns_200(self):
        token = _make_token()
        resp = client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["company_id"] == "acme"
        assert body["employee_id"] == "E101"
        assert body["role"] == "employee"

    def test_forged_token_returns_401(self):
        token = _make_token(secret="totally-wrong-key")
        resp = client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_expired_token_returns_401(self):
        token = _make_token(exp_delta=-60)
        resp = client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_no_stack_trace_in_401_body(self):
        """Auth errors must never leak internal details."""
        token = _make_token(secret="bad-secret")
        resp = client.get("/me", headers={"Authorization": f"Bearer {token}"})
        body = resp.text
        assert "traceback" not in body.lower()
        assert "exception" not in body.lower()
        assert "jwt" not in body.lower()

    def test_role_escalation_blocked(self):
        """A token that claims role=manager must be accepted as manager —
        but we ensure a forged role escalation (tampered token) is rejected."""
        # Build a legit employee token, then tamper with it.
        token = _make_token(role="employee")
        # Decode without verification, modify, re-sign with wrong secret.
        import base64, json
        header_b64, payload_b64, sig = token.split(".")
        # Add padding
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
        claims["role"] = "manager"
        forged = jwt.encode(claims, "wrong-key", algorithm="HS256")
        resp = client.get("/me", headers={"Authorization": f"Bearer {forged}"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Dev token endpoint
# ---------------------------------------------------------------------------

class TestDevToken:
    def test_dev_token_issues_valid_jwt(self):
        resp = client.post(
            "/dev/token",
            json={"company_id": "acme", "employee_id": "E101", "role": "employee"},
        )
        assert resp.status_code == 200
        token = resp.json()["token"]
        # Use the token to call /me
        me_resp = client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp.status_code == 200
        assert me_resp.json()["company_id"] == "acme"

    def test_dev_token_respects_role(self):
        resp = client.post(
            "/dev/token",
            json={"company_id": "zenith", "employee_id": "MGR002", "role": "manager"},
        )
        token = resp.json()["token"]
        me_resp = client.get("/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp.json()["role"] == "manager"


def test_unknown_company_or_employee_is_401():
    """SPEC §5: a correctly signed token for someone not in the directory is rejected."""
    from fastapi.testclient import TestClient
    from app.main import app
    from tests.conftest import make_token
    with TestClient(app, raise_server_exceptions=False) as client:
        for company, employee in (("nocorp", "E101"), ("acme", "E999")):
            r = client.get("/me", headers={"Authorization": f"Bearer {make_token(company, employee)}"})
            assert r.status_code == 401
            assert r.json()["detail"] == "Your session is not valid. Please sign in again."
        assert client.get("/me", headers={"Authorization": f"Bearer {make_token('acme', 'E101')}"}).status_code == 200
