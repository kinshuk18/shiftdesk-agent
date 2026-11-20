"""
app/auth.py — JWT verification and FastAPI dependency.

JWT claims expected: sub (employee_id), company_id, role, exp, iat.
Algorithm: HS256. Secret from settings.jwt_secret.

All auth failures return 401 with a plain English message — no stack traces,
no field names that reveal implementation details, no 403 that leaks existence.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    """Verified identity extracted from a valid JWT. Immutable."""
    company_id: str
    employee_id: str
    role: str       # "employee" | "manager"


def _decode(token: str) -> dict:
    """
    Decode and verify a HS256 JWT. Raises HTTPException(401) on any failure.
    Returns the raw claims dict on success.
    """
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            options={"require": ["sub", "exp", "iat", "company_id", "role"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Your session has expired. Please sign in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Your session is not valid. Please sign in again.")

    # Validate role value explicitly — the token is untrusted input.
    if claims.get("role") not in ("employee", "manager"):
        raise HTTPException(status_code=401, detail="Your session is not valid. Please sign in again.")

    return claims


def verify_token(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    """
    FastAPI dependency. Extracts Bearer token from Authorization header,
    verifies it, and returns a Principal.

    company_id and employee_id come ONLY from the verified token —
    never from request body or query params.
    """
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session is not valid. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    claims = _decode(creds.credentials)
    # SPEC §5: unknown company / unknown employee -> 401 (a validly signed token
    # for someone who is not in the directory is still not a session).
    from app.db import get_pool
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM employees WHERE company_id=%s AND employee_id=%s",
            (claims["company_id"], claims["sub"]),
        )
        known = cur.fetchone() is not None
    if not known:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session is not valid. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Principal(
        company_id=claims["company_id"],
        employee_id=claims["sub"],
        role=claims["role"],
    )


def require_manager(principal: Principal = Depends(verify_token)) -> Principal:
    """Additional dependency: also requires role == 'manager'."""
    if principal.role != "manager":
        raise HTTPException(status_code=401, detail="Your session is not valid. Please sign in again.")
    return principal


# ---------------------------------------------------------------------------
# Mock token issuer (only mounted when ENABLE_DEV_ISSUER=true)
# ---------------------------------------------------------------------------

def make_dev_token(company_id: str, employee_id: str, role: str = "employee", ttl_minutes: int = 60) -> str:
    """
    Issue a signed JWT for development/testing.
    Not mounted in production (gated by ENABLE_DEV_ISSUER).
    """
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": employee_id,
        "company_id": company_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
