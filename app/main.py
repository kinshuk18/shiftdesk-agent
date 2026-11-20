"""
app/main.py — FastAPI application entry point.

Stage 1: health check, dev token issuer, DB connection lifecycle.
LangGraph and chat endpoints are added in later stages.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from app.auth import make_dev_token, verify_token, Principal
from app.config import settings
from app.db import close_pool, get_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open the connection pool on startup; close on shutdown.
    get_pool()
    yield
    close_pool()


app = FastAPI(title="ShiftDesk", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health check — no auth required.
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    """Return 200 if the app is running and can reach the database."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dev token issuer — only mounted when ENABLE_DEV_ISSUER=true.
# This is the mock IdP for local development and tests; never expose in prod.
# ---------------------------------------------------------------------------

class DevTokenRequest(BaseModel):
    company_id: str
    employee_id: str
    role: str = "employee"
    ttl_minutes: int = 60


class DevTokenResponse(BaseModel):
    token: str


if settings.enable_dev_issuer:

    @app.post("/dev/token", response_model=DevTokenResponse)
    def dev_token(body: DevTokenRequest):
        """
        Issue a signed JWT for development and testing.
        Only available when ENABLE_DEV_ISSUER=true.
        """
        token = make_dev_token(
            company_id=body.company_id,
            employee_id=body.employee_id,
            role=body.role,
            ttl_minutes=body.ttl_minutes,
        )
        return DevTokenResponse(token=token)


# ---------------------------------------------------------------------------
# Protected example endpoint (proves auth dependency works).
# ---------------------------------------------------------------------------

@app.get("/me")
def get_me(principal: Principal = Depends(verify_token)):
    """Return the caller's identity from the JWT (no DB query)."""
    return {
        "company_id": principal.company_id,
        "employee_id": principal.employee_id,
        "role": principal.role,
    }
