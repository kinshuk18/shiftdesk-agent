"""
app/main.py — FastAPI application entry point (Stage 1 + Stage 2).

Stage 1: health check, dev token issuer, DB connection lifecycle.
Stage 2: POST /chat — runs the LangGraph chat graph with a 4.5-second hard
         deadline; falls back to a plain template message on timeout or error.

SPEC §2: sync endpoint, runs graph in a thread-pool worker.
SPEC §4: identity comes only from the JWT dependency.
SPEC §12: global exception handler — logs traceback server-side, returns
          plain message + trace_id; never a stack trace.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from contextlib import asynccontextmanager
from typing import Optional

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from app.auth import make_dev_token, verify_token, Principal
from app.config import settings
from app.db import close_pool, get_pool
from app.messages import GENERIC_FAILURE, GENERIC_ERROR

logger = logging.getLogger(__name__)

# Hard deadline for a single /chat turn.
_CHAT_DEADLINE_SECONDS = 4.5

# Thread pool used to run the synchronous graph without blocking the event loop.
_executor = ThreadPoolExecutor(max_workers=8)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm():
    provider = settings.llm_provider.lower()
    if provider == "fake":
        from app.llm.fake import FakeLLM
        return FakeLLM()
    if provider == "groq":
        import os
        os.environ.setdefault("GROQ_MODEL", settings.groq_model)
        from app.llm.groq import GroqLLM
        return GroqLLM()
    if provider == "gemini":
        import os
        os.environ.setdefault("GEMINI_MODEL", settings.gemini_model)
        from app.llm.gemini import GeminiLLM
        return GeminiLLM()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


# ---------------------------------------------------------------------------
# Lifespan — build graph once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open the app-level connection pool (used by tools / repo).
    get_pool()

    # Build a separate autocommit pool for PostgresSaver.
    pg_pool = ConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=True,
    )

    from langgraph.checkpoint.postgres import PostgresSaver
    checkpointer = PostgresSaver(pg_pool)
    checkpointer.setup()

    from app.graph.chat import build_chat_graph
    # Inject the LLM singleton into the graph module so all nodes share one client.
    import app.graph.chat as _chat_mod
    _chat_mod._llm_singleton = _build_llm()

    app.state.chat_graph = build_chat_graph(checkpointer=checkpointer)
    from app.graph.approval import build_approval_graph, set_approval_graph
    approval_graph = build_approval_graph(checkpointer=checkpointer)
    app.state.approval_graph = approval_graph
    set_approval_graph(approval_graph)
    app.state.pg_pool = pg_pool

    yield

    set_approval_graph(None)
    close_pool()
    pg_pool.close()
    # Module-level executor is shared by every app lifespan (tests build several
    # TestClients on the same app); shutting it down breaks the next lifespan.
    # _executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(title="ShiftDesk", lifespan=lifespan)

from app.manager import router as manager_router
app.include_router(manager_router)


# ---------------------------------------------------------------------------
# Global exception handler (SPEC §12)
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    trace_id = str(uuid.uuid4())
    logger.error(
        "Unhandled exception trace_id=%s path=%s\n%s",
        trace_id,
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "reply": GENERIC_ERROR.format(
                desk_phone=settings.desk_phone,
                trace_id=trace_id,
            ),
            "status": "error",
            "request_id": None,
            "options": [],
            "trace_id": trace_id,
        },
    )


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


# ---------------------------------------------------------------------------
# POST /chat — the main Stage-2 endpoint.
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    status: str
    request_id: Optional[str] = None
    options: list[str] = []
    trace_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    request: Request,
    principal: Principal = Depends(verify_token),
) -> ChatResponse:
    """
    Process one chat message.

    - Identity comes exclusively from the JWT (principal).
    - Any company_id / employee_id in the body is ignored (SPEC §4, §13).
    - Runs the LangGraph graph in a thread-pool worker.
    - Hard 4.5-second deadline; on timeout returns a plain fallback reply.
    - 1000-char truncation; empty message handled gracefully.
    """
    trace_id = str(uuid.uuid4())

    # SPEC §13: truncate at 1000 chars; empty message gets plain reply, no LLM.
    raw_message = body.message[:1000]

    if not raw_message.strip():
        return ChatResponse(
            reply=(
                "I didn't receive a message. "
                "You can ask me to show your roster, change your pickup or drop time, "
                "or check the status of a request."
            ),
            status="ok",
            request_id=None,
            options=[],
            trace_id=trace_id,
        )

    thread_id = f"chat:{principal.company_id}:{principal.employee_id}"
    graph = request.app.state.chat_graph

    # Build the initial state for this invocation.
    input_state = {
        "current_message": raw_message,
        "principal": {
            "company_id": principal.company_id,
            "employee_id": principal.employee_id,
            "role": principal.role,
        },
        "trace_id": trace_id,
        "llm_calls_used": 0,
        "tool_results": [],
        "tool_errors": [],
        "options": [],
        # Per-turn outputs must be reset: the checkpointer merges this input into
        # the saved state, and a leftover "reply" short-circuits every node.
        "reply": "",
        "status": "ok",
        "request_id": None,
        "plan": None,
        "plan_error": None,
    }

    config = {"configurable": {"thread_id": thread_id}}

    def _run_graph():
        result = graph.invoke(input_state, config=config)
        return result

    try:
        future = _executor.submit(_run_graph)
        result = future.result(timeout=_CHAT_DEADLINE_SECONDS)
    except (FuturesTimeout, TimeoutError):
        logger.error(
            "chat graph timed out after %.1fs trace_id=%s thread=%s",
            _CHAT_DEADLINE_SECONDS,
            trace_id,
            thread_id,
        )
        return ChatResponse(
            reply=GENERIC_FAILURE.format(desk_phone=settings.desk_phone),
            status="error",
            request_id=None,
            options=[],
            trace_id=trace_id,
        )
    except Exception:
        logger.error(
            "chat graph raised an exception trace_id=%s\n%s",
            trace_id,
            traceback.format_exc(),
        )
        return ChatResponse(
            reply=GENERIC_ERROR.format(
                desk_phone=settings.desk_phone,
                trace_id=trace_id,
            ),
            status="error",
            request_id=None,
            options=[],
            trace_id=trace_id,
        )

    return ChatResponse(
        reply=result.get("reply", ""),
        status=result.get("status", "ok"),
        request_id=result.get("request_id"),
        options=result.get("options", []),
        trace_id=trace_id,
    )
