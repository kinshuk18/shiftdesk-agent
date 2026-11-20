"""
app/tools.py — the 5 allow-listed tools (SPEC §8).

Security invariants:
  - company_id and employee_id come ONLY from Principal (injected by code,
    never accepted from the LLM or request body).
  - Every tool is wrapped in a 1.5-second timeout.
  - Failures never raise into the graph — always return ToolResult(ok=False, ...).

Stage 3 adds transactional approvals for R1/R2 exceptions when callers opt in.
"""

from __future__ import annotations

import hashlib
import logging
import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Optional

from app.auth import Principal
from app.clock import clock
from app.db import get_pool
from app.repo import (
    get_cab,
    get_cabs_for_route,
    get_company,
    get_roster,
    get_roster_entry,
    count_active_bookings,
)
from app.rules import (
    check_cutoff,
    check_max_delta,
    check_service_window,
    check_capacity,
)

logger = logging.getLogger(__name__)

_TOOL_TIMEOUT_SECONDS = 1.5
_tool_executor = ThreadPoolExecutor(max_workers=8)
_EXEC = _tool_executor


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    ok: bool
    data: Optional[Any] = None      # structured result on success
    error_code: Optional[str] = None  # e.g. "R1_CUTOFF_VIOLATED", "TIMEOUT"
    error_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool: get_my_roster
# ---------------------------------------------------------------------------


def get_my_roster(principal: Principal, date: str) -> ToolResult:
    """
    Return the employee's roster entries for a given date.
    company_id / employee_id come from Principal — never from args.
    Parameter name is 'date' to match the planner's tool call schema.
    """
    date_str = date  # local alias for clarity
    try:
        service_date = datetime.date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return ToolResult(ok=False, error_code="INVALID_DATE", error_params={"date": date_str})

    def _query():
        with get_pool().connection() as conn:
            entries = get_roster(conn, principal, service_date)
            company = get_company(conn, principal.company_id)
        return entries, company

    try:
        entries, company = _query()
    except Exception as exc:
        logger.exception("get_my_roster failed: %s", exc)
        return ToolResult(ok=False, error_code="DB_ERROR", error_params={"step": "reading your roster"})

    return ToolResult(
        ok=True,
        data={
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "direction": e.direction,
                    "cab_id": e.cab_id,
                    "scheduled_time": e.scheduled_time.strftime("%H:%M"),
                    "service_date": e.service_date.isoformat(),
                    "status": e.status,
                }
                for e in entries
            ],
            "company": {
                "cutoff_minutes": company.cutoff_minutes if company else None,
                "max_shift_delta_minutes": company.max_shift_delta_minutes if company else None,
                "service_start": company.service_start.strftime("%H:%M") if company else None,
                "service_end": company.service_end.strftime("%H:%M") if company else None,
                "desk_phone": company.desk_phone if company else None,
            } if company else {},
        },
    )


# ---------------------------------------------------------------------------
# Tool: check_cab_options
# ---------------------------------------------------------------------------


def check_cab_options(principal: Principal, entry_id: str, new_time: str) -> ToolResult:
    """
    Given an entry_id and proposed new_time (HH:MM), return:
      - rule results (R1, R2, R3, R4)
      - exact_match cab (if one at exactly new_time with free seat)
      - nearest_slots (up to 3 cabs, nearest in time, with free seats)
    company_id / employee_id come from Principal.
    """
    # Parse new_time.
    try:
        hh, mm = new_time.split(":")
        new_t = datetime.time(int(hh), int(mm))
    except (ValueError, AttributeError):
        return ToolResult(ok=False, error_code="INVALID_TIME", error_params={"new_time": new_time})

    def _query():
        with get_pool().connection() as conn:
            entry = get_roster_entry(conn, principal, entry_id)
            if entry is None:
                return None, None, [], None
            company = get_company(conn, principal.company_id)
            # Cabs on the same route and direction.
            cab_rows = get_cabs_for_route(
                conn, principal.company_id, entry.cab_id, entry.direction
            )
            # Actually we need the route_id from the current cab.
            current_cab = get_cab(conn, principal.company_id, entry.cab_id)
            if current_cab is None:
                return entry, company, [], None
            route_cabs = get_cabs_for_route(
                conn, principal.company_id, current_cab.route_id, entry.direction
            )
            # Count bookings for each candidate cab.
            candidates = []
            for cab in route_cabs:
                booked = count_active_bookings(conn, principal.company_id, cab.cab_id, entry.service_date)
                free = cab.capacity - booked
                candidates.append((cab, free))
            return entry, company, candidates, current_cab

    try:
        entry, company, candidates, current_cab = _query()
    except Exception as exc:
        logger.exception("check_cab_options failed: %s", exc)
        return ToolResult(ok=False, error_code="DB_ERROR", error_params={"step": "checking cab availability"})

    if entry is None:
        return ToolResult(ok=False, error_code="ENTRY_NOT_FOUND", error_params={"entry_id": entry_id})

    now = clock.now()

    # Run rules.
    r1 = check_cutoff(now, entry.scheduled_time, company.cutoff_minutes)
    r2 = check_max_delta(entry.scheduled_time, new_t, company.max_shift_delta_minutes)
    r3 = check_service_window(new_t, company.service_start, company.service_end)

    # Find exact match and nearest slots.
    exact_match = None
    nearest: list[dict] = []

    for cab, free in sorted(candidates, key=lambda x: abs(
        (x[0].departure_time.hour * 60 + x[0].departure_time.minute)
        - (new_t.hour * 60 + new_t.minute)
    )):
        cab_time = cab.departure_time
        if cab_time == new_t and free > 0 and exact_match is None:
            exact_match = {
                "cab_id": cab.cab_id,
                "departure_time": cab_time.strftime("%H:%M"),
                "free_seats": free,
            }
        elif free > 0 and len(nearest) < 3:
            nearest.append({
                "cab_id": cab.cab_id,
                "departure_time": cab_time.strftime("%H:%M"),
                "free_seats": free,
            })

    r4_seats = exact_match["free_seats"] if exact_match else (nearest[0]["free_seats"] if nearest else 0)
    r4 = check_capacity(r4_seats if exact_match else 0)

    return ToolResult(
        ok=True,
        data={
            "entry": {
                "entry_id": entry.entry_id,
                "direction": entry.direction,
                "scheduled_time": entry.scheduled_time.strftime("%H:%M"),
                "cab_id": entry.cab_id,
            },
            "rules": {
                "r1": {"ok": r1.ok, "code": r1.code, "params": r1.message_params},
                "r2": {"ok": r2.ok, "code": r2.code, "params": r2.message_params},
                "r3": {"ok": r3.ok, "code": r3.code, "params": r3.message_params},
                "r4_exact": {"ok": bool(exact_match), "code": "OK" if exact_match else "R4_NO_SEAT"},
            },
            "exact_match": exact_match,
            "nearest_slots": nearest,
            "escalatable": not r1.ok or not r2.ok,  # R1/R2 violations only
        },
    )


# ---------------------------------------------------------------------------
# Idempotency key helper
# ---------------------------------------------------------------------------


def _idempotency_key(company_id: str, employee_id: str, entry_id: str, new_cab_id: str, new_time: str, attempt: int = 0) -> str:
    raw = f"{company_id}|{employee_id}|{entry_id}|{new_cab_id}|{new_time}"
    if attempt > 0:
        raw += f"|{attempt}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Tool: raise_shift_change
# ---------------------------------------------------------------------------


def raise_shift_change(principal: Principal, entry_id: str, new_time: str, cab_id: str, escalate: bool = False) -> ToolResult:
    """Validate R1–R3, then apply directly or create a pending approval."""
    try:
        hh, mm = new_time.split(":")
        new_t = datetime.time(int(hh), int(mm))
    except (ValueError, AttributeError):
        return ToolResult(ok=False, error_code="INVALID_TIME", error_params={"new_time": new_time})
    # Canonical HH:MM so "9:00" and "09:00" give the same idempotency key.
    new_time = new_t.strftime("%H:%M")
    try:
        with get_pool().connection() as conn:
            entry = get_roster_entry(conn, principal, entry_id)
            if entry is None:
                return ToolResult(ok=False, error_code="ENTRY_NOT_FOUND", error_params={"entry_id": entry_id})
            company = get_company(conn, principal.company_id)
            if company is None:
                return ToolResult(ok=False, error_code="COMPANY_NOT_FOUND")
            r1 = check_cutoff(clock.now(), entry.scheduled_time, company.cutoff_minutes)
            r2 = check_max_delta(entry.scheduled_time, new_t, company.max_shift_delta_minutes)
            r3 = check_service_window(new_t, company.service_start, company.service_end)
            if not r3.ok:
                return ToolResult(ok=False, error_code=r3.code, error_params=r3.message_params)
            if not r1.ok or not r2.ok:
                failed = r1 if not r1.ok else r2
                if not escalate:
                    return ToolResult(ok=False, error_code=failed.code, error_params=failed.message_params)
                with conn.cursor() as cur:
                    cur.execute("SELECT capacity FROM cabs WHERE company_id=%s AND cab_id=%s", (principal.company_id, cab_id))
                    cab = cur.fetchone()
                    if cab is None:
                        return ToolResult(ok=False, error_code="CAB_NOT_FOUND")
                    cur.execute("SELECT COUNT(*) FROM roster_entries WHERE company_id=%s AND cab_id=%s AND service_date=%s AND entry_id<>%s AND status IN ('active','pending_change')", (principal.company_id, cab_id, entry.service_date, entry_id))
                    # R4 is never escalated. Handled after this block so the read
                    # connection is released before _seat_taken() takes its own.
                    seat_taken = cur.fetchone()[0] >= cab[0]
                escalate_needed = True
                with conn.cursor() as cur:
                    cur.execute("SELECT name FROM employees WHERE company_id=%s AND employee_id=%s", (principal.company_id, company.transport_manager_id))
                    mgr = cur.fetchone()
                manager_name = mgr[0] if mgr else None
            else:
                seat_taken = False
                escalate_needed = False
        if seat_taken:
            return _seat_taken(principal, entry_id, new_time)
        # All read connections close before acquiring a connection for key lookup or writes.
        from app.changes import idempotency_key_for, submit_direct
        _base_key, attempt = idempotency_key_for(principal.company_id, principal.employee_id, entry_id, cab_id, new_time)
        key = _idempotency_key(principal.company_id, principal.employee_id, entry_id, cab_id, new_time, attempt)
        if escalate_needed:
            from app.graph.approval import start_approval
            rid, status = start_approval(principal, entry_id, new_time, cab_id, key)
            if status == "APPROVAL_UNAVAILABLE":
                return ToolResult(ok=False, error_code=status)
            if status == "pending_approval":
                return ToolResult(ok=True, data={"request_id": rid, "status": status, "manager_name": manager_name})
        else:
            rid, status = submit_direct(principal, entry_id, new_t, cab_id, key)
        if status == "SEAT_TAKEN":
            return _seat_taken(principal, entry_id, new_time)
        if status in ("CAB_NOT_FOUND", "ENTRY_NOT_FOUND"):
            return ToolResult(ok=False, error_code=status)
        return ToolResult(ok=True, data={"request_id": rid, "status": status})
    except Exception as exc:
        logger.exception("raise_shift_change failed: %s", exc)
        return ToolResult(ok=False, error_code="DB_ERROR", error_params={"step": "applying shift change"})



def _seat_taken(principal: Principal, entry_id: str, new_time: str) -> ToolResult:
    """SEAT_TAKEN plus the nearest free slots (best effort, never raises)."""
    nearest: list = []
    try:
        opts = check_cab_options(principal, entry_id, new_time)
        if opts.ok:
            nearest = opts.data.get("nearest_slots", [])
    except Exception:
        logger.exception("nearest-slot lookup after SEAT_TAKEN failed")
    return ToolResult(ok=False, error_code="SEAT_TAKEN", error_params={"nearest_slots": nearest})


# ---------------------------------------------------------------------------
# Tool: get_request_status
# ---------------------------------------------------------------------------


def get_request_status(principal: Principal) -> ToolResult:
    """
    Return the employee's open shift change requests.
    company_id / employee_id from Principal.
    """
    from app.changes import expire_stale
    expire_stale(principal.company_id)

    def _query():
        with get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, entry_id, old_cab_id, new_cab_id, new_time,
                           status, reason, created_at, decided_at, decision_note
                    FROM shift_change_requests
                    WHERE company_id = %s AND employee_id = %s
                    ORDER BY created_at DESC
                    LIMIT 10
                    """,
                    (principal.company_id, principal.employee_id),
                )
                rows = cur.fetchall()
        return rows

    try:
        rows = _query()
    except Exception as exc:
        logger.exception("get_request_status failed: %s", exc)
        return ToolResult(ok=False, error_code="DB_ERROR", error_params={"step": "fetching request status"})

    requests = [
        {
            "request_id": str(r[0]),
            "entry_id": r[1],
            "old_cab_id": r[2],
            "new_cab_id": r[3],
            "new_time": r[4].strftime("%H:%M") if r[4] else None,
            "status": r[5],
            "created_at": r[7].isoformat() if r[7] else None,
        }
        for r in rows
    ]
    return ToolResult(ok=True, data={"requests": requests})


# ---------------------------------------------------------------------------
# Tool: ask_clarification
# ---------------------------------------------------------------------------


def ask_clarification(
    principal: Principal,
    question_code: str,
    options: list[str],
) -> ToolResult:
    """
    No side effect — signals that the graph should ask the employee a question.
    Returns the question_code and options so the graph can compose the reply.
    """
    return ToolResult(
        ok=True,
        data={
            "question_code": question_code,
            "options": options,
        },
    )


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "get_my_roster": get_my_roster,
    "check_cab_options": check_cab_options,
    "raise_shift_change": raise_shift_change,
    "get_request_status": get_request_status,
    "ask_clarification": ask_clarification,
}


# Human step names for failure messages (SPEC §12: "naming the step").
TOOL_STEPS = {
    "get_my_roster": "reading your roster",
    "check_cab_options": "checking cab availability",
    "raise_shift_change": "applying your shift change",
    "get_request_status": "fetching your request status",
    "ask_clarification": "preparing a question",
}


def dispatch_tool(principal: Principal, tool_name: str, args: dict) -> ToolResult:
    """
    Dispatch a tool call with identity injected from Principal.
    tool_name must be in TOOL_REGISTRY (allow-list checked by Plan validator).
    args must NOT contain company_id or employee_id (enforced by Plan validator).
    """
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return ToolResult(
            ok=False,
            error_code="UNKNOWN_TOOL",
            error_params={"tool": tool_name},
        )
    try:
        fut = _tool_executor.submit(fn, principal, **args)
        return fut.result(timeout=1.5)
    except (TimeoutError, FuturesTimeoutError):
        logger.warning("Tool %s timed out after 1.5s", tool_name)
        return ToolResult(
            ok=False,
            error_code="TOOL_TIMEOUT",
            error_params={"tool": tool_name, "step": TOOL_STEPS.get(tool_name, "processing your request")},
        )
    except Exception as exc:
        logger.exception("Tool %s unexpected error: %s", tool_name, exc)
        return ToolResult(
            ok=False,
            error_code="TOOL_ERROR",
            error_params={"tool": tool_name, "error": str(exc), "step": TOOL_STEPS.get(tool_name, "processing your request")},
        )
