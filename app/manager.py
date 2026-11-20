"""Company scoped manager approval and employee request endpoints."""

from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from langgraph.types import Command

from app.auth import Principal, verify_token
from app.changes import apply_approved, expire_stale, reject_request
from app.db import get_pool
from app.graph.approval import get_approval_graph, notify_request
from app.repo import list_employee_shift_change_requests, list_shift_change_requests

logger = logging.getLogger(__name__)
router = APIRouter()


def _manager(principal: Principal):
    if principal.role != "manager":
        raise HTTPException(status_code=404, detail="Not Found")
    return principal


@router.get("/manager/requests")
def manager_requests(status: str = Query("pending"), principal: Principal = Depends(verify_token)):
    principal = _manager(principal)
    expire_stale(principal.company_id)
    requested_status = "pending_approval" if status == "pending" else status
    with get_pool().connection() as conn:
        return {"requests": list_shift_change_requests(conn, principal.company_id, requested_status)}


class DecisionBody(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=500)


@router.post("/manager/requests/{request_id}/decision")
def manager_decision(request_id: str, body: DecisionBody, principal: Principal = Depends(verify_token)):
    principal = _manager(principal)
    try:
        rid = str(uuid.UUID(request_id))
    except (ValueError, AttributeError):
        raise HTTPException(status_code=404, detail="Not Found")
    expire_stale(principal.company_id)
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT status FROM shift_change_requests WHERE company_id=%s AND id=%s", (principal.company_id, rid))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Not Found")
    if row[0] != "pending_approval":
        raise HTTPException(status_code=409, detail=f"This request is no longer waiting for a decision (it is {row[0].replace('_', ' ')}). Nothing was changed.")
    graph = get_approval_graph()
    config = {"configurable": {"thread_id": f"approval:{rid}"}}
    snapshot = graph.get_state(config) if graph else None
    paused = bool(snapshot and any(getattr(task, "interrupts", ()) for task in snapshot.tasks))
    if paused:
        graph.invoke(Command(resume={"decision": body.decision, "decided_by": principal.employee_id, "note": body.note}), config=config)
    else:
        logger.warning("Pending approval has no paused graph thread request_id=%s", rid)
        if body.decision == "approve":
            apply_approved(principal.company_id, rid, principal.employee_id, body.note)
        else:
            reject_request(principal.company_id, rid, principal.employee_id, body.note)
        notify_request(principal.company_id, rid, _read_status(principal.company_id, rid))
    return {"request_id": rid, "status": _read_status(principal.company_id, rid)}


def _read_status(company_id: str, request_id: str):
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT status FROM shift_change_requests WHERE company_id=%s AND id=%s", (company_id, request_id))
        row = cur.fetchone()
        return row[0] if row else None


@router.get("/requests/mine")
def my_requests(principal: Principal = Depends(verify_token)):
    expire_stale(principal.company_id)
    with get_pool().connection() as conn:
        requests = list_employee_shift_change_requests(conn, principal.company_id, principal.employee_id)
    for row in requests:
        row["request_id"] = str(row["request_id"])
        row["new_time"] = row["new_time"].strftime("%H:%M") if row["new_time"] else None
    return {"requests": requests}
