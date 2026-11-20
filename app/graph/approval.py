"""Durable human approval graph for roster changes."""

from __future__ import annotations

import logging
import uuid

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from app.auth import Principal
from app.changes import apply_approved, create_pending_request, reject_request
from app.db import get_pool

logger = logging.getLogger(__name__)
_approval_graph = None


class ApprovalState(TypedDict, total=False):
    company_id: str
    employee_id: str
    actor_role: str
    entry_id: str
    new_time: str
    cab_id: str
    idempotency_key: str
    request_id: str
    created: bool
    decision: dict
    status: str


def build_approval_graph(checkpointer):
    def create_request(state: ApprovalState):
        principal = Principal(state["company_id"], state["employee_id"], state.get("actor_role", "employee"))
        rid, status, created = create_pending_request(principal, state["entry_id"], state["new_time"], state["cab_id"], state["idempotency_key"], state["request_id"])
        return {"request_id": rid, "status": status, "created": created}

    def await_manager(state: ApprovalState):
        decision = interrupt({"request_id": state["request_id"], "company_id": state["company_id"], "employee_id": state["employee_id"]})
        return {"decision": decision}

    def apply_or_reject(state: ApprovalState):
        decision = state["decision"]
        if decision.get("decision") == "approve":
            status = apply_approved(state["company_id"], state["request_id"], decision["decided_by"], decision.get("note"))
        else:
            status = reject_request(state["company_id"], state["request_id"], decision["decided_by"], decision.get("note"))
        return {"status": status}

    def notify(state: ApprovalState):
        _notify_request(state["company_id"], state["request_id"], state.get("status", "failed"))
        return {}

    graph = StateGraph(ApprovalState)
    graph.add_node("create_request", create_request)
    graph.add_node("await_manager", await_manager)
    graph.add_node("apply_or_reject", apply_or_reject)
    graph.add_node("notify", notify)
    graph.add_edge(START, "create_request")
    graph.add_edge("create_request", "await_manager")
    graph.add_edge("await_manager", "apply_or_reject")
    graph.add_edge("apply_or_reject", "notify")
    graph.add_edge("notify", END)
    return graph.compile(checkpointer=checkpointer)


def _notify_request(company_id, request_id, status):
    messages = {
        "applied": "Your manager approved the change. Your roster has been updated.",
        "rejected": "Your manager declined the change. Your roster is unchanged. Contact the transport desk if you need help.",
        "approved_not_applied": "Your manager approved the change, but the cab filled up. Your roster is unchanged. Please choose another cab.",
        "expired": "Your approval request expired. Your roster is unchanged. Please submit a new request.",
        "failed": "We could not apply the approved change. Your roster is unchanged. Contact the transport desk.",
    }
    with get_pool().connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT employee_id,entry_id,new_cab_id,new_time FROM shift_change_requests WHERE company_id=%s AND id=%s", (company_id, request_id))
        row = cur.fetchone()
        if not row: return
        message = messages.get(status, messages["failed"])
        if status == "approved_not_applied":
            cur.execute("""SELECT c.route_id,c.direction,r.service_date FROM cabs c JOIN roster_entries r ON r.company_id=c.company_id AND r.entry_id=%s WHERE c.company_id=%s AND c.cab_id=%s""", (row[1], company_id, row[2]))
            route = cur.fetchone()
            if route:
                cur.execute("""SELECT c.cab_id,c.departure_time,c.capacity-COUNT(r.entry_id) AS free FROM cabs c LEFT JOIN roster_entries r ON r.company_id=c.company_id AND r.cab_id=c.cab_id AND r.service_date=%s AND r.status IN ('active','pending_change') WHERE c.company_id=%s AND c.route_id=%s AND c.direction=%s AND c.cab_id<>%s GROUP BY c.company_id,c.cab_id HAVING c.capacity-COUNT(r.entry_id)>0 ORDER BY ABS(EXTRACT(EPOCH FROM (c.departure_time-%s::time))) LIMIT 3""", (route[2], company_id, route[0], route[1], row[2], row[3]))
                options = cur.fetchall()
                if options:
                    message += " Nearest available options: " + ", ".join(f"{opt[1].strftime('%H:%M')} (cab {opt[0]}, {opt[2]} seats free)" for opt in options) + "."
        cur.execute("INSERT INTO notifications(company_id,employee_id,request_id,message) SELECT %s,%s,%s,%s WHERE NOT EXISTS(SELECT 1 FROM notifications WHERE request_id=%s)", (company_id, row[0], request_id, message, request_id))


def set_approval_graph(graph):
    global _approval_graph
    _approval_graph = graph


def get_approval_graph():
    return _approval_graph


def start_approval(principal, entry_id, new_time, cab_id, idempotency_key):
    graph = get_approval_graph()
    if graph is None:
        return None, "APPROVAL_UNAVAILABLE"
    request_id = str(uuid.uuid5(uuid.NAMESPACE_URL, idempotency_key))
    config = {"configurable": {"thread_id": f"approval:{request_id}"}}
    snapshot = graph.get_state(config)
    if snapshot and snapshot.values:
        with get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT id,status FROM shift_change_requests WHERE company_id=%s AND id=%s", (principal.company_id, request_id))
            row = cur.fetchone()
        return (str(row[0]), row[1]) if row else (request_id, "pending_approval")
    graph.invoke({"company_id": principal.company_id, "employee_id": principal.employee_id, "actor_role": principal.role, "entry_id": entry_id, "new_time": new_time, "cab_id": cab_id, "idempotency_key": idempotency_key, "request_id": request_id}, config=config)
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id,status FROM shift_change_requests WHERE company_id=%s AND id=%s", (principal.company_id, request_id))
        row = cur.fetchone()
    return (str(row[0]), row[1]) if row else (request_id, "failed")


def notify_request(company_id, request_id, status):
    _notify_request(company_id, request_id, status)
