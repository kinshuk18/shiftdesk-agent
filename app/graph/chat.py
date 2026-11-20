"""
app/graph/chat.py — LangGraph chat graph with PostgresSaver.

SPEC §9: nodes: authorize → plan → validate_plan → execute_tools → compose_reply
         state: last 6 turns, pending_slots, awaiting, llm_calls_used
         thread_id: "chat:{company_id}:{employee_id}"

SPEC §4: identity injected from Principal; tool args never trusted for identity.
SPEC §12: failures return plain messages, never stack traces.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

from app.auth import Principal
from app.clock import clock
from app.config import settings
from app.llm.base import BudgetExhausted, LLMBudget, LLMTimeoutError, LLMQuotaError
from app.llm.planner import Plan, PlannerError, call_planner
from app.messages import (
    GENERIC_FAILURE,
    GENERIC_ERROR,
    TOOL_TIMEOUT,
    CAB_FILLED,
    format_rule_message,
)
from app.tools import ToolResult, dispatch_tool

logger = logging.getLogger(__name__)

# SPEC §9: keep last 6 turns.
MAX_HISTORY_TURNS = 6


# ---------------------------------------------------------------------------
# Graph state schema
# ---------------------------------------------------------------------------


class ChatState(TypedDict, total=False):
    # Conversation context
    principal: dict                     # serialised Principal (company_id, employee_id, role)
    history: list[dict]                 # list of {"role": "user"|"assistant", "content": str}
    current_message: str                # the raw user message this turn
    trace_id: str                       # UUID for observability

    # Planner output
    plan: Optional[dict]                # serialised Plan dict
    plan_error: Optional[str]           # set if planning failed

    # Tool execution results
    tool_results: list[dict]            # list of {"tool": str, "result": ToolResult.data|error}
    tool_errors: list[str]              # error codes encountered

    # Awaiting state (SPEC §9 ambiguity policy)
    awaiting: Optional[str]             # "time" | "direction" | "am_pm" | "which_entry" | None
    pending_slots: dict                 # e.g. {"entry_id": "...", "direction": "drop"}

    # LLM budget
    llm_calls_used: int

    # Final reply
    reply: str
    status: str                         # "ok" | "error" | "awaiting_input" | "pending_approval"
    request_id: Optional[str]
    options: list[str]                  # offered options for the user


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

_llm_singleton = None
_llm_lock = threading.Lock()


def _get_llm():
    global _llm_singleton
    with _llm_lock:
        if _llm_singleton is None:
            provider = settings.llm_provider.lower()
            if provider == "fake":
                from app.llm.fake import FakeLLM
                _llm_singleton = FakeLLM()
            elif provider == "groq":
                from app.llm.groq import GroqLLM
                _llm_singleton = GroqLLM()
            elif provider == "gemini":
                from app.llm.gemini import GeminiLLM
                _llm_singleton = GeminiLLM()
            else:
                raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
    return _llm_singleton


# ---------------------------------------------------------------------------
# Node: authorize
# ---------------------------------------------------------------------------


def node_authorize(state: ChatState) -> ChatState:
    """
    Validate that the principal is present and has required fields.
    In production this is already done by the FastAPI dependency before the
    graph is invoked, but we double-check here for defence-in-depth.
    """
    p = state.get("principal", {})
    if not p.get("company_id") or not p.get("employee_id"):
        return {
            **state,
            "reply": "Your session is not valid. Please sign in again.",
            "status": "error",
            "options": [],
        }
    return state


# ---------------------------------------------------------------------------
# Node: plan
# ---------------------------------------------------------------------------


def node_plan(state: ChatState) -> ChatState:
    """
    Call the LLM planner to turn the user message into a structured Plan.
    Uses the per-message LLMBudget; on failure returns a plain error reply.
    """
    if state.get("reply"):
        # Already short-circuited by authorize.
        return state

    budget = LLMBudget(max=4)
    budget._used = state.get("llm_calls_used", 0)

    # Build conversation history for context.
    history = state.get("history", [])

    llm = _get_llm()
    try:
        plan = call_planner(
            llm=llm,
            budget=budget,
            user_message=state.get("current_message", ""),
            history=history[-MAX_HISTORY_TURNS * 2:] if history else None,
        )
        return {
            **state,
            "plan": plan.model_dump(),
            "plan_error": None,
            "llm_calls_used": budget.used,
        }
    except BudgetExhausted:
        logger.warning("LLM budget exhausted during planning for trace %s", state.get("trace_id"))
        return {
            **state,
            "plan": None,
            "plan_error": "BUDGET_EXHAUSTED",
            "llm_calls_used": budget.used,
            "reply": GENERIC_FAILURE.format(desk_phone="the transport desk"),
            "status": "error",
            "options": [],
        }
    except (LLMTimeoutError, LLMQuotaError) as exc:
        logger.warning("LLM provider error: %s for trace %s", exc, state.get("trace_id"))
        return {
            **state,
            "plan": None,
            "plan_error": type(exc).__name__,
            "llm_calls_used": budget.used,
            "reply": GENERIC_FAILURE.format(desk_phone="the transport desk"),
            "status": "error",
            "options": [],
        }
    except PlannerError as exc:
        logger.warning("PlannerError: %s for trace %s", exc, state.get("trace_id"))
        return {
            **state,
            "plan": None,
            "plan_error": "PLANNER_ERROR",
            "llm_calls_used": budget.used,
            "reply": GENERIC_FAILURE.format(desk_phone="the transport desk"),
            "status": "error",
            "options": [],
        }
    except Exception as exc:
        logger.exception("Unexpected planner error for trace %s: %s", state.get("trace_id"), exc)
        return {
            **state,
            "plan": None,
            "plan_error": "UNEXPECTED",
            "llm_calls_used": budget.used,
            "reply": GENERIC_FAILURE.format(desk_phone="the transport desk"),
            "status": "error",
            "options": [],
        }


# ---------------------------------------------------------------------------
# Node: validate_plan
# ---------------------------------------------------------------------------


def node_validate_plan(state: ChatState) -> ChatState:
    """
    Code-level validation of the plan (SPEC §4).
    - If plan is None (error already set), pass through.
    - Merge pending_slots context (awaiting-slot follow-ups).
    - Reject any tool args that contain company_id / employee_id (belt-and-suspenders;
      pydantic already rejects these but we guard again here).
    """
    if state.get("reply") or state.get("plan") is None:
        return state

    plan_dict = state["plan"]
    tool_calls = plan_dict.get("tool_calls", [])

    # Security: strip identity from args (double-enforcement).
    forbidden = {"company_id", "employee_id"}
    for tc in tool_calls:
        args = tc.get("args", {})
        for key in list(args.keys()):
            if key in forbidden:
                del args[key]

    # Awaiting-slot context injection.
    awaiting = state.get("awaiting")
    pending = state.get("pending_slots", {})

    if awaiting and pending:
        # If the follow-up doesn't specify an entry_id, inject from context.
        for tc in tool_calls:
            args = tc.get("args", {})
            if tc.get("tool") in ("check_cab_options", "raise_shift_change"):
                if "entry_id" not in args and "entry_id" in pending:
                    args["entry_id"] = pending["entry_id"]
                if "direction" not in args and "direction" in pending:
                    # direction is part of context but not an arg — stored in pending
                    pass

    return {**state, "plan": plan_dict}


# ---------------------------------------------------------------------------
# Node: execute_tools
# ---------------------------------------------------------------------------


def node_execute_tools(state: ChatState) -> ChatState:
    """
    Execute the tool calls in the plan sequentially.
    Failures never raise — always return ToolResult(ok=False, ...).
    """
    if state.get("reply") or state.get("plan") is None:
        return state

    plan_dict = state["plan"]
    tool_calls = plan_dict.get("tool_calls", [])

    p = state["principal"]
    principal = Principal(
        company_id=p["company_id"],
        employee_id=p["employee_id"],
        role=p["role"],
    )

    results = []
    errors = []
    new_pending = dict(state.get("pending_slots", {}))

    for tc in tool_calls:
        tool_name = tc.get("tool", "")
        args = dict(tc.get("args", {}))
        result: ToolResult = dispatch_tool(principal, tool_name, args)

        result_dict = {
            "tool": tool_name,
            "ok": result.ok,
            "data": result.data,
            "error_code": result.error_code,
            "error_params": result.error_params,
        }
        results.append(result_dict)
        if not result.ok:
            errors.append(result.error_code or "UNKNOWN_ERROR")
        else:
            # Update pending_slots from roster results (store entry context for follow-ups).
            if tool_name == "get_my_roster" and result.data:
                entries = result.data.get("entries", [])
                if len(entries) == 1:
                    new_pending["entry_id"] = entries[0]["entry_id"]
                    new_pending["direction"] = entries[0]["direction"]

    # Determine awaiting state from plan.
    plan_ambiguity = plan_dict.get("ambiguity", "none")
    new_awaiting = state.get("awaiting")
    if plan_ambiguity != "none":
        new_awaiting = plan_ambiguity
    elif any(r["tool"] == "raise_shift_change" and r["ok"] for r in results):
        new_awaiting = None  # completed
    elif plan_dict.get("intent") not in ("unknown",):
        # Clear awaiting only if we got a concrete result.
        pass

    return {
        **state,
        "tool_results": results,
        "tool_errors": errors,
        "pending_slots": new_pending,
        "awaiting": new_awaiting,
    }


# ---------------------------------------------------------------------------
# Node: compose_reply
# ---------------------------------------------------------------------------


def node_compose_reply(state: ChatState) -> ChatState:
    """
    Compose a plain-text reply from structured tool results.
    Messages come from templates (SPEC §4 / §12) — the LLM never writes
    rule explanations or error messages.
    """
    if state.get("reply"):
        # Already set (error path).
        return _append_to_history(state)

    plan_dict = state.get("plan") or {}
    tool_results = state.get("tool_results", [])
    p_dict = state.get("principal", {})
    options: list[str] = []
    request_id: Optional[str] = None
    status = "ok"

    # -- Helper: get company desk_phone from tool results --
    desk_phone = "the transport desk"
    for r in tool_results:
        if r["tool"] == "get_my_roster" and r.get("ok") and r.get("data"):
            co = r["data"].get("company", {})
            if co.get("desk_phone"):
                desk_phone = co["desk_phone"]
                break

    # -- Tool errors --
    tool_errors = state.get("tool_errors", [])
    if tool_errors:
        first_err = tool_errors[0]
        if first_err in ("TIMEOUT", "TOOL_TIMEOUT"):
            # Find which step.
            step = "checking your request"
            for r in tool_results:
                if r.get("error_code") in ("TIMEOUT", "TOOL_TIMEOUT"):
                    step = r.get("error_params", {}).get("step", step)
                    break
            reply = TOOL_TIMEOUT.format(step=step, desk_phone=desk_phone)
            status = "error"
            return _append_to_history({**state, "reply": reply, "status": status, "options": [], "request_id": None})

        if first_err == "SEAT_TAKEN":
            reply = CAB_FILLED
            # Try to append nearest slots from check_cab_options result.
            for r in tool_results:
                if r["tool"] == "check_cab_options" and r.get("ok") and r.get("data"):
                    slots = r["data"].get("nearest_slots", [])
                    if slots:
                        options = [f"{s['departure_time']} (cab {s['cab_id']}, {s['free_seats']} seats free)" for s in slots]
            status = "error"
            return _append_to_history({**state, "reply": reply, "status": status, "options": options, "request_id": None})

        for err_code in ("R1_CUTOFF_VIOLATED", "R2_DELTA_EXCEEDED", "R3_OUTSIDE_SERVICE_WINDOW"):
            if first_err == err_code:
                params = {}
                for r in tool_results:
                    if r.get("error_code") == err_code:
                        params = r.get("error_params", {})
                reply = format_rule_message(err_code, params)
                status = "error"
                return _append_to_history({**state, "reply": reply, "status": status, "options": [], "request_id": None})

        # Other tool errors (DB_ERROR, TOOL_ARG_ERROR, UNKNOWN_TOOL, etc.)
        # Return generic failure — no internal details to the user.
        step = "processing your request"
        for r in tool_results:
            if not r.get("ok"):
                ep = r.get("error_params", {})
                if "step" in ep:
                    step = ep["step"]
                break
        reply = TOOL_TIMEOUT.format(step=step, desk_phone=desk_phone)
        status = "error"
        return _append_to_history({**state, "reply": reply, "status": status, "options": [], "request_id": None})

    # -- No tool calls (unknown intent or empty) --
    if not tool_results:
        if plan_dict.get("intent") == "unknown":
            reply = "I'm not sure what you'd like to do. You can ask me to show your roster, change your pickup or drop time, or check the status of a request."
        else:
            reply = "I didn't quite understand that. Could you please rephrase?"
        return _append_to_history({**state, "reply": reply, "status": status, "options": [], "request_id": None})

    # -- ask_clarification --
    for r in tool_results:
        if r["tool"] == "ask_clarification" and r.get("ok") and r.get("data"):
            q_code = r["data"].get("question_code", "")
            opts = r["data"].get("options", [])
            if q_code == "WHICH_TIME":
                reply = f"What time would you like? Options: {', '.join(opts)}."
            elif q_code == "AM_PM":
                reply = f"Did you mean {' or '.join(opts)}?"
            elif q_code == "WHICH_TRIP":
                reply = f"Which trip would you like to change — {' or '.join(opts)}?"
            else:
                reply = f"Could you clarify? Options: {', '.join(opts)}."
            options = opts
            status = "awaiting_input"
            return _append_to_history({**state, "reply": reply, "status": status, "options": options, "request_id": None})

    # -- check_cab_options (rule violations or options) —
    # Evaluated before roster display so a R1/R2 violation wins even when
    # get_my_roster also ran (both are in the same multi-step plan).
    for r in tool_results:
        if r["tool"] == "check_cab_options" and r.get("ok") and r.get("data"):
            data = r["data"]
            rules = data.get("rules", {})

            # R1 violation.
            if not rules.get("r1", {}).get("ok", True):
                params = rules["r1"]["params"]
                reply = format_rule_message("R1_CUTOFF_VIOLATED", params)
                nearest = data.get("nearest_slots", [])
                if nearest:
                    reply += "\n\nNearest available options:\n"
                    reply += "\n".join(
                        f"• {s['departure_time']} (cab {s['cab_id']}, {s['free_seats']} seats free)"
                        for s in nearest
                    )
                    options = [s["departure_time"] for s in nearest]
                status = "error"
                return _append_to_history({**state, "reply": reply, "status": status, "options": options, "request_id": None})

            # R2 violation.
            if not rules.get("r2", {}).get("ok", True):
                params = rules["r2"]["params"]
                reply = format_rule_message("R2_DELTA_EXCEEDED", params)
                status = "error"
                return _append_to_history({**state, "reply": reply, "status": status, "options": [], "request_id": None})

            # R3 violation.
            if not rules.get("r3", {}).get("ok", True):
                params = rules["r3"]["params"]
                reply = format_rule_message("R3_OUTSIDE_SERVICE_WINDOW", params)
                status = "error"
                return _append_to_history({**state, "reply": reply, "status": status, "options": [], "request_id": None})

            # No rule violation — R4 / nearest slots.
            exact = data.get("exact_match")
            nearest = data.get("nearest_slots", [])
            if exact:
                reply = (
                    f"I found a cab at {exact['departure_time']} with {exact['free_seats']} seat(s) free "
                    f"(cab {exact['cab_id']}). Shall I book it?"
                )
                options = [exact["cab_id"]]
                status = "awaiting_input"
            elif nearest:
                reply = "No cab at that exact time, but here are the nearest options:\n"
                reply += "\n".join(
                    f"• {s['departure_time']} (cab {s['cab_id']}, {s['free_seats']} seats free)"
                    for s in nearest
                )
                options = [s["departure_time"] for s in nearest]
                status = "awaiting_input"
            else:
                reply = "No available cabs found near that time."
            return _append_to_history({**state, "reply": reply, "status": status, "options": options, "request_id": None})

    # -- get_my_roster --
    for r in tool_results:
        if r["tool"] == "get_my_roster" and r.get("ok") and r.get("data"):
            entries = r["data"].get("entries", [])
            if not entries:
                reply = "You have no active roster entries today."
            else:
                lines = []
                for e in entries:
                    lines.append(f"• {e['direction'].capitalize()} at {e['scheduled_time']} (cab {e['cab_id']})")
                reply = "Here is your roster for today:\n" + "\n".join(lines)
            return _append_to_history({**state, "reply": reply, "status": "ok", "options": [], "request_id": None})

    # -- raise_shift_change --
    for r in tool_results:
        if r["tool"] == "raise_shift_change" and r.get("ok") and r.get("data"):
            data = r["data"]
            rid = data.get("request_id")
            applied_status = data.get("status", "applied")
            if applied_status == "already_applied":
                reply = f"Your shift change (request #{str(rid)[:8]}) has already been applied."
            else:
                reply = f"Done! Your shift change has been applied (request #{str(rid)[:8]})."
            return _append_to_history({
                **state,
                "reply": reply,
                "status": "ok",
                "options": [],
                "request_id": rid,
            })

    # -- get_request_status --
    for r in tool_results:
        if r["tool"] == "get_request_status" and r.get("ok") and r.get("data"):
            reqs = r["data"].get("requests", [])
            if not reqs:
                reply = "You have no shift change requests."
            else:
                lines = []
                for req in reqs:
                    lines.append(
                        f"• Request #{req['request_id'][:8]}: {req['status']} "
                        f"(new_time={req.get('new_time', 'N/A')})"
                    )
                reply = "Your shift change requests:\n" + "\n".join(lines)
            return _append_to_history({**state, "reply": reply, "status": "ok", "options": [], "request_id": None})

    # Fallback.
    trace_id = state.get("trace_id", "")
    logger.warning("compose_reply fell through to fallback; trace_id=%s", trace_id)
    reply = GENERIC_ERROR.format(
        desk_phone=settings.desk_phone,
        trace_id=trace_id,
    )
    return _append_to_history({**state, "reply": reply, "status": "error", "options": [], "request_id": None})


def _append_to_history(state: ChatState) -> ChatState:
    """
    Append the current turn to history (keep last MAX_HISTORY_TURNS * 2 messages).
    """
    history = list(state.get("history", []))
    user_msg = state.get("current_message", "")
    reply = state.get("reply", "")

    if user_msg:
        history.append({"role": "user", "content": user_msg})
    if reply:
        history.append({"role": "assistant", "content": reply})

    # Keep last MAX_HISTORY_TURNS turns (each turn = 2 messages).
    max_msgs = MAX_HISTORY_TURNS * 2
    history = history[-max_msgs:]

    return {**state, "history": history}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


def _should_continue_after_authorize(state: ChatState) -> str:
    if state.get("reply"):
        return END
    return "plan"


def _should_continue_after_plan(state: ChatState) -> str:
    if state.get("reply") or state.get("plan") is None:
        return "compose_reply"
    return "validate_plan"


def build_chat_graph(checkpointer=None) -> Any:
    """Build and compile the chat StateGraph."""
    builder = StateGraph(ChatState)

    builder.add_node("authorize", node_authorize)
    builder.add_node("plan", node_plan)
    builder.add_node("validate_plan", node_validate_plan)
    builder.add_node("execute_tools", node_execute_tools)
    builder.add_node("compose_reply", node_compose_reply)

    builder.set_entry_point("authorize")
    builder.add_conditional_edges(
        "authorize",
        _should_continue_after_authorize,
        {END: END, "plan": "plan"},
    )
    builder.add_conditional_edges(
        "plan",
        _should_continue_after_plan,
        {"compose_reply": "compose_reply", "validate_plan": "validate_plan"},
    )
    builder.add_edge("validate_plan", "execute_tools")
    builder.add_edge("execute_tools", "compose_reply")
    builder.add_edge("compose_reply", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# PostgresSaver singleton
# ---------------------------------------------------------------------------

_saver_instance = None
_saver_lock = threading.Lock()


def get_checkpointer() -> PostgresSaver:
    global _saver_instance
    with _saver_lock:
        if _saver_instance is None:
            _saver_instance = PostgresSaver.from_conn_string(settings.database_url)
            _saver_instance.setup()
    return _saver_instance


def get_chat_graph():
    """Return the compiled chat graph with Postgres checkpointer."""
    return build_chat_graph(checkpointer=get_checkpointer())
