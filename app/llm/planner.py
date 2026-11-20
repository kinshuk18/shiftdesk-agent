"""
app/llm/planner.py — Plan schema (pydantic) and planner logic.

SPEC §4: LLM output is untrusted input — validated by pydantic.
         Tool names checked against allow-list; company/employee never accepted.
SPEC §3: One repair retry on malformed JSON, counted in budget.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

from app.llm.base import BudgetExhausted, LLMBudget, LLMClient, LLMError
from app.llm.prompts import PLANNER_SYSTEM_PROMPT, build_user_block

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allow-listed tool names (SPEC §8).
# ---------------------------------------------------------------------------
ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "get_my_roster",
        "check_cab_options",
        "raise_shift_change",
        "get_request_status",
        "ask_clarification",
    }
)

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = {}

    @field_validator("tool")
    @classmethod
    def tool_must_be_allowed(cls, v: str) -> str:
        if v not in ALLOWED_TOOLS:
            raise ValueError(
                f"Tool '{v}' is not in the allow-list. "
                f"Allowed: {sorted(ALLOWED_TOOLS)}"
            )
        return v

    @field_validator("args")
    @classmethod
    def args_must_not_contain_identity(cls, v: dict) -> dict:
        """
        SPEC §4: company_id and employee_id must NEVER come from the model.
        Reject any plan that tries to pass them through tool args.
        """
        forbidden = {"company_id", "employee_id"}
        found = forbidden & set(v.keys())
        if found:
            raise ValueError(
                f"Tool args must not contain identity fields: {sorted(found)}"
            )
        return v


class Plan(BaseModel):
    intent: Literal[
        "change_shift", "get_roster", "get_status", "ask_clarification",
        "confirm", "decline", "unknown"
    ]
    direction: Optional[Literal["pickup", "drop"]] = None
    target_time: Optional[str] = None  # "HH:MM" or null
    relative_hint: Optional[Literal["earlier", "later"]] = None
    chosen_option: Optional[str] = None
    ambiguity: Literal["none", "time", "direction", "am_pm", "which_entry"] = "none"
    tool_calls: list[ToolCall] = []

    @field_validator("target_time")
    @classmethod
    def validate_time_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError(f"target_time must be HH:MM, got '{v}'")
        hh, mm = parts
        if not (hh.isdigit() and mm.isdigit()):
            raise ValueError(f"target_time must be HH:MM digits, got '{v}'")
        if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
            raise ValueError(f"target_time out of range: '{v}'")
        return v


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class PlannerError(Exception):
    """LLM produced invalid output after repair attempt."""


def _parse_plan(raw: str) -> Plan:
    """
    Parse raw LLM output into a Plan.
    Raises ValueError / pydantic.ValidationError on failure.
    """
    data = json.loads(raw)
    return Plan.model_validate(data)


def call_planner(
    llm: LLMClient,
    budget: LLMBudget,
    user_message: str,
    history: list[dict] | None = None,
    roster: list[dict] | None = None,
) -> Plan:
    """
    Call the LLM planner, validate the result, and repair once on failure.

    SPEC §3: "One repair retry on malformed JSON, counted in the budget."
    Total calls for planner ≤ 2 (1 normal + 1 repair).

    Args:
        llm           — LLMClient implementation
        budget        — per-message call counter (shared with graph)
        user_message  — raw employee text (will be wrapped in <msg> block)
        history       — optional prior turns for context (list of role/content dicts)

    Returns:
        Plan validated by pydantic.

    Raises:
        BudgetExhausted — if budget is full before the call
        PlannerError    — LLM produced invalid output after repair attempt
        LLMTimeoutError — provider timed out
        LLMQuotaError   — provider rate-limited
    """
    messages = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": build_user_block(user_message, roster)})

    # First attempt.
    budget.charge()
    raw = llm.chat(messages)

    try:
        return _parse_plan(raw)
    except (json.JSONDecodeError, ValueError, Exception) as first_exc:
        logger.warning("Planner first attempt failed: %s — retrying once", first_exc)

        # Repair attempt: add the bad output + a correction instruction.
        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON matching the required schema. "
                    "Return ONLY the JSON object, no markdown, no explanation."
                ),
            },
        ]
        budget.charge()  # Repair counts in budget (SPEC §3).
        raw2 = llm.chat(repair_messages)
        try:
            return _parse_plan(raw2)
        except (json.JSONDecodeError, ValueError, Exception) as second_exc:
            raise PlannerError(
                f"Planner failed after repair. First error: {first_exc}. "
                f"Second error: {second_exc}."
            ) from second_exc
