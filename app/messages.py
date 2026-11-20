"""
app/messages.py — user-facing message templates keyed by rule code.

Templates use the company's real numbers (cutoff_minutes, desk_phone, etc.)
so the LLM never writes rule explanations.

SPEC §12: each failure template states: what happened, whether the roster is
unchanged, and the next step.  No internals, no tracebacks.

All format() calls use keyword arguments that match the message_params
produced by rules.py.
"""

# --- Rule violation messages ---
# Each key matches a RuleResult.code.

RULE_MESSAGES: dict[str, str] = {
    "R1_CUTOFF_VIOLATED": (
        "Sorry, changes must be requested at least {cutoff_minutes} minutes before your scheduled time. "
        "Your cab is at {scheduled_time}, which is only {minutes_until} minutes away. "
        "Your roster has not been changed. "
        # TODO stage3: escalation offer
        "Please contact the transport desk or try a different time."
    ),
    "R2_DELTA_EXCEEDED": (
        "The requested time ({new_time}) is {actual_delta_minutes} minutes away from your scheduled time "
        "({scheduled_time}). Your company policy allows a maximum shift of {max_shift_delta_minutes} minutes. "
        "Your roster has not been changed. Please choose a time within the allowed range."
    ),
    "R3_OUTSIDE_SERVICE_WINDOW": (
        "The time {new_time} is outside the service window ({service_start}–{service_end}). "
        "Your roster has not been changed. "
        "Please choose a time within the service window."
    ),
    "R4_NO_SEAT": (
        "That cab is full. Your roster has not been changed. Here are the nearest available options:"
    ),
}

# --- Failure messages (SPEC §12) ---

# LLM timeout / quota / malformed JSON twice
GENERIC_FAILURE = (
    "I couldn't understand that just now. Your roster has not been changed. "
    "Please try again, or contact the transport desk on {desk_phone}."
)

# Tool step timeout — {step} names the failing operation
TOOL_TIMEOUT = (
    "Something went wrong while {step}. Your roster has not been changed. "
    "Please try again, or contact the transport desk on {desk_phone}."
)

WRITE_TOOL_UNCERTAIN = (
    "I couldn't confirm your change in time. Your roster may or may not have been updated. "
    'Ask me "my requests" to check before trying again.'
)

# Seat filled between check and apply
CAB_FILLED = (
    "That cab just filled up while you were deciding. "
    "Your roster has not been changed. Nearest available options:"
)

# SPEC §12 wording. Pending must never claim the change is applied.
PENDING_APPROVAL = (
    "Your request #{short_id} is waiting for {manager}. "
    "Nothing has changed yet: your roster stays as it is until the request is approved. "
    'Typical wait is a few hours; you will see the result here, or ask me "my requests".'
)

REQUEST_STATUS_MESSAGES = {
    "pending_approval": "waiting for your manager; your roster is unchanged",
    "applied": "approved and applied to your roster",
    "rejected": "declined; your roster is unchanged",
    "expired": "expired; your roster is unchanged",
    "approved_not_applied": "approved, but the cab filled up; your roster is unchanged",
    "failed": "could not be applied; your roster is unchanged",
}

NON_APPLIED_MESSAGES = {
    "rejected": "Your manager declined this change. Your roster is unchanged. You can submit a new request or contact the transport desk.",
    "expired": "This approval request expired. Your roster is unchanged. Please submit a new request.",
    "approved_not_applied": "Your manager approved the change, but the cab filled up. Your roster is unchanged. Please choose another cab.",
    "failed": "We could not apply this change. Your roster is unchanged. Please contact the transport desk.",
}

# Escalation (R1/R2 only, SPEC §7/§9)
ESCALATION_OFFER = (
    "If you still need this change, I can send it to your transport manager for approval "
    "(no seat is held while it waits). Reply \"yes, send it to my manager\" to do that, or \"no\" to leave it."
)
ESCALATION_PICK_TIME = (
    "Pick one of these times and I can send that change to your transport manager for approval."
)
ESCALATION_NOTHING_PENDING = (
    "There is no blocked change waiting to be sent to your manager. Your roster has not been changed. "
    "Tell me which trip and what time you would like."
)
ESCALATION_DECLINED = "Okay, I won't send it. Your roster has not been changed."

# Unexpected server error — includes trace_id so ops can correlate
GENERIC_ERROR = (
    "Something unexpected happened. Your roster has not been changed. "
    "If the problem persists, contact the transport desk on {desk_phone} "
    "and quote reference {trace_id}."
)

SESSION_INVALID = "Your session is not valid. Please sign in again."
SESSION_EXPIRED = "Your session has expired. Please sign in again."


def format_rule_message(code: str, params: dict) -> str:
    """Return the human-readable explanation for a rule violation."""
    template = RULE_MESSAGES.get(code, "A policy rule was violated (code: {code}).")
    return template.format(**params, code=code)

