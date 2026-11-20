"""
app/messages.py — user-facing message templates keyed by rule code.

Templates use the company's real numbers (cutoff_minutes, desk_phone, etc.)
so the LLM never writes rule explanations.

All format() calls use keyword arguments that match the message_params
produced by rules.py.
"""

from string import Template

# --- Rule violation messages ---
# Each key matches a RuleResult.code.

RULE_MESSAGES: dict[str, str] = {
    "R1_CUTOFF_VIOLATED": (
        "Sorry, changes must be requested at least {cutoff_minutes} minutes before your scheduled time. "
        "Your cab is at {scheduled_time}, which is only {minutes_until} minutes away. "
        "Would you like to escalate this to your transport manager?"
    ),
    "R2_DELTA_EXCEEDED": (
        "The requested time ({new_time}) is {actual_delta_minutes} minutes away from your scheduled time "
        "({scheduled_time}). Your company policy allows a maximum shift of {max_shift_delta_minutes} minutes."
    ),
    "R3_OUTSIDE_SERVICE_WINDOW": (
        "The time {new_time} is outside the service window ({service_start}–{service_end}). "
        "Please choose a time within the service window."
    ),
    "R4_NO_SEAT": (
        "That cab is full. Here are the nearest available options:"
    ),
}

# --- Other system messages ---

GENERIC_FAILURE = (
    "I couldn't understand that just now. Your roster has not been changed. "
    "Please try again, or contact the transport desk on {desk_phone}."
)

TOOL_TIMEOUT = (
    "Something went wrong while {step}. Your roster has not been changed. "
    "Please try again, or contact the transport desk on {desk_phone}."
)

CAB_FILLED = (
    "That cab just filled up while you were deciding. "
    "Your roster has not been changed. Nearest available options:"
)

PENDING_APPROVAL = (
    "Your request #{short_id} is waiting for your transport manager. "
    "Nothing has changed yet. Typical wait is a few hours; you will see the result here."
)

SESSION_INVALID = "Your session is not valid. Please sign in again."
SESSION_EXPIRED = "Your session has expired. Please sign in again."


def format_rule_message(code: str, params: dict) -> str:
    """Return the human-readable explanation for a rule violation."""
    template = RULE_MESSAGES.get(code, "A policy rule was violated (code: {code}).")
    return template.format(**params, code=code)
