"""
app/rules.py — pure business rules (R1-R4).

No I/O. No imports from app.db or app.repo.
All functions take plain values and return a RuleResult.
Tests can call these without a database connection.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta


@dataclass(frozen=True)
class RuleResult:
    ok: bool
    code: str           # e.g. "R1_CUTOFF_VIOLATED"
    message_params: dict = field(default_factory=dict)


# Sentinel "ok" result with a neutral code.
OK = RuleResult(ok=True, code="OK", message_params={})


def _combine_date_time(ref_dt: datetime, t: time) -> datetime:
    """
    Return a timezone-aware datetime by combining the date from ref_dt with
    the given time, using ref_dt's tzinfo.
    """
    return datetime(
        ref_dt.year, ref_dt.month, ref_dt.day,
        t.hour, t.minute, t.second,
        tzinfo=ref_dt.tzinfo,
    )


def check_cutoff(
    now: datetime,
    scheduled_time: time,       # the ORIGINAL scheduled_time from roster
    cutoff_minutes: int,
) -> RuleResult:
    """
    R1: Change is allowed only if (original_scheduled - now) >= cutoff_minutes.
    Measured against the ORIGINAL scheduled time, never the proposed new time.
    """
    original_dt = _combine_date_time(now, scheduled_time)
    minutes_until = (original_dt - now).total_seconds() / 60

    if minutes_until >= cutoff_minutes:
        return OK

    return RuleResult(
        ok=False,
        code="R1_CUTOFF_VIOLATED",
        message_params={
            "cutoff_minutes": cutoff_minutes,
            "minutes_until": round(minutes_until),
            "scheduled_time": scheduled_time.strftime("%H:%M"),
        },
    )


def check_max_delta(
    scheduled_time: time,
    new_time: time,
    max_shift_delta_minutes: int,
) -> RuleResult:
    """
    R2: |new_time - scheduled_time| <= max_shift_delta_minutes.
    Times are compared as minutes-from-midnight; we take absolute difference.
    """
    old_minutes = scheduled_time.hour * 60 + scheduled_time.minute
    new_minutes = new_time.hour * 60 + new_time.minute
    delta = abs(new_minutes - old_minutes)

    if delta <= max_shift_delta_minutes:
        return OK

    return RuleResult(
        ok=False,
        code="R2_DELTA_EXCEEDED",
        message_params={
            "max_shift_delta_minutes": max_shift_delta_minutes,
            "actual_delta_minutes": delta,
            "scheduled_time": scheduled_time.strftime("%H:%M"),
            "new_time": new_time.strftime("%H:%M"),
        },
    )


def check_service_window(
    new_time: time,
    service_start: time,
    service_end: time,
) -> RuleResult:
    """
    R3: new_time must be within [service_start, service_end] (inclusive).
    """
    if service_start <= new_time <= service_end:
        return OK

    return RuleResult(
        ok=False,
        code="R3_OUTSIDE_SERVICE_WINDOW",
        message_params={
            "new_time": new_time.strftime("%H:%M"),
            "service_start": service_start.strftime("%H:%M"),
            "service_end": service_end.strftime("%H:%M"),
        },
    )


def check_capacity(available_seats: int) -> RuleResult:
    """
    R4: target cab must have at least one free seat.
    available_seats = capacity - count_of_active_bookings (computed by caller).
    Note: escalation is NOT offered for R4 — the SPEC says R4 never escalates.
    """
    if available_seats > 0:
        return OK

    return RuleResult(
        ok=False,
        code="R4_NO_SEAT",
        message_params={"available_seats": 0},
    )
