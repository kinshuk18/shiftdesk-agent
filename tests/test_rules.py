"""
tests/test_rules.py — unit tests for app/rules.py

Pure function tests: no DB, no network.
Covers all four rules with boundary cases: exactly at cut-off, 1 min before, 1 min after.
Two-company cases verify that each company's own numbers are used correctly.
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from app.rules import (
    check_cutoff,
    check_max_delta,
    check_service_window,
    check_capacity,
    RuleResult,
)

IST = ZoneInfo("Asia/Kolkata")


def _now(hour: int, minute: int) -> datetime:
    """Create a datetime in IST on 2026-11-20 at the given HH:MM."""
    return datetime(2026, 11, 20, hour, minute, 0, tzinfo=IST)


# ---------------------------------------------------------------------------
# R1: cut-off rule
# ---------------------------------------------------------------------------

class TestCutoffRule:
    """
    acme cutoff = 120 min
    zenith cutoff = 240 min
    Scheduled time is always 20:30 for these tests.
    """

    SCHEDULED = time(20, 30)
    ACME_CUTOFF = 120
    ZENITH_CUTOFF = 240

    def test_acme_exactly_at_cutoff(self):
        # 18:30 → (20:30 - 18:30) = 120 min == cutoff → allowed
        result = check_cutoff(_now(18, 30), self.SCHEDULED, self.ACME_CUTOFF)
        assert result.ok, f"Expected OK at exact cutoff, got {result}"

    def test_acme_one_minute_before_cutoff(self):
        # 18:31 → 119 min remaining < 120 → rejected
        result = check_cutoff(_now(18, 31), self.SCHEDULED, self.ACME_CUTOFF)
        assert not result.ok
        assert result.code == "R1_CUTOFF_VIOLATED"
        assert result.message_params["cutoff_minutes"] == 120

    def test_acme_one_minute_after_cutoff_window(self):
        # 18:29 → 121 min remaining > 120 → allowed
        result = check_cutoff(_now(18, 29), self.SCHEDULED, self.ACME_CUTOFF)
        assert result.ok

    def test_acme_frozen_clock_scenario(self):
        # Frozen clock 18:45 → (20:30 - 18:45) = 105 min < 120 → rejected (seed invariant)
        result = check_cutoff(_now(18, 45), self.SCHEDULED, self.ACME_CUTOFF)
        assert not result.ok
        assert result.code == "R1_CUTOFF_VIOLATED"
        assert result.message_params["minutes_until"] == 105

    def test_zenith_exactly_at_cutoff(self):
        # 16:30 → 240 min exactly == 240 → allowed
        result = check_cutoff(_now(16, 30), self.SCHEDULED, self.ZENITH_CUTOFF)
        assert result.ok

    def test_zenith_one_minute_before_cutoff(self):
        # 16:31 → 239 min < 240 → rejected
        result = check_cutoff(_now(16, 31), self.SCHEDULED, self.ZENITH_CUTOFF)
        assert not result.ok
        assert result.message_params["cutoff_minutes"] == 240

    def test_zenith_one_minute_after_cutoff_window(self):
        # 16:29 → 241 min > 240 → allowed
        result = check_cutoff(_now(16, 29), self.SCHEDULED, self.ZENITH_CUTOFF)
        assert result.ok

    def test_message_params_contain_scheduled_time(self):
        result = check_cutoff(_now(19, 0), time(20, 0), 120)
        assert not result.ok
        assert "scheduled_time" in result.message_params

    def test_well_before_cutoff(self):
        # 10:00 → 10.5h before 20:30 → definitely allowed
        result = check_cutoff(_now(10, 0), self.SCHEDULED, self.ACME_CUTOFF)
        assert result.ok


# ---------------------------------------------------------------------------
# R2: max shift delta
# ---------------------------------------------------------------------------

class TestMaxDeltaRule:
    """
    acme max_shift_delta = 180 min
    zenith max_shift_delta = 120 min
    """

    SCHEDULED = time(20, 0)
    ACME_DELTA = 180
    ZENITH_DELTA = 120

    def test_acme_exactly_at_max_delta(self):
        # 23:00 → delta = 180 min == 180 → allowed
        result = check_max_delta(self.SCHEDULED, time(23, 0), self.ACME_DELTA)
        assert result.ok

    def test_acme_one_minute_over_delta(self):
        # 23:01 → delta = 181 min > 180 → rejected
        result = check_max_delta(self.SCHEDULED, time(23, 1), self.ACME_DELTA)
        assert not result.ok
        assert result.code == "R2_DELTA_EXCEEDED"
        assert result.message_params["max_shift_delta_minutes"] == 180

    def test_acme_one_minute_under_delta(self):
        # 22:59 → delta = 179 min < 180 → allowed
        result = check_max_delta(self.SCHEDULED, time(22, 59), self.ACME_DELTA)
        assert result.ok

    def test_zenith_exactly_at_max_delta(self):
        # 22:00 → delta = 120 min == 120 → allowed
        result = check_max_delta(self.SCHEDULED, time(22, 0), self.ZENITH_DELTA)
        assert result.ok

    def test_zenith_one_minute_over_delta(self):
        # 22:01 → delta = 121 min > 120 → rejected
        result = check_max_delta(self.SCHEDULED, time(22, 1), self.ZENITH_DELTA)
        assert not result.ok
        assert result.message_params["max_shift_delta_minutes"] == 120

    def test_zenith_one_minute_under_delta(self):
        # 21:59 → delta = 119 min < 120 → allowed
        result = check_max_delta(self.SCHEDULED, time(21, 59), self.ZENITH_DELTA)
        assert result.ok

    def test_earlier_time_allowed(self):
        # Moving earlier: 20:00 → 17:00, delta = 180 → at acme limit
        result = check_max_delta(self.SCHEDULED, time(17, 0), self.ACME_DELTA)
        assert result.ok

    def test_same_time_always_ok(self):
        result = check_max_delta(self.SCHEDULED, self.SCHEDULED, self.ACME_DELTA)
        assert result.ok


# ---------------------------------------------------------------------------
# R3: service window
# ---------------------------------------------------------------------------

class TestServiceWindowRule:
    """acme service: 07:00–23:00  zenith: 06:00–22:00"""

    ACME_START = time(7, 0)
    ACME_END   = time(23, 0)
    ZENITH_START = time(6, 0)
    ZENITH_END   = time(22, 0)

    def test_exactly_at_start(self):
        result = check_service_window(time(7, 0), self.ACME_START, self.ACME_END)
        assert result.ok

    def test_one_minute_before_start(self):
        result = check_service_window(time(6, 59), self.ACME_START, self.ACME_END)
        assert not result.ok
        assert result.code == "R3_OUTSIDE_SERVICE_WINDOW"

    def test_one_minute_after_start(self):
        result = check_service_window(time(7, 1), self.ACME_START, self.ACME_END)
        assert result.ok

    def test_exactly_at_end(self):
        result = check_service_window(time(23, 0), self.ACME_START, self.ACME_END)
        assert result.ok

    def test_one_minute_after_end(self):
        result = check_service_window(time(23, 1), self.ACME_START, self.ACME_END)
        assert not result.ok

    def test_zenith_exactly_at_start(self):
        result = check_service_window(time(6, 0), self.ZENITH_START, self.ZENITH_END)
        assert result.ok

    def test_zenith_one_minute_before_start(self):
        result = check_service_window(time(5, 59), self.ZENITH_START, self.ZENITH_END)
        assert not result.ok
        assert result.message_params["service_start"] == "06:00"

    def test_zenith_exactly_at_end(self):
        result = check_service_window(time(22, 0), self.ZENITH_START, self.ZENITH_END)
        assert result.ok

    def test_zenith_one_minute_after_end(self):
        result = check_service_window(time(22, 1), self.ZENITH_START, self.ZENITH_END)
        assert not result.ok


# ---------------------------------------------------------------------------
# R4: capacity
# ---------------------------------------------------------------------------

class TestCapacityRule:
    def test_seats_available(self):
        result = check_capacity(2)
        assert result.ok

    def test_exactly_one_seat(self):
        result = check_capacity(1)
        assert result.ok

    def test_no_seats(self):
        result = check_capacity(0)
        assert not result.ok
        assert result.code == "R4_NO_SEAT"

    def test_negative_seats_treated_as_full(self):
        # Should never happen, but be safe.
        result = check_capacity(-1)
        assert not result.ok
