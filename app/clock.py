"""
app/clock.py — single source of truth for "now".

All time logic in the codebase must call clock.now(), never datetime.now() directly.
This lets tests and the frozen-clock env override time deterministically.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.config import settings

IST = ZoneInfo("Asia/Kolkata")


class Clock:
    def now(self) -> datetime:
        """
        Return current time as timezone-aware datetime.
        If APP_NOW is set, parse and return that fixed instant.
        Otherwise return real wall-clock time (in IST).
        """
        raw = settings.app_now.strip()
        if raw:
            return datetime.fromisoformat(raw)
        return datetime.now(tz=IST)


# Module-level singleton.
clock = Clock()
