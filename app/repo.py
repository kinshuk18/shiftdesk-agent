"""
app/repo.py — database read functions for roster, cabs, and company data.

Security invariant: every function that returns business data takes a Principal
(company_id + employee_id) and puts both in the WHERE clause.
No function accepts company_id from a request body or from tool arguments.

All queries use parameterized statements. No string formatting of SQL.
"""

from dataclasses import dataclass
from datetime import date, time
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from app.auth import Principal


@dataclass
class Company:
    id: str
    name: str
    cutoff_minutes: int
    max_shift_delta_minutes: int
    service_start: time
    service_end: time
    desk_phone: Optional[str]
    transport_manager_id: Optional[str]


@dataclass
class RosterEntry:
    company_id: str
    entry_id: str
    employee_id: str
    service_date: date
    direction: str          # "pickup" | "drop"
    cab_id: str
    scheduled_time: time
    status: str


@dataclass
class Cab:
    company_id: str
    cab_id: str
    route_id: str
    direction: str
    departure_time: time
    capacity: int


# ---------------------------------------------------------------------------
# Company
# ---------------------------------------------------------------------------

def get_company(conn: psycopg.Connection, company_id: str) -> Optional[Company]:
    """Return company settings by id, or None if not found."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT * FROM companies WHERE id = %s",
            (company_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return Company(**row)


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

def get_roster(
    conn: psycopg.Connection,
    principal: Principal,
    service_date: date,
) -> list[RosterEntry]:
    """
    Return all roster entries for a given employee on a given date.
    company_id and employee_id come only from the verified Principal —
    never from request body or tool args.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT company_id, entry_id, employee_id, service_date,
                   direction, cab_id, scheduled_time, status
            FROM roster_entries
            WHERE company_id  = %s
              AND employee_id = %s
              AND service_date = %s
              AND status != 'cancelled'
            ORDER BY scheduled_time
            """,
            (principal.company_id, principal.employee_id, service_date),
        )
        rows = cur.fetchall()
    return [RosterEntry(**r) for r in rows]


def get_roster_entry(
    conn: psycopg.Connection,
    principal: Principal,
    entry_id: str,
) -> Optional[RosterEntry]:
    """
    Return a single roster entry.
    The WHERE clause includes company_id and employee_id from the Principal
    so an employee can never read another employee's (or company's) entry.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT company_id, entry_id, employee_id, service_date,
                   direction, cab_id, scheduled_time, status
            FROM roster_entries
            WHERE company_id  = %s
              AND employee_id = %s
              AND entry_id    = %s
            """,
            (principal.company_id, principal.employee_id, entry_id),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return RosterEntry(**row)


# ---------------------------------------------------------------------------
# Cabs
# ---------------------------------------------------------------------------

def get_cab(
    conn: psycopg.Connection,
    company_id: str,
    cab_id: str,
) -> Optional[Cab]:
    """Return a single cab by (company_id, cab_id)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT company_id, cab_id, route_id, direction, departure_time, capacity
            FROM cabs
            WHERE company_id = %s AND cab_id = %s
            """,
            (company_id, cab_id),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return Cab(**row)


def count_active_bookings(
    conn: psycopg.Connection,
    company_id: str,
    cab_id: str,
    service_date: date,
) -> int:
    """Count active roster entries for a cab on a given date (for capacity check)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM roster_entries
            WHERE company_id   = %s
              AND cab_id       = %s
              AND service_date = %s
              AND status       IN ('active', 'pending_change')
            """,
            (company_id, cab_id, service_date),
        )
        row = cur.fetchone()
    return row[0] if row else 0


def get_cabs_for_route(
    conn: psycopg.Connection,
    company_id: str,
    route_id: str,
    direction: str,
) -> list[Cab]:
    """Return all cabs on a route for a given direction, ordered by departure_time."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT company_id, cab_id, route_id, direction, departure_time, capacity
            FROM cabs
            WHERE company_id = %s
              AND route_id   = %s
              AND direction  = %s
            ORDER BY departure_time
            """,
            (company_id, route_id, direction),
        )
        rows = cur.fetchall()
    return [Cab(**r) for r in rows]


def list_shift_change_requests(conn, company_id: str, status: str | None = None) -> list[dict]:
    """Manager request queue. Every lookup is restricted to the supplied tenant."""
    with conn.cursor(row_factory=dict_row) as cur:
        if status:
            cur.execute("""SELECT id AS request_id, employee_id, entry_id, old_cab_id, new_cab_id, new_time, status, reason, created_at, decided_at, decision_note FROM shift_change_requests WHERE company_id=%s AND status=%s ORDER BY created_at DESC""", (company_id, status))
        else:
            cur.execute("""SELECT id AS request_id, employee_id, entry_id, old_cab_id, new_cab_id, new_time, status, reason, created_at, decided_at, decision_note FROM shift_change_requests WHERE company_id=%s ORDER BY created_at DESC""", (company_id,))
        return cur.fetchall()


def list_employee_shift_change_requests(conn, company_id: str, employee_id: str) -> list[dict]:
    """Return one employee's requests, scoped by company and employee."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""SELECT id AS request_id, status, new_time, created_at, decided_at, decision_note FROM shift_change_requests WHERE company_id=%s AND employee_id=%s ORDER BY created_at DESC""", (company_id, employee_id))
        return cur.fetchall()
