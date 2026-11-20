"""
Seed script for ShiftDesk.

Idempotent: uses INSERT ... ON CONFLICT DO NOTHING throughout.
All times are relative to the frozen clock date: 2026-11-20.
Run with: python db/seed.py
Requires DATABASE_URL in env or .env file.
"""

import os
import sys
from datetime import date, time
from pathlib import Path

# Allow running from repo root or from db/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg
from dotenv import load_dotenv

load_dotenv()

DSN = os.environ.get("DATABASE_URL", "postgresql://shiftdesk:shiftdesk@db:5432/shiftdesk")

# The frozen clock date — seed dates are relative to this.
SEED_DATE = date(2026, 11, 20)


def run(conn: psycopg.Connection) -> None:
    seed_companies(conn)
    seed_employees(conn)
    seed_routes(conn)
    seed_cabs(conn)
    seed_roster(conn)
    conn.commit()
    print("Seed complete.")


def seed_companies(conn: psycopg.Connection) -> None:
    rows = [
        # id, name, cutoff_minutes, max_shift_delta_minutes, service_start, service_end, desk_phone, transport_manager_id
        ("acme",   "Acme Corp",    120, 180, time(7, 0),  time(23, 0), "+91-80-4100-0001", "MGR001"),
        ("zenith", "Zenith Tech",  240, 120, time(6, 0),  time(22, 0), "+91-80-4200-0001", "MGR002"),
        ("nimbus", "Nimbus Labs",   60, 120, time(7, 30), time(21, 30), "+91-80-4300-0001", "MGR003"),
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO companies (id, name, cutoff_minutes, max_shift_delta_minutes,
                                   service_start, service_end, desk_phone, transport_manager_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            rows,
        )


def seed_employees(conn: psycopg.Connection) -> None:
    # E101 exists in BOTH acme and zenith (different people) — isolation invariant.
    rows = [
        # (company_id, employee_id, name)
        # --- Acme ---
        ("acme", "E101", "Arjun Nair"),          # has drop at 20:30 → 105 min away (breaks 120-min cutoff)
        ("acme", "E102", "Priya Sharma"),         # has both pickup + drop today
        ("acme", "E103", "Ravi Kumar"),
        ("acme", "E104", "Sneha Reddy"),
        ("acme", "E105", "Karthik Iyer"),
        ("acme", "E106", "Meera Pillai"),
        ("acme", "MGR001", "Deepak Menon"),       # transport manager

        # --- Zenith ---
        ("zenith", "E101", "Ananya Krishnan"),    # same code, different person — isolation check
        ("zenith", "E201", "Rahul Joshi"),
        ("zenith", "E202", "Divya Patel"),
        ("zenith", "E203", "Suresh Babu"),
        ("zenith", "MGR002", "Lakshmi Subramaniam"),

        # --- Nimbus ---
        ("nimbus", "E301", "Vikram Rao"),
        ("nimbus", "E302", "Pooja Nambiar"),
        ("nimbus", "MGR003", "Arun Dasgupta"),
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO employees (company_id, employee_id, name)
            VALUES (%s, %s, %s)
            ON CONFLICT (company_id, employee_id) DO NOTHING
            """,
            rows,
        )


def seed_routes(conn: psycopg.Connection) -> None:
    rows = [
        # (company_id, route_id, name) — Bengaluru routes
        ("acme", "R1", "Electronic City - Koramangala"),
        ("acme", "R2", "Whitefield - MG Road"),
        ("acme", "R3", "HSR Layout - Silk Board"),

        ("zenith", "R1", "Whitefield - Indiranagar"),
        ("zenith", "R2", "Electronic City - BTM Layout"),

        ("nimbus", "R1", "HSR Layout - Marathahalli"),
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO routes (company_id, route_id, name)
            VALUES (%s, %s, %s)
            ON CONFLICT (company_id, route_id) DO NOTHING
            """,
            rows,
        )


def seed_cabs(conn: psycopg.Connection) -> None:
    # capacity=4 for acme (SPEC), capacity=6 for zenith, capacity=4 for nimbus.
    # Drop cabs: acme R1 around 20:30-22:00, R2 around 20:00-21:30
    # Pickup cabs: morning runs
    rows = [
        # (company_id, cab_id, route_id, direction, departure_time, capacity)

        # === ACME pickup cabs ===
        ("acme", "C_AC_P01", "R1", "pickup", time(8,  0), 4),
        ("acme", "C_AC_P02", "R1", "pickup", time(9,  0), 4),
        ("acme", "C_AC_P03", "R2", "pickup", time(8, 30), 4),
        ("acme", "C_AC_P04", "R3", "pickup", time(8, 15), 4),

        # === ACME drop cabs ===
        # C_AC_D01 at 20:30 — E101 (Arjun) is on this; 105 min from 18:45 → breaks cutoff (120 min)
        ("acme", "C_AC_D01", "R1", "drop", time(20, 30), 4),
        # C_AC_D02 at 21:00 — exactly 1 free seat (for race condition test):
        #   capacity=4, we'll book 3 employees on it in roster
        ("acme", "C_AC_D02", "R1", "drop", time(21,  0), 4),
        # C_AC_D03 at 21:30 — same route R1, has free seats (for nearest-slot test)
        ("acme", "C_AC_D03", "R1", "drop", time(21, 30), 4),
        # C_AC_D04 at 20:00 — FULL cab on R2 (capacity=4, all 4 booked)
        ("acme", "C_AC_D04", "R2", "drop", time(20,  0), 4),
        # C_AC_D05 at 20:30 — same route R2 as D04, has free seats → nearest slot
        ("acme", "C_AC_D05", "R2", "drop", time(20, 30), 4),

        # === ZENITH pickup cabs (capacity=6) ===
        ("zenith", "C_ZN_P01", "R1", "pickup", time(8,  0), 6),
        ("zenith", "C_ZN_P02", "R2", "pickup", time(8, 30), 6),

        # === ZENITH drop cabs ===
        ("zenith", "C_ZN_D01", "R1", "drop", time(19,  0), 6),
        ("zenith", "C_ZN_D02", "R1", "drop", time(20,  0), 6),
        ("zenith", "C_ZN_D03", "R2", "drop", time(19, 30), 6),

        # === NIMBUS pickup/drop ===
        ("nimbus", "C_NB_P01", "R1", "pickup", time(8, 0), 4),
        ("nimbus", "C_NB_D01", "R1", "drop",   time(19, 0), 4),
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO cabs (company_id, cab_id, route_id, direction, departure_time, capacity)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (company_id, cab_id) DO NOTHING
            """,
            rows,
        )


def seed_roster(conn: psycopg.Connection) -> None:
    d = SEED_DATE  # 2026-11-20

    # Roster entry helper: (company_id, entry_id, employee_id, service_date, direction, cab_id, scheduled_time, status)
    rows = [
        # === ACME roster ===

        # E101 (Arjun Nair) — drop at 20:30 on R1.
        # At frozen clock 18:45, (20:30 - 18:45) = 105 min < acme cutoff 120 min → R1 violation.
        ("acme", "ENT_AC_001", "E101", d, "drop",   "C_AC_D01", time(20, 30), "active"),

        # E102 (Priya Sharma) — BOTH pickup and drop today (for clarification test).
        ("acme", "ENT_AC_002", "E102", d, "pickup", "C_AC_P01", time( 8,  0), "active"),
        ("acme", "ENT_AC_003", "E102", d, "drop",   "C_AC_D02", time(21,  0), "active"),

        # E103 (Ravi Kumar) — drop at 21:00 (C_AC_D02, the 1-free-seat cab).
        ("acme", "ENT_AC_004", "E103", d, "drop",   "C_AC_D02", time(21,  0), "active"),

        # E104 (Sneha Reddy) — drop at 21:00 (fills C_AC_D02 to 3 bookings; 1 seat left for race test).
        ("acme", "ENT_AC_005", "E104", d, "drop",   "C_AC_D02", time(21,  0), "active"),

        # E105 (Karthik Iyer) — drop on R2 fully booked cab C_AC_D04 (20:00).
        ("acme", "ENT_AC_006", "E105", d, "drop",   "C_AC_D04", time(20,  0), "active"),

        # E106 (Meera Pillai) — drop on R2 fully booked cab C_AC_D04 (20:00).
        ("acme", "ENT_AC_007", "E106", d, "drop",   "C_AC_D04", time(20,  0), "active"),

        # MGR001 and one more to fill C_AC_D04 (capacity=4, need 4 bookings).
        # We use MGR001 and E103 as extra passengers on the full cab.
        ("acme", "ENT_AC_008", "MGR001", d, "drop", "C_AC_D04", time(20,  0), "active"),
        ("acme", "ENT_AC_009", "E103",   d, "drop", "C_AC_D04", time(20,  0), "active"),
        # C_AC_D04 now has 4 bookings → FULL (for nearest-slot test).

        # E101 also has a morning pickup (to test multi-day roster reads).
        ("acme", "ENT_AC_010", "E101", d, "pickup", "C_AC_P01", time( 8,  0), "active"),

        # === ZENITH roster ===

        # Zenith E101 (Ananya Krishnan) — different person from Acme E101.
        ("zenith", "ENT_ZN_001", "E101", d, "pickup", "C_ZN_P01", time( 8,  0), "active"),
        ("zenith", "ENT_ZN_002", "E101", d, "drop",   "C_ZN_D01", time(19,  0), "active"),

        ("zenith", "ENT_ZN_003", "E201", d, "drop",   "C_ZN_D02", time(20,  0), "active"),
        ("zenith", "ENT_ZN_004", "E202", d, "drop",   "C_ZN_D03", time(19, 30), "active"),
        ("zenith", "ENT_ZN_005", "E203", d, "pickup", "C_ZN_P02", time( 8, 30), "active"),

        # === NIMBUS roster ===
        ("nimbus", "ENT_NB_001", "E301", d, "pickup", "C_NB_P01", time( 8,  0), "active"),
        ("nimbus", "ENT_NB_002", "E302", d, "drop",   "C_NB_D01", time(19,  0), "active"),
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO roster_entries
                (company_id, entry_id, employee_id, service_date,
                 direction, cab_id, scheduled_time, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (company_id, entry_id) DO NOTHING
            """,
            rows,
        )


if __name__ == "__main__":
    with psycopg.connect(DSN) as conn:
        run(conn)
