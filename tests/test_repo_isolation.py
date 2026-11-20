"""
tests/test_repo_isolation.py — verify multi-tenant isolation in repo functions.

Employee code E101 exists in both acme and zenith (seed invariant).
These tests confirm that each company's Principal can only see its own data.

Requires a live Postgres connection (run via docker compose).
"""

import pytest
from datetime import date

import psycopg

from app.auth import Principal
from app.repo import get_roster, get_roster_entry


SEED_DATE = date(2026, 11, 20)

# Principals for the same employee code E101 in two different companies.
ACME_E101    = Principal(company_id="acme",   employee_id="E101", role="employee")
ZENITH_E101  = Principal(company_id="zenith", employee_id="E101", role="employee")


@pytest.fixture(scope="module")
def conn(db_conn):
    """Re-use the session-scoped DB connection from conftest."""
    return db_conn


class TestRosterIsolation:
    def test_acme_e101_sees_acme_roster_only(self, conn):
        entries = get_roster(conn, ACME_E101, SEED_DATE)
        assert len(entries) >= 1
        # All returned entries must belong to acme.
        for e in entries:
            assert e.company_id == "acme", f"Got company {e.company_id} in acme query"
            assert e.employee_id == "E101"

    def test_zenith_e101_sees_zenith_roster_only(self, conn):
        entries = get_roster(conn, ZENITH_E101, SEED_DATE)
        assert len(entries) >= 1
        for e in entries:
            assert e.company_id == "zenith", f"Got company {e.company_id} in zenith query"
            assert e.employee_id == "E101"

    def test_rosters_are_different(self, conn):
        """Same employee code, different data — proves tenant isolation."""
        acme_entries   = get_roster(conn, ACME_E101,   SEED_DATE)
        zenith_entries = get_roster(conn, ZENITH_E101, SEED_DATE)

        acme_cabs   = {e.cab_id for e in acme_entries}
        zenith_cabs = {e.cab_id for e in zenith_entries}

        # The two sets of cabs must not overlap — they're in different companies.
        assert acme_cabs.isdisjoint(zenith_cabs), (
            f"Cab IDs overlap between companies: {acme_cabs & zenith_cabs}"
        )

    def test_acme_principal_cannot_read_zenith_entry(self, conn):
        """
        An acme Principal asking for a zenith entry_id returns None —
        the WHERE clause includes company_id.
        """
        # First grab a real zenith entry id.
        zenith_entries = get_roster(conn, ZENITH_E101, SEED_DATE)
        assert zenith_entries, "Need at least one zenith entry for this test"
        zenith_entry_id = zenith_entries[0].entry_id

        # Try to fetch it as the acme employee — must return None.
        result = get_roster_entry(conn, ACME_E101, zenith_entry_id)
        assert result is None, (
            f"Security violation: acme E101 could read zenith entry {zenith_entry_id}"
        )

    def test_zenith_principal_cannot_read_acme_entry(self, conn):
        acme_entries = get_roster(conn, ACME_E101, SEED_DATE)
        assert acme_entries
        acme_entry_id = acme_entries[0].entry_id

        result = get_roster_entry(conn, ZENITH_E101, acme_entry_id)
        assert result is None, (
            f"Security violation: zenith E101 could read acme entry {acme_entry_id}"
        )

    def test_acme_drop_at_2030_exists(self, conn):
        """Seed invariant: E101 in acme has a drop at 20:30."""
        from datetime import time
        entries = get_roster(conn, ACME_E101, SEED_DATE)
        drops = [e for e in entries if e.direction == "drop"]
        assert any(e.scheduled_time == time(20, 30) for e in drops), (
            "Missing seed invariant: acme E101 should have a drop at 20:30"
        )

    def test_e102_has_both_pickup_and_drop(self, conn):
        """Seed invariant: one employee has both pickup and drop today."""
        principal = Principal(company_id="acme", employee_id="E102", role="employee")
        entries = get_roster(conn, principal, SEED_DATE)
        directions = {e.direction for e in entries}
        assert "pickup" in directions and "drop" in directions, (
            "Missing seed invariant: E102 should have both pickup and drop"
        )
