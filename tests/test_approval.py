from datetime import timedelta

import psycopg
from fastapi.testclient import TestClient

from app.auth import make_dev_token
from app.clock import clock
from app.config import settings
from app.main import app
from app.tools import raise_shift_change


def _headers(company, employee, role="employee"):
    return {"Authorization": f"Bearer {make_dev_token(company, employee, role)}"}


def test_approval_request_lifecycle_and_company_scope(stage3_data, db_conn):
    d = stage3_data
    employee = d["employees"][10]
    with TestClient(app) as client:
        result = raise_shift_change(__import__("app.auth", fromlist=["Principal"]).Principal("acme", employee, "employee"), d["approval_entry"], "21:00", d["cab"], True)
        assert result.ok and result.data["status"] == "pending_approval"
        rid = result.data["request_id"]
        with db_conn.cursor() as cur:
            cur.execute("SELECT cab_id FROM roster_entries WHERE company_id='acme' AND entry_id=%s", (d["approval_entry"],))
            assert cur.fetchone()[0] == "C_AC_D01"
        manager = _headers("acme", d["manager"], "manager")
        other = _headers("zenith", d["zenith_manager"], "manager")
        assert [r["request_id"] for r in client.get("/manager/requests", headers=manager).json()["requests"]] == [rid]
        assert client.get("/manager/requests", headers=other).json()["requests"] == []
        assert client.post(f"/manager/requests/{rid}/decision", json={"decision": "approve"}, headers=other).status_code == 404
        assert client.get("/manager/requests", headers=_headers("acme", employee)).status_code == 404
        assert client.post(f"/manager/requests/{rid}/decision", json={"decision": "approve"}, headers=_headers("acme", employee)).status_code == 404
        response = client.post(f"/manager/requests/{rid}/decision", json={"decision": "approve"}, headers=manager)
        assert response.status_code == 200 and response.json()["status"] == "applied"
        with db_conn.cursor() as cur:
            cur.execute("SELECT cab_id FROM roster_entries WHERE company_id='acme' AND entry_id=%s", (d["approval_entry"],))
            assert cur.fetchone()[0] == d["cab"]
            cur.execute("SELECT COUNT(*) FROM notifications WHERE company_id='acme' AND request_id=%s", (rid,))
            assert cur.fetchone()[0] == 1
        mine = client.get("/requests/mine", headers=_headers("acme", employee))
        assert mine.status_code == 200 and mine.json()["requests"][0]["status"] == "applied"
        assert client.post(f"/manager/requests/{rid}/decision", json={"decision": "reject"}, headers=manager).status_code == 409
        assert client.post("/manager/requests/not-a-uuid/decision", json={"decision": "approve"}, headers=manager).status_code == 404


def test_rejected_request_retry_gets_new_attempt(stage3_data):
    d = stage3_data
    employee = d["employees"][10]
    with TestClient(app) as client:
        principal = __import__("app.auth", fromlist=["Principal"]).Principal("acme", employee, "employee")
        first = raise_shift_change(principal, d["approval_entry"], "21:00", d["cab"], True)
        headers = _headers("acme", d["manager"], "manager")
        assert client.post(f"/manager/requests/{first.data['request_id']}/decision", json={"decision": "reject"}, headers=headers).json()["status"] == "rejected"
        second = raise_shift_change(principal, d["approval_entry"], "21:00", d["cab"], True)
        assert second.ok and second.data["status"] == "pending_approval"
        assert second.data["request_id"] != first.data["request_id"]
        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM notifications WHERE company_id='acme' AND request_id=%s", (first.data["request_id"],))
            assert cur.fetchone()[0] == 1


def test_approval_when_seat_fills_notifies_nearest_options(stage3_data):
    d = stage3_data
    employee = d["employees"][10]
    with TestClient(app) as client:
        from app.auth import Principal
        pending = raise_shift_change(Principal("acme", employee, "employee"), d["approval_entry"], "21:00", d["cab"], True)
        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute("UPDATE roster_entries SET cab_id=%s,scheduled_time='21:00' WHERE company_id='acme' AND entry_id=%s", (d["cab"], d["entries"][0]))
        response = client.post(f"/manager/requests/{pending.data['request_id']}/decision", json={"decision": "approve"}, headers=_headers("acme", d["manager"], "manager"))
        assert response.status_code == 200 and response.json()["status"] == "approved_not_applied"
        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute("SELECT message FROM notifications WHERE company_id='acme' AND request_id=%s", (pending.data["request_id"],))
            message = cur.fetchone()[0]
        assert "roster is unchanged" in message and "Nearest available options:" in message


def test_expired_approval_is_shown_as_expired(stage3_data):
    d = stage3_data
    employee = d["employees"][10]
    with TestClient(app) as client:
        from app.auth import Principal
        pending = raise_shift_change(Principal("acme", employee, "employee"), d["approval_entry"], "21:00", d["cab"], True)
        rid = pending.data["request_id"]
        old = clock.now() - timedelta(hours=settings.approval_ttl_hours + 1)
        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute("UPDATE shift_change_requests SET created_at=%s WHERE company_id='acme' AND id=%s", (old, rid))
        response = client.get("/requests/mine", headers=_headers("acme", employee))
        assert response.json()["requests"][0]["status"] == "expired"
        decision = client.post(f"/manager/requests/{rid}/decision", json={"decision": "approve"}, headers=_headers("acme", d["manager"], "manager"))
        assert decision.status_code == 409


def test_escalation_is_never_offered_for_r3_or_r4(stage3_data):
    from app.auth import Principal
    d = stage3_data
    principal = Principal("acme", d["employees"][10], "employee")
    # R3: outside acme's 07:00-23:00 window, escalate requested -> still a plain R3 error.
    r3 = raise_shift_change(principal, d["approval_entry"], "23:30", d["cab"], True)
    assert not r3.ok and r3.error_code == "R3_OUTSIDE_SERVICE_WINDOW"
    # R4: fill the 1-seat cab, then escalate an R1-breaking change onto it -> SEAT_TAKEN, no request row.
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute("UPDATE roster_entries SET cab_id=%s,scheduled_time='21:00' WHERE company_id='acme' AND entry_id=%s", (d["cab"], d["entries"][0]))
    r4 = raise_shift_change(principal, d["approval_entry"], "21:00", d["cab"], True)
    assert not r4.ok and r4.error_code == "SEAT_TAKEN"
    assert isinstance(r4.error_params.get("nearest_slots"), list)
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM shift_change_requests WHERE company_id='acme' AND employee_id=%s", (d["employees"][10],))
        assert cur.fetchone()[0] == 0


def test_duplicate_escalation_returns_same_pending_request(stage3_data):
    from app.auth import Principal
    d = stage3_data
    principal = Principal("acme", d["employees"][10], "employee")
    with TestClient(app):
        first = raise_shift_change(principal, d["approval_entry"], "21:00", d["cab"], True)
        second = raise_shift_change(principal, d["approval_entry"], "21:00", d["cab"], True)
    assert first.data["status"] == second.data["status"] == "pending_approval"
    assert first.data["request_id"] == second.data["request_id"]
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT cab_id FROM roster_entries WHERE company_id='acme' AND entry_id=%s", (d["approval_entry"],))
        assert cur.fetchone()[0] == "C_AC_D01"  # no seat held, roster untouched while pending
