from concurrent.futures import ThreadPoolExecutor
import threading

from app.auth import Principal
from app.tools import raise_shift_change


def _race(calls):
    barrier = threading.Barrier(len(calls))

    def invoke(args):
        barrier.wait(timeout=10)
        principal, entry_id, cab_id = args
        return raise_shift_change(principal, entry_id, "21:00", cab_id, False)

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return list(pool.map(invoke, [(principal, entry, cab) for principal, entry, cab in calls]))


def test_competing_employees_get_only_one_free_seat(db_conn, stage3_data):
    data = stage3_data
    calls = [(Principal("acme", data["employees"][i], "employee"), data["entries"][i], data["cab"]) for i in range(10)]
    results = _race(calls)
    assert sum(bool(r.ok and r.data["status"] == "applied") for r in results) == 1
    assert sum(r.error_code == "SEAT_TAKEN" for r in results) == 9
    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM roster_entries WHERE company_id='acme' AND cab_id=%s AND service_date='2026-11-20' AND status IN ('active','pending_change')", (data["cab"],))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM audit_log WHERE company_id='acme' AND action='shift_change_applied' AND detail->>'new_cab_id'=%s", (data["cab"],))
        assert cur.fetchone()[0] == 1


def test_identical_parallel_retries_are_idempotent(db_conn, stage3_data):
    data = stage3_data
    principal = Principal("acme", data["employees"][0], "employee")
    results = _race([(principal, data["entries"][0], data["cab"])] * 10)
    assert sum(bool(r.ok and r.data["status"] == "applied") for r in results) == 1
    assert sum(bool(r.ok and r.data["status"] == "already_applied") for r in results) == 9
    request_id = next(r.data["request_id"] for r in results if r.ok)
    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM shift_change_requests WHERE company_id='acme' AND id=%s", (request_id,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT COUNT(*) FROM audit_log WHERE company_id='acme' AND action='shift_change_applied' AND request_id=%s", (request_id,))
        assert cur.fetchone()[0] == 1
