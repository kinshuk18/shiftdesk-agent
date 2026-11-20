"""Transactional shift-change writes and approval lifecycle helpers."""

from __future__ import annotations

from datetime import timedelta
import hashlib
import uuid

import psycopg
from psycopg.types.json import Jsonb

from app.auth import Principal
from app.clock import clock
from app.config import settings
from app.db import get_pool


def idempotency_key_for(company_id, employee_id, entry_id, cab_id, new_time):
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute("""SELECT COUNT(*) FROM shift_change_requests WHERE company_id=%s AND employee_id=%s AND entry_id=%s AND new_cab_id=%s AND new_time=%s AND status IN ('rejected','expired','approved_not_applied','failed')""", (company_id, employee_id, entry_id, cab_id, new_time))
        attempt = cur.fetchone()[0]
    raw = f"{company_id}|{employee_id}|{entry_id}|{cab_id}|{new_time}"
    if attempt:
        raw += f"|{attempt}"
    return hashlib.sha256(raw.encode()).hexdigest(), attempt


def apply_change(cur, company_id, employee_id, entry_id, new_cab_id, new_time, actor_id, actor_role, action, request_id):
    """The only code path that moves a seat. Caller owns the transaction."""
    cur.execute("SELECT capacity FROM cabs WHERE company_id=%s AND cab_id=%s FOR UPDATE", (company_id, new_cab_id))
    cab = cur.fetchone()
    if cab is None:
        return "CAB_NOT_FOUND"
    cur.execute("SELECT service_date, cab_id FROM roster_entries WHERE company_id=%s AND employee_id=%s AND entry_id=%s FOR UPDATE", (company_id, employee_id, entry_id))
    entry = cur.fetchone()
    if entry is None:
        return "ENTRY_NOT_FOUND"
    cur.execute("SELECT COUNT(*) FROM roster_entries WHERE company_id=%s AND cab_id=%s AND service_date=%s AND entry_id<>%s AND status IN ('active','pending_change')", (company_id, new_cab_id, entry[0], entry_id))
    if cur.fetchone()[0] >= cab[0]:
        return "SEAT_TAKEN"
    cur.execute("UPDATE roster_entries SET cab_id=%s, scheduled_time=%s, status='active' WHERE company_id=%s AND employee_id=%s AND entry_id=%s", (new_cab_id, new_time, company_id, employee_id, entry_id))
    # Request id comes from the caller: created_at uses the frozen clock, so
    # "latest request" lookups by created_at are ambiguous.
    # cur.execute("SELECT id FROM shift_change_requests WHERE company_id=%s AND employee_id=%s AND entry_id=%s AND new_cab_id=%s AND new_time=%s ORDER BY created_at DESC LIMIT 1", (company_id, employee_id, entry_id, new_cab_id, new_time))
    cur.execute("INSERT INTO audit_log(company_id,actor_id,actor_role,action,request_id,detail) VALUES(%s,%s,%s,%s,%s,%s)", (company_id, actor_id, actor_role, action, request_id, Jsonb({"entry_id": entry_id, "old_cab_id": entry[1], "new_cab_id": new_cab_id, "new_time": str(new_time)})))
    return "OK"


def _insert_request(cur, principal, entry_id, new_time, cab_id, key, status):
    cur.execute("""INSERT INTO shift_change_requests(company_id,employee_id,entry_id,old_cab_id,new_cab_id,new_time,status,idempotency_key,created_at) SELECT %s,%s,re.entry_id,re.cab_id,%s,%s,%s,%s,%s FROM roster_entries re WHERE re.company_id=%s AND re.employee_id=%s AND re.entry_id=%s ON CONFLICT(company_id,idempotency_key) DO NOTHING RETURNING id,status""", (principal.company_id, principal.employee_id, cab_id, new_time, status, key, clock.now(), principal.company_id, principal.employee_id, entry_id))
    return cur.fetchone()


def _existing(cur, company_id, key):
    cur.execute("SELECT id,status FROM shift_change_requests WHERE company_id=%s AND idempotency_key=%s", (company_id, key))
    return cur.fetchone()


def submit_direct(principal, entry_id, new_time, cab_id, key):
    with get_pool().connection() as conn:
        try:
            with conn.transaction():
                with conn.cursor() as cur:
                    row = _insert_request(cur, principal, entry_id, new_time, cab_id, key, "applied")
                    if row is None:
                        old = _existing(cur, principal.company_id, key)
                        return (str(old[0]), "already_applied" if old[1] == "applied" else old[1]) if old else (None, "failed")
                    result = apply_change(cur, principal.company_id, principal.employee_id, entry_id, cab_id, new_time, principal.employee_id, principal.role, "shift_change_applied", row[0])
                    if result != "OK":
                        raise _RollbackResult(result)
                    return str(row[0]), "applied"
        except _RollbackResult as exc:
            return None, exc.value


class _RollbackResult(Exception):
    def __init__(self, value): self.value = value


def create_pending_request(principal, entry_id, new_time, cab_id, key, request_id):
    with get_pool().connection() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("""INSERT INTO shift_change_requests(id,company_id,employee_id,entry_id,old_cab_id,new_cab_id,new_time,status,idempotency_key,created_at) SELECT %s,%s,%s,re.entry_id,re.cab_id,%s,%s,'pending_approval',%s,%s FROM roster_entries re WHERE re.company_id=%s AND re.employee_id=%s AND re.entry_id=%s ON CONFLICT(company_id,idempotency_key) DO NOTHING RETURNING id,status""", (request_id, principal.company_id, principal.employee_id, cab_id, new_time, key, clock.now(), principal.company_id, principal.employee_id, entry_id))
                row = cur.fetchone()
                if row is None:
                    old = _existing(cur, principal.company_id, key)
                    return (str(old[0]), old[1], False) if old else (request_id, "failed", False)
                cur.execute("INSERT INTO audit_log(company_id,actor_id,actor_role,action,request_id,detail) VALUES(%s,%s,%s,'shift_change_requested',%s,%s)", (principal.company_id, principal.employee_id, principal.role, row[0], Jsonb({"entry_id": entry_id})))
                return str(row[0]), row[1], True


def apply_approved(company_id, request_id, decided_by, note):
    with get_pool().connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT employee_id,entry_id,new_cab_id,new_time,status FROM shift_change_requests WHERE company_id=%s AND id=%s FOR UPDATE", (company_id, request_id))
        row = cur.fetchone()
        if not row: return None
        if row[4] != "pending_approval": return row[4]
        result = apply_change(cur, company_id, row[0], row[1], row[2], row[3], decided_by, "manager", "shift_change_applied", request_id)
        final = "applied" if result == "OK" else "approved_not_applied" if result == "SEAT_TAKEN" else "failed"
        cur.execute("UPDATE shift_change_requests SET status=%s,decided_by=%s,decision_note=%s,decided_at=%s WHERE company_id=%s AND id=%s", (final, decided_by, note, clock.now(), company_id, request_id))
        if final == "failed":
            cur.execute("INSERT INTO audit_log(company_id,actor_id,actor_role,action,request_id,detail) VALUES(%s,%s,'manager','shift_change_failed',%s,%s)", (company_id, decided_by, request_id, Jsonb({"reason": result})))
        return final


def reject_request(company_id, request_id, decided_by, note):
    with get_pool().connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT status FROM shift_change_requests WHERE company_id=%s AND id=%s FOR UPDATE", (company_id, request_id))
        row = cur.fetchone()
        if not row: return None
        if row[0] != "pending_approval": return row[0]
        cur.execute("UPDATE shift_change_requests SET status='rejected',decided_by=%s,decision_note=%s,decided_at=%s WHERE company_id=%s AND id=%s", (decided_by, note, clock.now(), company_id, request_id))
        cur.execute("INSERT INTO audit_log(company_id,actor_id,actor_role,action,request_id) VALUES(%s,%s,'manager','shift_change_rejected',%s)", (company_id, decided_by, request_id))
        return "rejected"


def expire_stale(company_id):
    cutoff = clock.now() - timedelta(hours=settings.approval_ttl_hours)
    with get_pool().connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("UPDATE shift_change_requests SET status='expired',decided_at=%s WHERE company_id=%s AND status='pending_approval' AND created_at < %s RETURNING id,employee_id", (clock.now(), company_id, cutoff))
        rows = cur.fetchall()
        for rid, emp in rows:
            cur.execute("INSERT INTO audit_log(company_id,actor_id,actor_role,action,request_id) VALUES(%s,'system','system','shift_change_expired',%s)", (company_id, rid))
            cur.execute("INSERT INTO notifications(company_id,employee_id,request_id,message) SELECT %s,%s,%s,'Your approval request expired. Your roster was not changed. Please submit a new request or contact the transport desk.' WHERE NOT EXISTS(SELECT 1 FROM notifications WHERE request_id=%s)", (company_id, emp, rid, rid))
        return len(rows)
