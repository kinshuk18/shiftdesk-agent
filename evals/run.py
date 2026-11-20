"""
evals/run.py — eval runner for evals/eval_set.jsonl (SPEC §15).

    python -m evals.run            # fake LLM, deterministic, all cases; rewrites actual/pass in place
    python -m evals.run --live     # configured real provider, subset; writes evals/live_results.jsonl
    python -m evals.run --only cut-01,rst-02

Runs in-process against the compose Postgres (FastAPI TestClient, real graphs,
real tools). Cases that write data create their own throw-away employees,
entries and cab (setup op "tenant") and delete them afterwards, so seeded data
is never modified. Exit code is 1 if any case fails.

Each line has exactly the keys: id, input, expected, actual, pass, notes.
  input:    {token_as, setup: [ops], messages: [steps]}
  expected: {say: <plain sentence>, ...machine-checkable assertions}
Placeholders such as {emp0}, {ent0}, {cab}, {mgr}, {zmgr}, {last_request}
are filled per run (see Tenant).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

LIVE = "--live" in sys.argv
if not LIVE:
    os.environ["LLM_PROVIDER"] = "fake"  # must happen before app.config is imported

import jwt  # noqa: E402
import psycopg  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import messages  # noqa: E402
from app.auth import Principal, make_dev_token  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm.base import LLMError, LLMQuotaError  # noqa: E402

EVAL_FILE = Path(__file__).with_name("eval_set.jsonl")
LIVE_FILE = Path(__file__).with_name("live_results.jsonl")
KEYS = ["id", "input", "expected", "actual", "pass", "notes"]
CATEGORIES = {
    "iso": "isolation", "cut": "cut-off", "rst": "restart", "sel": "tool selection",
    "tfl": "tool failure", "amb": "ambiguity", "con": "concurrency", "inj": "injection", "auth": "auth",
}
SEED_DATE = date(2026, 11, 20)

# Template code -> fixed text that identifies it in a reply (text before the first placeholder).
TEMPLATES = {
    name: getattr(messages, name).split("{")[0].strip()[:45]
    for name in ("GENERIC_FAILURE", "TOOL_TIMEOUT", "WRITE_TOOL_UNCERTAIN", "CAB_FILLED", "PENDING_APPROVAL",
                 "ESCALATION_OFFER", "ESCALATION_PICK_TIME", "ESCALATION_NOTHING_PENDING", "ESCALATION_DECLINED",
                 "ENTRY_NOT_FOUND", "SESSION_INVALID")
}
TEMPLATES.update({code: text.split("{")[0].strip()[:45] for code, text in messages.RULE_MESSAGES.items()})


# ---------------------------------------------------------------------------
# LLM wrappers: count every provider call per message, record provider errors
# ---------------------------------------------------------------------------


class CountingLLM:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.errors: list[Exception] = []

    def chat(self, msgs):
        self.calls += 1
        try:
            return self.inner.chat(msgs)
        except Exception as exc:
            self.errors.append(exc)
            raise


class AlwaysMalformed:
    def chat(self, msgs):
        return "{not json"


def build_llm(setup):
    if LIVE:
        import app.main as main_mod
        return CountingLLM(main_mod._build_llm())
    from app.llm.fake import FakeLLM
    plans, mode = {}, None
    for op in setup:
        if op["op"] == "plans":
            plans.update(op["plans"])
        if op["op"] == "llm_fail":
            mode = op["mode"]
    if mode == "malformed_twice":
        return CountingLLM(AlwaysMalformed())
    return CountingLLM(FakeLLM(plans=plans, fail_mode=mode))


# ---------------------------------------------------------------------------
# Throw-away tenant data (never touches seeded rows)
# ---------------------------------------------------------------------------


class Tenant:
    def __init__(self, conn, op):
        s = uuid.uuid4().hex[:8]
        n = op.get("employees", 3)
        times = {int(k): v for k, v in op.get("times", {}).items()}
        self.conn = conn
        self.emps = [f"EV{s}{i}" for i in range(n)]
        self.ents = [f"EVE{s}{i}" for i in range(n)]
        self.vars = {f"{{emp{i}}}": e for i, e in enumerate(self.emps)}
        self.vars.update({f"{{ent{i}}}": e for i, e in enumerate(self.ents)})
        self.vars.update({"{cab}": f"EVC{s}", "{mgr}": f"EVM{s}", "{zmgr}": f"EVZ{s}"})
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany("INSERT INTO employees(company_id,employee_id,name) VALUES('acme',%s,%s)",
                            [(e, e) for e in self.emps + [self.vars["{mgr}"]]])
            cur.execute("INSERT INTO employees(company_id,employee_id,name) VALUES('zenith',%s,%s)",
                        (self.vars["{zmgr}"], self.vars["{zmgr}"]))
            cur.execute("INSERT INTO cabs(company_id,cab_id,route_id,direction,departure_time,capacity) "
                        "VALUES('acme',%s,'R1','drop',%s,%s)",
                        (self.vars["{cab}"], op.get("cab_time", "21:10"), op.get("cab_capacity", 1)))
            cur.executemany(
                "INSERT INTO roster_entries(company_id,entry_id,employee_id,service_date,direction,cab_id,scheduled_time,status) "
                "VALUES('acme',%s,%s,%s,'drop','C_AC_D01',%s,'active')",
                [(self.ents[i], self.emps[i], SEED_DATE, times.get(i, op.get("entry_time", "22:00"))) for i in range(n)])

    def cleanup(self):
        people = self.emps + [self.vars["{mgr}"]]
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute("DELETE FROM notifications WHERE company_id='acme' AND employee_id=ANY(%s)", (people,))
            cur.execute("DELETE FROM shift_change_requests WHERE company_id='acme' AND employee_id=ANY(%s)", (people,))
            cur.execute("DELETE FROM roster_entries WHERE company_id='acme' AND employee_id=ANY(%s)", (people,))
            cur.execute("DELETE FROM employees WHERE company_id='acme' AND employee_id=ANY(%s)", (people,))
            cur.execute("DELETE FROM employees WHERE company_id='zenith' AND employee_id=%s", (self.vars["{zmgr}"],))
            cur.execute("DELETE FROM cabs WHERE company_id='acme' AND cab_id=%s", (self.vars["{cab}"],))


def fill(obj, vars_):
    text = json.dumps(obj)
    for k, v in vars_.items():
        text = text.replace(k, v)
    return json.loads(text)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------


def token_for(who, kind=None):
    if kind == "missing":
        return {}
    if kind in ("expired", "forged"):
        now = datetime.now(tz=timezone.utc)
        exp = now - timedelta(minutes=5) if kind == "expired" else now + timedelta(minutes=30)
        secret = settings.jwt_secret if kind == "expired" else "not-the-real-secret"
        tok = jwt.encode({"sub": who["employee_id"], "company_id": who["company_id"], "role": who.get("role", "employee"),
                          "iat": now - timedelta(minutes=10), "exp": exp}, secret, algorithm="HS256")
        return {"Authorization": f"Bearer {tok}"}
    return {"Authorization": f"Bearer {make_dev_token(who['company_id'], who['employee_id'], who.get('role', 'employee'))}"}


class Run:
    """Holds the live app instance; restart() drops it and builds a brand-new one."""

    def __init__(self, llm):
        import app.graph.chat as chat_mod
        import app.tools as tools_mod
        self.chat_mod = chat_mod
        self.llm = llm
        self.turn_tools: list[str] = []
        real_dispatch = tools_mod.dispatch_tool

        def recording_dispatch(principal, name, args):
            self.turn_tools.append(name)
            return real_dispatch(principal, name, args)

        chat_mod.dispatch_tool = recording_dispatch
        self._real_dispatch = real_dispatch
        import app.main as main_mod
        self.main_mod = main_mod
        self._enter()

    def _enter(self):
        self.ctx = TestClient(self.main_mod.app, raise_server_exceptions=False)
        self.client = self.ctx.__enter__()
        self.chat_mod._llm_singleton = self.llm  # lifespan installs its own; use the counting one

    def restart(self):
        self.ctx.__exit__(None, None, None)
        self.main_mod = importlib.reload(self.main_mod)
        self._enter()

    def close(self):
        self.ctx.__exit__(None, None, None)
        self.chat_mod.dispatch_tool = self._real_dispatch


def roster_of(conn, company, employee):
    with conn.cursor() as cur:
        cur.execute("SELECT entry_id, cab_id, scheduled_time::text, status FROM roster_entries "
                    "WHERE company_id=%s AND employee_id=%s ORDER BY entry_id", (company, employee))
        rows = cur.fetchall()
    conn.commit()
    return rows


def q1(conn, sql, args):
    with conn.cursor() as cur:
        cur.execute(sql, args)
        row = cur.fetchone()
    conn.commit()
    return row


def run_case(case, conn):
    inp, exp = case["input"], case["expected"]
    setup = inp.get("setup", [])
    tenant = None
    tool_patch = None
    run = None
    try:
        with conn.cursor() as cur:  # fresh chat context for every case
            for t in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                cur.execute(f"DELETE FROM {t} WHERE thread_id LIKE 'chat:%%'")
        conn.commit()
        vars_ = {}
        for op in setup:
            if op["op"] == "tenant":
                tenant = Tenant(conn, op)
                vars_.update(tenant.vars)
        inp, exp = fill(inp, vars_), fill(exp, vars_)
        setup = inp.get("setup", [])
        who = inp.get("token_as") or {}

        import app.tools as tools_mod
        for op in setup:
            if op["op"] == "tool_fail":
                name, mode = op["tool"], op["mode"]
                orig = tools_mod.TOOL_REGISTRY[name]

                def broken(*a, _mode=mode, **k):
                    if _mode == "timeout":
                        time.sleep(2.0)  # > 1.5 s tool budget; thread is abandoned, as in prod
                        return orig(*a, **k)
                    raise RuntimeError("injected tool failure")

                tool_patch = (name, orig)
                tools_mod.TOOL_REGISTRY[name] = broken

        llm = build_llm(setup)
        before = roster_of(conn, who["company_id"], who["employee_id"]) if who else None
        other = exp.get("other_roster_unchanged")
        other_before = roster_of(conn, other["company_id"], other["employee_id"]) if other else None
        run = Run(llm)
        turns, http, lists, last_request = [], [], [], None
        race = None

        for step in inp.get("messages", []):
            if isinstance(step, str):
                step = {"chat": step}
            step = fill(step, {"{last_request}": last_request or ""})
            if "chat" in step:
                run.turn_tools = []
                llm.calls = 0
                as_ = step.get("as", who)
                r = run.client.post("/chat", json={"message": step["chat"]}, headers=token_for(as_, step.get("token")))
                body = r.json()
                reply = body.get("reply") or body.get("detail") or ""
                turns.append({"reply": reply, "status": body.get("status"), "http": r.status_code,
                              "tools": list(run.turn_tools), "llm_calls": llm.calls})
                http.append(r.status_code)
                last_request = body.get("request_id") or last_request
            elif "restart" in step:
                run.restart()
            elif "decide" in step:
                r = run.client.post(f"/manager/requests/{step.get('request', last_request)}/decision",
                                    json={"decision": step["decide"], "note": "eval"}, headers=token_for(step["as"]))
                http.append(r.status_code)
                lists.append(None)
            elif "get" in step:
                r = run.client.get(step["get"], headers=token_for(step["as"]))
                http.append(r.status_code)
                lists.append(len(r.json().get("requests", [])) if r.status_code == 200 else None)
            elif "escalate_direct" in step:
                e = step["escalate_direct"]
                res = tools_mod.raise_shift_change(Principal("acme", e["employee"], "employee"),
                                                   e["entry"], e["new_time"], e["cab"], True)
                last_request = (res.data or {}).get("request_id")
            elif "sql" in step:
                with conn.cursor() as cur:
                    cur.execute(step["sql"])
                conn.commit()
            elif "race" in step:
                cfg = step["race"]
                n = len(tenant.emps)
                calls = ([(tenant.emps[i], tenant.ents[i]) for i in range(n)] if cfg == "distinct"
                         else [(tenant.emps[0], tenant.ents[0])] * n)
                barrier = threading.Barrier(len(calls))

                def go(c):
                    barrier.wait(timeout=10)
                    return tools_mod.raise_shift_change(Principal("acme", c[0], "employee"), c[1], "21:10", tenant.vars["{cab}"], False)

                with ThreadPoolExecutor(max_workers=len(calls)) as ex:
                    results = list(ex.map(go, calls))
                race = {
                    "applied": sum(bool(x.ok and x.data["status"] == "applied") for x in results),
                    "already_applied": sum(bool(x.ok and x.data["status"] == "already_applied") for x in results),
                    "seat_taken": sum(x.error_code == "SEAT_TAKEN" for x in results),
                    "seats_used": q1(conn, "SELECT COUNT(*) FROM roster_entries WHERE company_id='acme' AND cab_id=%s "
                                           "AND status IN ('active','pending_change')", (tenant.vars["{cab}"],))[0],
                    "request_rows": q1(conn, "SELECT COUNT(*) FROM shift_change_requests WHERE company_id='acme' "
                                             "AND employee_id=ANY(%s)", (tenant.emps,))[0],
                }

        # ---- checks ----
        fails = []
        last = turns[-1] if turns else {}
        reply = last.get("reply", "")
        low = reply.lower()

        def want(cond, msg):
            if not cond:
                fails.append(msg)

        for code in ([exp["template"]] if isinstance(exp.get("template"), str) else exp.get("template", [])):
            want(TEMPLATES[code].lower() in low, f"template {code} not in reply")
        for m in exp.get("mentions", []):
            want(m.lower() in low, f"missing '{m}'")
        for m in exp.get("not_mentions", []):
            want(all(m.lower() not in t["reply"].lower() for t in turns), f"leaked '{m}'")
        for t in turns:
            want("traceback" not in t["reply"].lower() and 'file "' not in t["reply"].lower(), "traceback in reply")
            want(t["llm_calls"] <= 4, f"{t['llm_calls']} LLM calls > 4")
        if "status" in exp:
            want(last.get("status") == exp["status"], f"status {last.get('status')} != {exp['status']}")
        if "http_status" in exp:
            want(http and http[-1] == exp["http_status"], f"http {http[-1:]} != {exp['http_status']}")
        if "http_statuses" in exp:
            want(http == exp["http_statuses"], f"http {http} != {exp['http_statuses']}")
        if "list_lens" in exp:
            want(lists == exp["list_lens"], f"list lens {lists} != {exp['list_lens']}")
        if "tools" in exp:
            want(last.get("tools") == exp["tools"], f"tools {last.get('tools')} != {exp['tools']}")
        if "llm_calls" in exp:
            want(last.get("llm_calls") == exp["llm_calls"], f"llm calls {last.get('llm_calls')} != {exp['llm_calls']}")
        if exp.get("roster_unchanged"):
            want(roster_of(conn, who["company_id"], who["employee_id"]) == before, "roster changed")
        if other:
            want(roster_of(conn, other["company_id"], other["employee_id"]) == other_before, "other tenant roster changed")
        for entry, cab in exp.get("roster_cab", {}).items():
            row = q1(conn, "SELECT cab_id FROM roster_entries WHERE company_id='acme' AND entry_id=%s", (entry,))
            want(row and row[0] == cab, f"{entry} on {row and row[0]} not {cab}")
        if "request_status" in exp:
            row = q1(conn, "SELECT status FROM shift_change_requests WHERE id::text=%s", (last_request or "",))
            want(row and row[0] == exp["request_status"], f"request status {row and row[0]} != {exp['request_status']}")
        if "request_count" in exp:
            n = q1(conn, "SELECT COUNT(*) FROM shift_change_requests WHERE company_id=%s AND employee_id=%s",
                   (who["company_id"], who["employee_id"]))[0]
            want(n == exp["request_count"], f"request rows {n} != {exp['request_count']}")
        if "notification_contains" in exp:
            row = q1(conn, "SELECT message FROM notifications WHERE request_id::text=%s", (last_request or "",))
            want(row and exp["notification_contains"].lower() in row[0].lower(), f"notification {row and row[0]!r}")
        if "awaiting" in exp:
            thread = {"configurable": {"thread_id": f"chat:{who['company_id']}:{who['employee_id']}"}}
            got = run.client.app.state.chat_graph.get_state(thread).values.get("awaiting")
            want(got == exp["awaiting"], f"awaiting {got!r} != {exp['awaiting']!r}")
        if exp.get("approval_done"):
            snap = run.client.app.state.approval_graph.get_state({"configurable": {"thread_id": f"approval:{last_request}"}})
            want(snap.values and snap.next == (), f"approval thread next={snap.next}")
        if "race" in exp:
            for k, v in exp["race"].items():
                want(race and race.get(k) == v, f"race {k}={race and race.get(k)} != {v}")

        provider_errors = [e for e in getattr(llm, "errors", [])]
        actual = {
            "replies": [t["reply"][:220] for t in turns],
            "statuses": [t["status"] for t in turns],
            "http": http,
            "tools": [t["tools"] for t in turns],
            "llm_calls": [t["llm_calls"] for t in turns],
        }
        if race:
            actual["race"] = race
        if fails:
            actual["failures"] = fails
        skipped = LIVE and any(isinstance(e, LLMQuotaError) or ("HTTP 5" in str(e)) for e in provider_errors)
        if skipped:
            actual["skipped"] = "provider quota/429/5xx: " + str(provider_errors[0])[:120]
        return actual, (None if skipped else not fails)
    except Exception as exc:  # a broken case is a failure, never a crash of the whole run
        return {"error": f"{type(exc).__name__}: {exc}"[:400]}, False
    finally:
        if run:
            run.close()
        if tool_patch:
            import app.tools as tools_mod
            tools_mod.TOOL_REGISTRY[tool_patch[0]] = tool_patch[1]
        if tenant:
            try:
                tenant.cleanup()
            except Exception as exc:
                print(f"  cleanup failed for {case['id']}: {exc}", file=sys.stderr)


def live_subset(case):
    """Cases that only make sense with the real model: no scripted plans or injected LLM failures."""
    ops = {op["op"] for op in case["input"].get("setup", [])}
    return not ops & {"plans", "llm_fail"} and case["id"].split("-")[0] in ("iso", "cut", "sel", "amb", "inj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--verbose", action="store_true", help="show app logs (tracebacks are logged server-side by design)")
    args = ap.parse_args()
    if not args.verbose:
        logging.disable(logging.CRITICAL)

    cases = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]
    for c in cases:
        assert list(c.keys()) == KEYS, f"{c.get('id')}: keys must be exactly {KEYS}"

    if args.live:
        provider = settings.llm_provider.lower()
        key = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}.get(provider)
        if not key or not os.environ.get(key):
            print(f"--live needs LLM_PROVIDER=groq|gemini and its API key (got LLM_PROVIDER={provider!r}, "
                  f"{key or 'no key var'} {'set' if key and os.environ.get(key) else 'not set'}). Nothing was run.")
            return 2
        selected = [c for c in cases if live_subset(c)]
    else:
        selected = cases
    if args.only:
        wanted = set(args.only.split(","))
        selected = [c for c in selected if c["id"] in wanted]

    conn = psycopg.connect(settings.database_url)
    results = {}
    for c in selected:
        t0 = time.perf_counter()
        actual, ok = run_case(c, conn)
        results[c["id"]] = (actual, ok)
        mark = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"{mark}  {c['id']:<8} {time.perf_counter() - t0:5.2f}s  {'; '.join(actual.get('failures', [])) or actual.get('error', '')}")
    conn.close()

    if args.live:
        with LIVE_FILE.open("w") as f:
            for c in selected:
                actual, ok = results[c["id"]]
                f.write(json.dumps({**c, "actual": actual, "pass": ok}) + "\n")
    else:
        with EVAL_FILE.open("w") as f:
            for c in cases:
                if c["id"] in results:
                    c["actual"], c["pass"] = results[c["id"]]
                f.write(json.dumps({k: c[k] for k in KEYS}) + "\n")

    print(f"\n{'category':<16}{'total':>6}{'pass':>6}{'fail':>6}{'skip':>6}")
    totals = [0, 0, 0, 0]
    for prefix, name in CATEGORIES.items():
        rows = [ok for cid, (_, ok) in results.items() if cid.split("-")[0] == prefix]
        if not rows:
            continue
        line = [len(rows), rows.count(True), rows.count(False), rows.count(None)]
        totals = [a + b for a, b in zip(totals, line)]
        print(f"{name:<16}{line[0]:>6}{line[1]:>6}{line[2]:>6}{line[3]:>6}")
    print(f"{'TOTAL':<16}{totals[0]:>6}{totals[1]:>6}{totals[2]:>6}{totals[3]:>6}")
    print(f"wrote {'evals/live_results.jsonl' if args.live else 'evals/eval_set.jsonl'}")
    return 1 if totals[2] else 0


if __name__ == "__main__":
    sys.exit(main())
