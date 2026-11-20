"""Restart survival (SPEC §10, §14): pending approval created in app instance 1 is
decided in a brand-new instance 2 (new lifespan, pools, checkpointer) on the same DB."""

import importlib

import psycopg
from fastapi.testclient import TestClient

from app.auth import make_dev_token
from app.config import settings
from app.llm.fake import FakeLLM

DATE = "2026-11-20"


def _h(company, employee, role="employee"):
    return {"Authorization": f"Bearer {make_dev_token(company, employee, role)}"}


def _plans(entry):
    return {"move my drop to 21:00": {
        "intent": "change_shift", "direction": "drop", "target_time": "21:00", "relative_hint": None,
        "chosen_option": None, "ambiguity": "none",
        "tool_calls": [
            {"tool": "get_my_roster", "args": {"date": DATE}},
            {"tool": "check_cab_options", "args": {"entry_id": entry, "new_time": "21:00"}},
        ],
    }}


def test_pending_approval_survives_restart(stage3_data):
    import app.db as db_mod
    import app.graph.approval as approval_mod
    import app.graph.chat as chat_mod
    import app.main as main_mod

    d = stage3_data
    employee = d["employees"][10]
    entry = d["approval_entry"]
    chat_thread = {"configurable": {"thread_id": f"chat:acme:{employee}"}}

    # --- instance 1: employee escalates through chat ---
    app1 = main_mod.app
    with TestClient(app1) as c1:
        chat_mod._llm_singleton = FakeLLM(plans=_plans(entry))
        offer = c1.post("/chat", json={"message": "move my drop to 21:00"}, headers=_h("acme", employee)).json()
        assert "transport manager" in offer["reply"]
        pending = c1.post("/chat", json={"message": "yes, send it to my manager"}, headers=_h("acme", employee)).json()
        assert pending["status"] == "pending_approval"
        rid = pending["request_id"]

    # Instance 1 is fully gone: pools closed, singletons cleared.
    assert db_mod._pool is None
    assert approval_mod.get_approval_graph() is None

    # --- instance 2: brand-new app object, new lifespan, new checkpointer ---
    main_mod = importlib.reload(main_mod)
    app2 = main_mod.app
    assert app2 is not app1
    chat_mod._llm_singleton = None
    with TestClient(app2) as c2:
        chat_mod._llm_singleton = FakeLLM(plans=_plans(entry))
        graph = app2.state.approval_graph
        config = {"configurable": {"thread_id": f"approval:{rid}"}}
        before = graph.get_state(config)
        assert before.values.get("request_id") == rid
        assert any(t.interrupts for t in before.tasks), "approval thread should be paused at interrupt()"

        # Chat context (thread chat:{company}:{employee}) survived too.
        history = app2.state.chat_graph.get_state(chat_thread).values.get("history", [])
        assert any(m["content"] == "yes, send it to my manager" for m in history)

        r = c2.post(f"/manager/requests/{rid}/decision", json={"decision": "approve", "note": "ok"}, headers=_h("acme", d["manager"], "manager"))
        assert r.status_code == 200 and r.json()["status"] == "applied"

        after = graph.get_state(config)
        assert after.next == ()  # thread ran to END from the checkpoint
        assert after.values.get("status") == "applied"
        assert after.values.get("decision", {}).get("decision") == "approve"

        mine = c2.get("/requests/mine", headers=_h("acme", employee)).json()["requests"]
        assert [(m["request_id"], m["status"]) for m in mine] == [(rid, "applied")]

    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT new_cab_id FROM shift_change_requests WHERE company_id='acme' AND id=%s", (rid,))
        new_cab = cur.fetchone()[0]
        cur.execute("SELECT cab_id, scheduled_time::text FROM roster_entries WHERE company_id='acme' AND entry_id=%s", (entry,))
        assert cur.fetchone() == (new_cab, "21:00:00")
        cur.execute("SELECT COUNT(*) FROM notifications WHERE company_id='acme' AND request_id=%s", (rid,))
        assert cur.fetchone()[0] == 1
