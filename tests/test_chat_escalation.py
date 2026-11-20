"""Chat-level escalation flow (SPEC §7, §9, §10): offer -> confirm -> pending, never applied."""

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.auth import make_dev_token
from app.config import settings

DATE = "2026-11-20"


def _h(company, employee, role="employee"):
    return {"Authorization": f"Bearer {make_dev_token(company, employee, role)}"}


def _plan(intent, tool_calls, **kw):
    return {"intent": intent, "direction": kw.get("direction"), "target_time": kw.get("target_time"),
            "relative_hint": None, "chosen_option": None, "ambiguity": "none", "tool_calls": tool_calls}


def _plans(entry):
    return {
        "move my drop to 21:00": _plan("change_shift", [
            {"tool": "get_my_roster", "args": {"date": DATE}},
            {"tool": "check_cab_options", "args": {"entry_id": entry, "new_time": "21:00"}},
        ], direction="drop", target_time="21:00"),
        "move my drop to 23:30": _plan("change_shift", [
            {"tool": "check_cab_options", "args": {"entry_id": entry, "new_time": "23:30"}},
        ], direction="drop", target_time="23:30"),
        "force it through": _plan("change_shift", [
            {"tool": "raise_shift_change", "args": {"entry_id": entry, "new_time": "21:00", "cab_id": "C_AC_D02", "escalate": True}},
        ]),
    }


@pytest.fixture
def chat(stage3_data):
    from app.main import app
    import app.graph.chat as chat_mod
    from app.llm.fake import FakeLLM
    with TestClient(app, raise_server_exceptions=False) as client:
        chat_mod._llm_singleton = FakeLLM(plans=_plans(stage3_data["approval_entry"]))
        employee = stage3_data["employees"][10]
        yield client, employee, stage3_data


def _say(client, employee, text):
    r = client.post("/chat", json={"message": text}, headers=_h("acme", employee))
    assert r.status_code == 200
    return r.json()


def _roster_cab(entry):
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT cab_id FROM roster_entries WHERE company_id='acme' AND entry_id=%s", (entry,))
        return cur.fetchone()[0]


def _request_rows(employee):
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT id, status FROM shift_change_requests WHERE company_id='acme' AND employee_id=%s", (employee,))
        return cur.fetchall()


def test_cutoff_offer_confirm_pending_says_unchanged(chat):
    client, employee, d = chat
    first = _say(client, employee, "move my drop to 21:00")
    assert "120" in first["reply"] and "transport manager" in first["reply"]
    assert "not been changed" in first["reply"]
    assert _request_rows(employee) == []  # offering does not create anything

    second = _say(client, employee, "yes, send it to my manager")
    assert second["status"] == "pending_approval"
    assert second["request_id"]
    assert "Nothing has changed yet" in second["reply"]
    assert "applied" not in second["reply"].lower()
    assert _roster_cab(d["approval_entry"]) == "C_AC_D01"
    assert [str(r[0]) for r in _request_rows(employee)] == [second["request_id"]]
    assert _request_rows(employee)[0][1] == "pending_approval"

    # Employee role cannot decide; other-company manager gets 404 too.
    rid = second["request_id"]
    assert client.post(f"/manager/requests/{rid}/decision", json={"decision": "approve"}, headers=_h("acme", employee)).status_code == 404
    assert client.post(f"/manager/requests/{rid}/decision", json={"decision": "approve"}, headers=_h("zenith", d["zenith_manager"], "manager")).status_code == 404
    assert _request_rows(employee)[0][1] == "pending_approval"

    # Same confirmation again does not create a second request (offer consumed).
    third = _say(client, employee, "yes")
    assert "no change waiting" in third["reply"].lower()
    assert len(_request_rows(employee)) == 1


def test_yes_without_offer_does_nothing(chat):
    client, employee, d = chat
    reply = _say(client, employee, "yes, send it to my manager")
    assert "no change waiting" in reply["reply"].lower()
    assert _request_rows(employee) == []


def test_model_cannot_escalate_on_its_own(chat):
    client, employee, d = chat
    reply = _say(client, employee, "force it through")
    assert _request_rows(employee) == []
    assert _roster_cab(d["approval_entry"]) == "C_AC_D01"
    assert "not been changed" in reply["reply"] or "no change waiting" in reply["reply"].lower()


def test_decline_clears_offer(chat):
    client, employee, d = chat
    _say(client, employee, "move my drop to 21:00")
    declined = _say(client, employee, "no thanks")
    assert "won't send" in declined["reply"]
    after = _say(client, employee, "yes")
    assert "no change waiting" in after["reply"].lower()
    assert _request_rows(employee) == []


def test_service_window_breach_never_offers_escalation(chat):
    client, employee, d = chat
    reply = _say(client, employee, "move my drop to 23:30")
    assert "service window" in reply["reply"]
    assert "manager" not in reply["reply"].lower()
    assert "not been changed" in reply["reply"]


def test_booking_offer_confirmed_with_yes_applies(stage3_data):
    from app.main import app
    import app.graph.chat as chat_mod
    from app.llm.fake import FakeLLM
    d = stage3_data
    employee, entry = d["employees"][0], d["entries"][0]  # 22:00 drop: R1/R2 fine
    with TestClient(app, raise_server_exceptions=False) as client:
        chat_mod._llm_singleton = FakeLLM(plans=_plans(entry))
        offer = _say(client, employee, "move my drop to 21:00")
        assert offer["status"] == "awaiting_input" and "Shall I book it?" in offer["reply"]
        assert _roster_cab(entry) == "C_AC_D01"
        done = _say(client, employee, "yes")
        assert done["status"] == "ok" and "applied" in done["reply"]
        assert _roster_cab(entry) == offer["options"][0]
        again = _say(client, employee, "yes")  # offer consumed
        assert "no change waiting" in again["reply"].lower()
