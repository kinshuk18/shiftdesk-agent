"""
tests/test_roster_date.py — verify planner emitting wrong or garbage date still returns roster for Clock's today.
"""

from app.auth import Principal
from app.clock import clock
from app.llm.fake import FakeLLM
from app.graph.chat import build_chat_graph
import app.graph.chat as _chat_mod
from app.tools import get_my_roster


def test_planner_emits_wrong_or_garbage_date_returns_roster_for_clock_today():
    """
    SPEC requirement:
    A planner that emits a wrong or garbage date still returns the roster for Clock's today.
    """
    garbage_plan = {
        "intent": "get_roster",
        "direction": None,
        "target_time": None,
        "relative_hint": None,
        "chosen_option": None,
        "ambiguity": "none",
        "tool_calls": [
            {"tool": "get_my_roster", "args": {"date": "invalid-garbage-date"}}
        ],
    }

    old_singleton = _chat_mod._llm_singleton
    _chat_mod._llm_singleton = FakeLLM(plans={"show my roster": garbage_plan})
    try:
        graph = build_chat_graph(checkpointer=None)
        state = {
            "current_message": "show my roster",
            "principal": {
                "company_id": "acme",
                "employee_id": "E101",
                "role": "employee",
            },
            "trace_id": "test-trace-roster-date",
            "llm_calls_used": 0,
            "tool_results": [],
            "tool_errors": [],
            "options": [],
            "history": [],
            "pending_slots": {},
            "awaiting": None,
        }
        result = graph.invoke(state, config={"configurable": {"thread_id": "test-thread-roster-date"}})

        roster_results = [r for r in result.get("tool_results", []) if r["tool"] == "get_my_roster"]
        assert len(roster_results) == 1
        assert roster_results[0]["ok"] is True

        today_iso = clock.now().date().isoformat()
        entries = roster_results[0]["data"]["entries"]
        assert len(entries) > 0
        for entry in entries:
            assert entry["service_date"] == today_iso

        assert "Here is your roster for today:" in result.get("reply", "")
        assert result.get("status") == "ok"

        # Direct verification with wrong/garbage date
        direct_result = get_my_roster(
            Principal(company_id="acme", employee_id="E101", role="employee"),
            date="wrong-date",
        )
        assert direct_result.ok is True
        assert any(e["service_date"] == today_iso for e in direct_result.data["entries"])
    finally:
        _chat_mod._llm_singleton = old_singleton
