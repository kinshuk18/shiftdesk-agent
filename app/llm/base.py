"""
app/llm/base.py — LLMClient Protocol and LLMBudget counter.

SPEC §3: <= 4 LLM calls per message, enforced in code, not prompt.
Every provider attempt (including retries) increments the budget.
When exhausted: raise BudgetExhausted — callers degrade gracefully.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class BudgetExhausted(Exception):
    """Raised when LLMBudget.max calls have been consumed."""


class LLMBudget:
    """
    Per-message call counter.  Shared across the graph so every node
    that calls the LLM picks it up from state.

    Usage:
        budget = LLMBudget(max=4)
        budget.charge()      # increments; raises BudgetExhausted if at limit
        budget.used          # int: how many calls consumed so far
    """

    def __init__(self, max: int = 4) -> None:
        self._max = max
        self._used = 0

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return self._max - self._used

    def charge(self) -> None:
        """
        Mark one LLM call as consumed.
        Raises BudgetExhausted before making the call when the budget is exhausted.
        """
        if self._used >= self._max:
            raise BudgetExhausted(
                f"LLM budget exhausted: {self._used}/{self._max} calls used."
            )
        self._used += 1

    def reset(self) -> None:
        """Reset the counter (useful between messages in tests)."""
        self._used = 0


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol that every LLM provider must satisfy.

    chat(messages) -> str
      messages: list of {"role": "system"|"user"|"assistant", "content": str}
      Returns the assistant's reply text (raw, before JSON parsing).
      Raises:
        BudgetExhausted — budget was full before this call
        LLMTimeoutError — provider did not respond within timeout
        LLMQuotaError   — provider returned rate-limit / quota error
        LLMError        — any other provider error
    """

    def chat(self, messages: list[dict]) -> str:
        ...


class LLMError(Exception):
    """Generic LLM provider error."""


class LLMTimeoutError(LLMError):
    """Provider did not respond within the timeout window."""


class LLMQuotaError(LLMError):
    """Provider returned a rate-limit or quota error."""
