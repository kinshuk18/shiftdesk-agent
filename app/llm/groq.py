"""
app/llm/groq.py — Groq provider using httpx with 3 s timeout.

SPEC §3: model name from env GROQ_MODEL.
TODO: check the current free-tier model name on https://console.groq.com/docs/models
      before using in production — the env var is the override mechanism.

Raises:
    LLMTimeoutError — httpx.TimeoutException
    LLMQuotaError   — HTTP 429
    LLMError        — any other HTTP or parse error
"""

from __future__ import annotations

import os

import httpx

from app.llm.base import LLMClient, LLMError, LLMQuotaError, LLMTimeoutError

_TIMEOUT_SECONDS = 3.0
_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqLLM:
    """
    Groq chat completion client.

    Reads GROQ_API_KEY and GROQ_MODEL from environment.
    TODO: verify the current free-tier model name at https://console.groq.com/docs/models
    """

    def __init__(self) -> None:
        self._api_key: str = os.environ.get("GROQ_API_KEY", "")
        self._model: str = os.environ.get("GROQ_MODEL", "")
        if not self._model:
            raise LLMError(
                "GROQ_MODEL env var is not set. "
                "TODO: check https://console.groq.com/docs/models for the current free-tier model."
            )
        if not self._api_key:
            raise LLMError("GROQ_API_KEY env var is not set.")

    def chat(self, messages: list[dict]) -> str:
        """
        Call Groq chat completions and return the assistant message text.
        Forces JSON output via response_format.
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        try:
            resp = httpx.post(
                _GROQ_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=_TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Groq request timed out after {_TIMEOUT_SECONDS}s") from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Groq network error: {exc}") from exc

        if resp.status_code == 429:
            raise LLMQuotaError(f"Groq rate-limit / quota error: {resp.text[:200]}")
        if resp.status_code != 200:
            raise LLMError(f"Groq returned HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            body = resp.json()
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Groq response parse error: {exc}") from exc
