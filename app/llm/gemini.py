"""
app/llm/gemini.py — Gemini provider using httpx with 3 s timeout.

SPEC §3: model name from env GEMINI_MODEL.
TODO: check the current free-tier model name at
      https://ai.google.dev/gemini-api/docs/models/gemini
      before using in production — the env var is the override mechanism.

Raises:
    LLMTimeoutError — httpx.TimeoutException
    LLMQuotaError   — HTTP 429
    LLMError        — any other HTTP or parse error
"""

from __future__ import annotations

import json
import os

import httpx

from app.llm.base import LLMClient, LLMError, LLMQuotaError, LLMTimeoutError

_TIMEOUT_SECONDS = 3.0
_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiLLM:
    """
    Gemini generateContent client.

    Reads GEMINI_API_KEY and GEMINI_MODEL from environment.
    TODO: verify the current free-tier model name at
          https://ai.google.dev/gemini-api/docs/models/gemini
    """

    def __init__(self) -> None:
        self._api_key: str = os.environ.get("GEMINI_API_KEY", "")
        self._model: str = os.environ.get("GEMINI_MODEL", "")
        if not self._model:
            raise LLMError(
                "GEMINI_MODEL env var is not set. "
                "TODO: check https://ai.google.dev/gemini-api/docs/models/gemini "
                "for the current free-tier model."
            )
        if not self._api_key:
            raise LLMError("GEMINI_API_KEY env var is not set.")

    def chat(self, messages: list[dict]) -> str:
        """
        Call Gemini generateContent and return the text.
        Forces JSON output via responseMimeType.

        Converts the OpenAI-style messages list to Gemini's contents format:
          - "system" role → systemInstruction
          - "user" / "assistant" → contents with role "user" / "model"
        """
        system_text = ""
        contents = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0,
            },
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        url = (
            f"{_GEMINI_API_BASE}/{self._model}:generateContent"
            f"?key={self._api_key}"
        )
        try:
            resp = httpx.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=_TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Gemini request timed out after {_TIMEOUT_SECONDS}s"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Gemini network error: {exc}") from exc

        if resp.status_code == 429:
            raise LLMQuotaError(f"Gemini rate-limit / quota error: {resp.text[:200]}")
        if resp.status_code != 200:
            raise LLMError(f"Gemini returned HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            body = resp.json()
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Gemini response parse error: {exc}") from exc
