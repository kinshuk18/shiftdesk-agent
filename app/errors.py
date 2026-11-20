"""
app/errors.py — Stage-2 error types.

Defines custom exceptions used by the graph and endpoint.
No tracebacks or internal details escape to the caller.
"""

from __future__ import annotations


class GraphTimeoutError(Exception):
    """Raised when the chat graph does not finish within the hard 4.5-second deadline."""


class UnexpectedGraphError(Exception):
    """Raised when the chat graph raises an unexpected exception."""
