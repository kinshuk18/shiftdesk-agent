"""
app/db.py — database connection pool and transaction helper.

Uses psycopg 3 (psycopg[binary,pool]).
One ConnectionPool is created at startup and shared across requests.
"""

import threading

import psycopg
from psycopg_pool import ConnectionPool
from contextlib import contextmanager
from typing import Generator

from app.config import settings

# Pool is created lazily on first call to get_pool().
_pool: ConnectionPool | None = None
# Guards lazy creation: concurrent first calls must not each build a pool.
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """Return the shared connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=settings.database_url,
                    min_size=2,
                    max_size=10,
                    # Open connections eagerly so the first request is not slow.
                    open=True,
                )
    return _pool


@contextmanager
def transaction() -> Generator[psycopg.Connection, None, None]:
    """
    Context manager that yields a connection with an open transaction.
    Commits on clean exit, rolls back on exception.

    Usage:
        with transaction() as conn:
            conn.execute(...)
    """
    with get_pool().connection() as conn:
        with conn.transaction():
            yield conn


def close_pool() -> None:
    """Call on application shutdown to drain the pool."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None
