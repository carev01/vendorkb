"""A progress record for long-running commands (the ``task_runs`` table).

``track(dsn, kind)`` wraps a command: it inserts a ``running`` row, keeps its heartbeat alive
in the background, takes progress from ``progress()`` calls (throttled to one write per
``HEARTBEAT_S``) and ends the row ``succeeded`` or ``failed``. Each new run also ends
``failed`` (reason ``stalled``) every running row whose heartbeat stopped ``DEAD_AFTER_S``
ago: a killed process never ends its own. Recording never fails the task: an unreachable
database or a failed write is logged and swallowed. Runs nest: a ``track`` inside another one
records the outer run as its parent.
"""
import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextvars import ContextVar
from typing import Literal

import psycopg

log = logging.getLogger(__name__)

KINDS = ("sync", "refresh", "extract", "map", "support", "capabilities", "propose")
HEARTBEAT_S = 30.0
STALLED_AFTER_S = 300  # a running row whose heartbeat is older than this counts as stalled
DEAD_AFTER_S = 3600  # a running row silent this long is dead: the next run ends it failed
RETENTION_DAYS = 90
CONNECT_TIMEOUT_S = 5
WRITE_TIMEOUT_S = 5.0  # bound on every write and reconnect: recording never slows the task

_clock = time.monotonic
_current: ContextVar["TaskRun | Recorder | None"] = ContextVar("task_run", default=None)


class TaskRun:
    """One run. With no connection (the database was unreachable) every method is a no-op."""

    def __init__(self, conn: psycopg.AsyncConnection | None = None, run_id: str | None = None,
                 dsn: str | None = None):
        self.conn = conn
        self.dsn = dsn  # set once the row exists: a broken connection reconnects to it
        self.run_id = run_id or str(uuid.uuid4())
        self._last_write = float("-inf")
        self._finished = False
        self._latest: tuple[int, int | None, float | None] | None = None  # last reported

    async def _drop_conn(self) -> None:
        conn, self.conn = self.conn, None
        if conn is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(conn.close(), WRITE_TIMEOUT_S)

    async def _write(self, query: str, params: tuple) -> None:
        if self.dsn is None:  # never recorded (database unreachable): nothing to write to
            return
        try:
            await asyncio.wait_for(self._execute(query, params), WRITE_TIMEOUT_S)
        except Exception as exc:  # recording must never fail the task (TimeoutError included)
            log.warning("task_runs write failed: %s", exc)
            await self._drop_conn()  # the next write reconnects, one attempt

    async def _execute(self, query: str, params: tuple) -> None:
        if self.conn is None or self.conn.closed:
            assert self.dsn is not None
            self.conn = await psycopg.AsyncConnection.connect(
                self.dsn, autocommit=True, connect_timeout=CONNECT_TIMEOUT_S)
        await self.conn.execute(query, params)

    async def progress(self, done: int, total: int | None = None,
                       spent_usd: float | None = None, *, force: bool = False) -> None:
        # done and spent only grow within a run (a retry or a new round must not lower them);
        # total may change freely. Kept even when throttled: finish() writes them.
        prev_done, prev_total, prev_spent = self._latest or (0, None, None)
        done = max(done, prev_done)
        if spent_usd is None or (prev_spent is not None and prev_spent > spent_usd):
            spent_usd = prev_spent
        total = total if total is not None else prev_total
        self._latest = (done, total, spent_usd)
        now = _clock()
        if not force and now - self._last_write < HEARTBEAT_S:
            return
        self._last_write = now
        await self._write(
            "UPDATE task_runs SET done = %s, total = COALESCE(%s, total),"
            " spent_usd = COALESCE(%s, spent_usd), heartbeat_at = now() WHERE run_id = %s",
            (done, total, spent_usd, self.run_id))

    async def finish(self, state: Literal["succeeded", "failed"],
                     details: dict | None = None) -> None:
        self._finished = True
        done, total, spent = self._latest if self._latest is not None else (None, None, None)
        await self._write(
            "UPDATE task_runs SET state = %s, details = %s::jsonb, ended_at = now(),"
            " heartbeat_at = now(), done = COALESCE(%s, done), total = COALESCE(%s, total),"
            " spent_usd = COALESCE(%s, spent_usd) WHERE run_id = %s",
            (state, json.dumps(details or {}), done, total, spent, self.run_id))

    async def _beat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            await self._write("UPDATE task_runs SET heartbeat_at = now() WHERE run_id = %s",
                              (self.run_id,))


class Recorder:
    """A stand-in current run for tests: it keeps the ``progress()`` calls and needs no database."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None, float | None]] = []

    async def progress(self, done: int, total: int | None = None,
                       spent_usd: float | None = None, *, force: bool = False) -> None:
        self.calls.append((done, total, spent_usd))


async def progress(done: int, total: int | None = None, spent_usd: float | None = None,
                   *, force: bool = False) -> None:
    """Report progress to the current run, if there is one."""
    run = _current.get()
    if run is not None:
        await run.progress(done, total, spent_usd, force=force)


@contextlib.contextmanager
def recording() -> Iterator[Recorder]:
    rec = Recorder()
    token = _current.set(rec)
    try:
        yield rec
    finally:
        _current.reset(token)


@contextlib.asynccontextmanager
async def track(dsn: str, kind: str, *, stage: str | None = None) -> AsyncIterator[TaskRun]:
    parent = _current.get()
    # dsn, not conn: a failed write drops the connection until the next one reconnects
    parent_id = parent.run_id if isinstance(parent, TaskRun) and parent.dsn else None
    run = await _start(dsn, kind, stage, parent_id)
    token = _current.set(run)
    beat = asyncio.create_task(run._beat()) if run.dsn is not None else None
    try:
        yield run
    except BaseException:
        if not run._finished:
            # bounded by the write timeout; shielded so a second cancellation neither
            # interrupts it nor replaces the exception being raised
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(run.finish("failed"))
        raise
    else:
        if not run._finished:
            await run.finish("succeeded")
    finally:
        _current.reset(token)
        if beat is not None:
            beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat
        await run._drop_conn()


async def _start(dsn: str, kind: str, stage: str | None, parent_id: str | None) -> TaskRun:
    run = TaskRun()
    try:
        conn = await psycopg.AsyncConnection.connect(
            dsn, autocommit=True, connect_timeout=CONNECT_TIMEOUT_S)
    except Exception as exc:
        log.warning("task_runs: database unreachable, not recording: %s", exc)
        return run
    run.conn = conn
    run.dsn = dsn
    origin = "cluster" if os.environ.get("KUBERNETES_SERVICE_HOST") else "manual"
    await run._write(
        "INSERT INTO task_runs (run_id, parent_id, kind, stage, state, origin)"
        " VALUES (%s, %s, %s, %s, 'running', %s)",
        (run.run_id, parent_id, kind, stage, origin))
    # a run killed outright (SIGKILL, a lost node) never ran its finish(): end it failed at
    # its last heartbeat, so it reads as a failure and not as stalled forever
    await run._write(
        "UPDATE task_runs SET state = 'failed', ended_at = heartbeat_at,"
        " details = details || '{\"reason\": \"stalled\"}'::jsonb"
        " WHERE state = 'running' AND heartbeat_at < now() - make_interval(secs => %s)",
        (DEAD_AFTER_S,))
    await run._write(
        "DELETE FROM task_runs WHERE state <> 'running'"
        " AND ended_at < now() - make_interval(days => %s)", (RETENTION_DAYS,))
    return run
