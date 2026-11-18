"""task_runs (the progress record of long-running commands) against a real Postgres."""
import pytest

from vendorkb import task_runs
from vendorkb.db import connect, init_schema
from vendorkb.task_runs import HEARTBEAT_S, progress, track

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.fixture(autouse=True)
def _no_cluster(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)


async def _rows(dsn, where="true"):
    conn = await connect(dsn)
    try:
        await init_schema(conn, embed_dim=1024)
        cur = await conn.execute(
            "SELECT run_id, parent_id, kind, state, origin, done, total, spent_usd, details,"
            f" ended_at FROM task_runs WHERE {where} ORDER BY started_at")
        return await cur.fetchall()
    finally:
        await conn.close()


async def _reset(dsn):
    conn = await connect(dsn)
    try:
        await init_schema(conn, embed_dim=1024)
        await conn.execute("DELETE FROM task_runs")
    finally:
        await conn.close()


async def test_track_records_success(vkb_dsn):
    await _reset(vkb_dsn)
    async with track(vkb_dsn, "map"):
        await progress(5, 10, 0.25, force=True)
    (row,) = await _rows(vkb_dsn)
    assert row[2:8] == ("map", "succeeded", "manual", 5, 10, 0.25)
    assert row[9] is not None


async def test_track_records_failure_and_reraises(vkb_dsn):
    await _reset(vkb_dsn)
    with pytest.raises(RuntimeError):
        async with track(vkb_dsn, "map"):
            raise RuntimeError("boom")
    (row,) = await _rows(vkb_dsn)
    assert row[3] == "failed"
    assert row[9] is not None


async def test_nested_track_sets_parent(vkb_dsn):
    await _reset(vkb_dsn)
    async with track(vkb_dsn, "refresh") as outer:
        async with track(vkb_dsn, "map"):
            pass
    rows = await _rows(vkb_dsn)
    inner = next(r for r in rows if r[2] == "map")
    assert str(inner[1]) == outer.run_id
    assert next(r for r in rows if r[2] == "refresh")[1] is None


async def test_explicit_finish_wins(vkb_dsn):
    await _reset(vkb_dsn)
    async with track(vkb_dsn, "support") as run:
        await run.finish("failed", {"errored": 3})
    (row,) = await _rows(vkb_dsn)
    assert row[3] == "failed"
    assert row[8] == {"errored": 3}


async def test_progress_is_throttled(vkb_dsn, monkeypatch):
    await _reset(vkb_dsn)
    now = [1000.0]
    monkeypatch.setattr(task_runs, "_clock", lambda: now[0])
    async with track(vkb_dsn, "extract") as run:
        await run.progress(1)
        now[0] += HEARTBEAT_S / 2
        await run.progress(2)
        assert (await _rows(vkb_dsn))[0][5] == 1
        now[0] += HEARTBEAT_S
        await run.progress(3)
        assert (await _rows(vkb_dsn))[0][5] == 3
        await run.progress(4)
        assert (await _rows(vkb_dsn))[0][5] == 3
        await run.progress(5, force=True)
        assert (await _rows(vkb_dsn))[0][5] == 5


async def test_unreachable_database_never_fails_the_task():
    async with track("postgresql://nobody@127.0.0.1:1/x", "sync") as run:
        await progress(1, 2)
        await run.progress(2, 2, force=True)
        await run.finish("succeeded")
        value = 42
    assert value == 42


async def test_origin_cluster(vkb_dsn, monkeypatch):
    await _reset(vkb_dsn)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    async with track(vkb_dsn, "sync"):
        pass
    assert (await _rows(vkb_dsn))[0][4] == "cluster"


async def test_old_finished_rows_pruned(vkb_dsn):
    await _reset(vkb_dsn)
    conn = await connect(vkb_dsn)
    try:
        await conn.execute(
            "INSERT INTO task_runs (run_id, kind, state, origin, started_at, ended_at) VALUES"
            " (gen_random_uuid(), 'sync', 'succeeded', 'manual',"
            "  now() - interval '92 days', now() - interval '91 days'),"
            " (gen_random_uuid(), 'extract', 'running', 'manual', now() - interval '91 days', NULL)")
    finally:
        await conn.close()
    async with track(vkb_dsn, "map"):
        pass
    assert sorted(r[2] for r in await _rows(vkb_dsn)) == ["extract", "map"]


async def test_recording_needs_no_database():
    with task_runs.recording() as rec:
        await progress(1, 2, 0.5)
    assert rec.calls == [(1, 2, 0.5)]


async def test_connection_drop_mid_run_recovers(vkb_dsn):
    await _reset(vkb_dsn)
    async with track(vkb_dsn, "map") as run:
        await run.conn.close()
        await progress(7, 9, force=True)
    (row,) = await _rows(vkb_dsn)
    assert (row[3], row[5]) == ("succeeded", 7)


async def test_heartbeat_updates_heartbeat_at(vkb_dsn, monkeypatch):
    import asyncio
    await _reset(vkb_dsn)
    monkeypatch.setattr(task_runs, "HEARTBEAT_S", 0.2)
    async with track(vkb_dsn, "map"):
        first = (await _beat_at(vkb_dsn))
        await asyncio.sleep(0.7)
        assert (await _beat_at(vkb_dsn)) > first


async def _beat_at(dsn):
    conn = await connect(dsn)
    try:
        cur = await conn.execute("SELECT heartbeat_at FROM task_runs")
        return (await cur.fetchone())[0]
    finally:
        await conn.close()
