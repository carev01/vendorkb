"""task_runs (the progress record of long-running commands) against a real Postgres."""
import pytest

from vendorkb import task_runs
from vendorkb.claims import refresh as refresh_mod
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


async def _refresh_run(dsn, conn, **kwargs):
    from tests.integration import test_claims_refresh as t

    return await refresh_mod.refresh(
        conn, settings=t._settings(vendorkb_dsn=dsn), extract_client=t.FakeExtract(),
        map_client=t.NoCalls(), support_client=t.NoCalls(), embedder=t.NoCalls(),
        ready=t.Ready(), settle_minutes=60, extract_limit=100, map_max_usd=1.0, now=t.NOW,
        **kwargs)


async def test_refresh_stages_are_children(vkb_dsn):
    from tests.integration import test_claims_refresh as t

    conn = await connect(vkb_dsn)
    try:
        await t._fresh(conn)
        await conn.execute("DELETE FROM task_runs")
        await t._chunk(conn)
        report = await _refresh_run(vkb_dsn, conn)
        assert report.due
    finally:
        await conn.close()
    rows = await _rows(vkb_dsn)
    (top,) = [r for r in rows if r[2] == "refresh"]
    assert top[3] == "succeeded" and top[1] is None
    stages = {r[2]: r for r in rows if r[2] != "refresh"}
    assert set(stages) == {"extract", "map", "support", "capabilities"}
    assert all(r[1] == top[0] and r[3] == "succeeded" for r in stages.values())


async def test_refresh_not_due_records_reason(vkb_dsn):
    from tests.integration import test_claims_refresh as t

    conn = await connect(vkb_dsn)
    try:
        await t._fresh(conn)
        await conn.execute("DELETE FROM task_runs")
        report = await _refresh_run(vkb_dsn, conn)
        assert not report.due
    finally:
        await conn.close()
    (row,) = await _rows(vkb_dsn)
    assert row[2:4] == ("refresh", "succeeded")
    assert row[8]["reason"] == report.reason == "no new chunks"


async def test_cli_exit_code_sets_state(vkb_dsn, monkeypatch):
    import asyncio

    from vendorkb import cli
    from vendorkb.claims import extract

    await _reset(vkb_dsn)
    monkeypatch.setenv("VENDORKB_DSN", vkb_dsn)

    async def fake_run(conn, ids, **kwargs):
        return extract.ExtractReport(chunks=3, errored=2)

    monkeypatch.setattr(extract, "run", fake_run)
    monkeypatch.setattr(cli, "_extract_client", lambda settings: object())
    code = await asyncio.to_thread(cli.main, ["extract", "--unextracted"])
    assert code == 1
    (row,) = await _rows(vkb_dsn)
    assert row[2:4] == ("extract", "failed")
    assert row[8]["errored"] == 2 and row[8]["chunks"] == 3


async def test_finish_writes_the_last_progress_even_when_throttled(vkb_dsn):
    await _reset(vkb_dsn)
    async with track(vkb_dsn, "map"):
        await progress(1, 10, 0.1, force=True)
        await progress(7, 10, 0.7)  # inside the throttle window: not written by itself
        await progress(10, 10, 1.0)
    (row,) = await _rows(vkb_dsn)
    assert row[3] == "succeeded"
    assert row[5:8] == (10, 10, 1.0)


async def test_done_and_spend_are_high_water_marks(vkb_dsn):
    await _reset(vkb_dsn)
    async with track(vkb_dsn, "map"):
        await progress(5, 10, 0.5, force=True)
        await progress(2, 8, 0.2, force=True)  # lower: done and spend keep their maxima
        await progress(1, 9, None)  # throttled; finish() still writes the maxima
    (row,) = await _rows(vkb_dsn)
    assert row[5:8] == (5, 9, 0.5)  # the total may change freely


async def test_dead_running_rows_end_failed_when_a_run_starts(vkb_dsn):
    await _reset(vkb_dsn)
    conn = await connect(vkb_dsn)
    try:
        await conn.execute(
            "INSERT INTO task_runs (run_id, kind, state, origin, details, started_at,"
            " heartbeat_at) VALUES"
            " (gen_random_uuid(), 'sync', 'running', 'cluster', '{\"pages\": 3}',"
            "  now() - interval '3 hours', now() - interval '2 hours'),"
            " (gen_random_uuid(), 'extract', 'running', 'manual', '{}',"
            "  now() - interval '3 hours', now() - interval '1 minute')")
    finally:
        await conn.close()
    async with track(vkb_dsn, "map"):
        pass
    conn = await connect(vkb_dsn)
    try:
        cur = await conn.execute(
            "SELECT kind, state, details, ended_at = heartbeat_at FROM task_runs ORDER BY kind")
        rows = await cur.fetchall()
    finally:
        await conn.close()
    assert rows == [("extract", "running", {}, None),  # no end yet
                    ("map", "succeeded", {}, True),
                    ("sync", "failed", {"pages": 3, "reason": "stalled"}, True)]


async def test_parent_link_survives_a_dropped_parent_connection(vkb_dsn):
    await _reset(vkb_dsn)
    async with track(vkb_dsn, "refresh") as outer:
        await outer._drop_conn()  # a failed write leaves conn None until the next write
        async with track(vkb_dsn, "map"):
            pass
    rows = await _rows(vkb_dsn)
    inner = next(r for r in rows if r[2] == "map")
    assert str(inner[1]) == outer.run_id


async def _stages(dsn):
    conn = await connect(dsn)
    try:
        cur = await conn.execute("SELECT kind, stage, state FROM task_runs ORDER BY started_at")
        return await cur.fetchall()
    finally:
        await conn.close()


async def test_refresh_stages_end_failed_on_errors(vkb_dsn, monkeypatch):
    from tests.integration import test_claims_refresh as t
    from vendorkb.claims import extract, support
    from vendorkb.taxonomy import mapper

    async def bad_extract(conn, ids, **kwargs):
        return extract.ExtractReport(chunks=1, errored=1)

    async def bad_map(conn, **kwargs):
        return mapper.MapReport(claims=1, llm_invalid=1)

    async def bad_support(conn, **kwargs):
        report = support.SupportReport()
        report.check.invalid = 1
        return report

    monkeypatch.setattr(extract, "run", bad_extract)
    monkeypatch.setattr(mapper, "map_claims", bad_map)
    monkeypatch.setattr(support, "run", bad_support)
    conn = await connect(vkb_dsn)
    try:
        await t._fresh(conn)
        await conn.execute("DELETE FROM task_runs")
        await t._chunk(conn)
        report = await _refresh_run(vkb_dsn, conn)
        assert refresh_mod.failed(report)
    finally:
        await conn.close()
    states = {kind: state for kind, _, state in await _stages(vkb_dsn)}
    assert {k: states[k] for k in ("refresh", "extract", "map", "support")} == {
        "refresh": "failed", "extract": "failed", "map": "failed", "support": "failed"}
    rows = await _rows(vkb_dsn, "kind = 'extract'")
    assert rows[0][8]["errored"] == 1


async def test_collect_refresh_is_staged_collect(vkb_dsn):
    from tests.integration import test_claims_refresh as t

    conn = await connect(vkb_dsn)
    try:
        await t._fresh(conn)
        await conn.execute("DELETE FROM task_runs")
        await _refresh_run(vkb_dsn, conn, extract_new=False)
        await _refresh_run(vkb_dsn, conn)
    finally:
        await conn.close()
    assert [r[:2] for r in await _stages(vkb_dsn)] == [("refresh", "collect"), ("refresh", None)]


async def test_stats_creates_the_schema(vkb_dsn, monkeypatch):
    import asyncio

    from vendorkb import cli

    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=1024)
        await conn.execute("DROP TABLE task_runs CASCADE")
    finally:
        await conn.close()
    monkeypatch.setenv("VENDORKB_DSN", vkb_dsn)
    monkeypatch.setenv("EMBED_DIM", "1024")
    assert await asyncio.to_thread(cli.main, ["stats"]) == 0
    conn = await connect(vkb_dsn)
    try:
        cur = await conn.execute("SELECT to_regclass('task_runs') IS NOT NULL")
        assert (await cur.fetchone())[0]
    finally:
        await conn.close()
