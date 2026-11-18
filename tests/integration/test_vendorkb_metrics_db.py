"""The pipeline snapshot (metrics_db) against a real Postgres."""
import uuid

import pytest
from prometheus_client import CollectorRegistry

from vendorkb.db import connect, init_schema
from vendorkb.metrics_db import PipelineCollector, Snapshot, snapshot

pytestmark = pytest.mark.asyncio(loop_scope="module")

TABLES = ("task_runs", "capabilities", "raw_claims", "extractions", "chunks", "documents",
          "llm_batch_items", "llm_batches", "extract_batches", "propose_runs", "claim_support",
          "taxonomy_meta", "sync_state")


async def _fresh(dsn):
    conn = await connect(dsn)
    await init_schema(conn, embed_dim=1024)
    for t in TABLES:
        await conn.execute(f"DELETE FROM {t}")
    return conn


async def _registry(conn):
    collector = PipelineCollector()
    collector.update(await snapshot(conn), True)
    reg = CollectorRegistry()
    reg.register(collector)
    return reg, collector


async def _run(conn, kind, *, age_s=0, done=0, total=None, spent=0.0, stage=None):
    await conn.execute(
        "INSERT INTO task_runs (run_id, kind, stage, state, origin, done, total, spent_usd,"
        " heartbeat_at) VALUES (%s, %s, %s, 'running', 'manual', %s, %s, %s,"
        " now() - make_interval(secs => %s))",
        (str(uuid.uuid4()), kind, stage, done, total, spent, age_s))


async def _chunk(conn, doc_id, n):
    cid = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
        " embedding) VALUES (%s, %s, ARRAY['s'], %s, 'text', 1, array_fill(0, ARRAY[1024])::vector)",
        (cid, doc_id, n))
    return cid


async def _doc(conn):
    did = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO documents (doc_id, vendor, content_hash, fetched_at)"
        " VALUES (%s, 'V', 'h', now())", (did,))
    return did


async def test_running_and_stalled(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    await _run(conn, "map")
    await _run(conn, "support", age_s=600)
    reg, _ = await _registry(conn)
    assert reg.get_sample_value("vendorkb_task_running", {
        "kind": "map", "stage": "", "run": reg_run(reg, "map"), "origin": "manual"}) == 1
    assert reg.get_sample_value("vendorkb_task_stalled", {
        "kind": "support", "stage": "", "run": reg_run(reg, "support"), "origin": "manual"}) == 1
    kinds = {s.labels["kind"] for m in reg.collect() if m.name == "vendorkb_task_running"
             for s in m.samples}
    assert kinds == {"map"}
    await conn.close()


def reg_run(reg, kind):
    for m in reg.collect():
        if m.name == "vendorkb_task_done":
            for s in m.samples:
                if s.labels["kind"] == kind:
                    return s.labels["run"]
    raise AssertionError(kind)


async def test_two_runs_same_kind(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    await _run(conn, "map", done=3, total=10, spent=0.5)
    await _run(conn, "map", done=7)
    reg, _ = await _registry(conn)
    done = sorted(s.value for m in reg.collect() if m.name == "vendorkb_task_done"
                  for s in m.samples)
    runs = {s.labels["run"] for m in reg.collect() if m.name == "vendorkb_task_done"
            for s in m.samples}
    assert done == [3, 7] and len(runs) == 2
    totals = [s for m in reg.collect() if m.name == "vendorkb_task_total" for s in m.samples]
    assert [s.value for s in totals] == [10]  # only the run with a total
    await conn.close()


async def test_backlog_counts(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    did = await _doc(conn)
    for i in range(3):
        await _chunk(conn, did, i)
    cid = await _chunk(conn, did, 9)
    await conn.execute(
        "UPDATE chunks SET superseded_at = now() WHERE chunk_id = %s", (cid,))
    c2 = await _chunk(conn, did, 10)
    eid = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status) VALUES (%s, %s, 'm', 'claims-v3.1', 'done')", (eid, c2))
    await conn.execute(
        "INSERT INTO taxonomy_meta (key, value) VALUES ('taxonomy_version', 'v9')")
    for mv in ("old", "old", "v9"):
        await conn.execute(
            "INSERT INTO raw_claims (claim_id, extraction_id, chunk_id, vendor,"
            " capability_raw, support, evidence_quote, gate, extractor_model, prompt_version,"
            " mapped_version) VALUES (%s, %s, %s, 'V', 'c', 'supported', 'q', 'exact', 'm',"
            " 'claims-v3.1', %s)", (str(uuid.uuid4()), eid, c2, mv))
    reg, _ = await _registry(conn)
    assert reg.get_sample_value("vendorkb_backlog", {"kind": "extract"}) == 3
    assert reg.get_sample_value("vendorkb_backlog", {"kind": "map"}) == 2
    assert reg.get_sample_value("vendorkb_backlog", {"kind": "support"}) is not None
    assert reg.get_sample_value("vendorkb_corpus", {"what": "chunks"}) == 4
    assert reg.get_sample_value("vendorkb_corpus", {"what": "claims"}) == 3
    assert reg.get_sample_value("vendorkb_corpus", {"what": "documents"}) == 1
    assert reg.get_sample_value("vendorkb_taxonomy_info", {"version": "v9"}) == 1
    await conn.close()


async def test_batches(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    await conn.execute(
        "INSERT INTO llm_batches (batch_id, kind, model, custom_ids, status, submitted_at)"
        " VALUES ('b1', 'map-target', 'm', ARRAY['x'], 'submitted', now() - interval '2 hours')")
    await conn.execute(
        "INSERT INTO llm_batch_items (kind, custom_id, status) VALUES ('map-target', 'x',"
        " 'in_flight')")
    reg, _ = await _registry(conn)
    assert reg.get_sample_value("vendorkb_batches_open", {"kind": "map-target"}) == 1
    age = reg.get_sample_value("vendorkb_batch_oldest_age_seconds", {"kind": "map-target"})
    assert abs(age - 7200) < 60
    assert reg.get_sample_value(
        "vendorkb_batch_items", {"kind": "map-target", "status": "in_flight"}) == 1
    await conn.close()


async def test_spend_sums(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    parent = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO task_runs (run_id, kind, state, origin, spent_usd, started_at)"
        " VALUES (%s, 'refresh', 'succeeded', 'cluster', 1.0, now() - interval '1 hour')",
        (parent,))
    for kind, usd in (("extract", 0.25), ("support", 0.5)):
        await conn.execute(
            "INSERT INTO task_runs (run_id, parent_id, kind, state, origin, spent_usd,"
            " started_at) VALUES (%s, %s, %s, 'succeeded', 'cluster', %s,"
            " now() - interval '1 hour')", (str(uuid.uuid4()), parent, kind, usd))
    await conn.execute(
        "INSERT INTO task_runs (run_id, kind, state, origin, spent_usd, started_at)"
        " VALUES (%s, 'map', 'succeeded', 'manual', 0.1, now() - interval '3 days')",
        (str(uuid.uuid4()),))
    reg, _ = await _registry(conn)

    def v(kind, window):
        return reg.get_sample_value("vendorkb_pipeline_spend_usd",
                                    {"kind": kind, "window": window})
    assert v("extract", "24h") == pytest.approx(0.25)
    assert v("support", "24h") == pytest.approx(0.5)
    assert v("extract", "7d") == pytest.approx(0.25)
    assert v("map", "24h") is None
    assert v("map", "7d") == pytest.approx(0.1)
    assert v("refresh", "24h") is None and v("refresh", "7d") is None
    await conn.close()


async def test_support_backlog_matches_waiting(vkb_dsn, monkeypatch):
    from vendorkb.claims import support
    from vendorkb.metrics_db import _support_waiting

    async def dims(_conn):  # the topic loader's dimensions: one in-scope dimension
        return ["d.one"]

    monkeypatch.setattr(support, "topic_dimensions", dims)
    conn = await _fresh(vkb_dsn)
    did = await _doc(conn)
    cid = await _chunk(conn, did, 0)
    eid = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status) VALUES (%s, %s, 'm', 'claims-v3.1', 'done')", (eid, cid))
    ids = [str(uuid.uuid4()) for _ in range(4)]
    for claim in ids:
        await conn.execute(
            "INSERT INTO raw_claims (claim_id, extraction_id, chunk_id, vendor,"
            " capability_raw, support, evidence_quote, gate, extractor_model, prompt_version,"
            " mapped_version, dimension_id) VALUES (%s, %s, %s, 'V', 'c', 'supported', 'q',"
            " 'exact', 'm', 'claims-v3.1', 'v1', 'd.one')", (claim, eid, cid))
    # ids[0] flagged and unreviewed, ids[1] checked clean, ids[2] held by an open check batch,
    # ids[3] due
    for claim, ok in ((ids[0], False), (ids[1], True)):
        await conn.execute(
            "INSERT INTO claim_support (claim_id, supported, checker) VALUES (%s, %s, 'c')",
            (claim, ok))
    await conn.execute(
        "INSERT INTO llm_batches (batch_id, kind, model, custom_ids, status)"
        " VALUES ('hb', %s, 'm', ARRAY[%s], 'submitted')", (support.CHECK_KIND, ids[2]))
    expected = await support.waiting(conn, "v1")
    assert expected == 2
    assert await _support_waiting(conn, "v1") == expected
    await conn.close()


async def test_last_success_failure_and_cursor(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    for state in ("succeeded", "failed"):
        await conn.execute(
            "INSERT INTO task_runs (run_id, kind, state, origin, ended_at)"
            " VALUES (%s, 'sync', %s, 'cluster', now())", (str(uuid.uuid4()), state))
    await conn.execute("INSERT INTO sync_state (key, value) VALUES ('cursor', 'c')")
    await conn.execute(
        "INSERT INTO task_runs (run_id, kind, stage, state, origin, ended_at)"
        " VALUES (%s, 'refresh', 'collect', 'succeeded', 'cluster', now() - interval '1 hour')",
        (str(uuid.uuid4()),))
    reg, _ = await _registry(conn)
    assert reg.get_sample_value(
        "vendorkb_task_last_success_timestamp_seconds", {"kind": "sync", "stage": ""}) > 0
    assert reg.get_sample_value(
        "vendorkb_task_last_failure_timestamp_seconds", {"kind": "sync", "stage": ""}) > 0
    # the hourly collect is its own series: the weekly refresh never succeeded
    assert reg.get_sample_value("vendorkb_task_last_success_timestamp_seconds",
                                {"kind": "refresh", "stage": "collect"}) > 0
    assert reg.get_sample_value("vendorkb_task_last_success_timestamp_seconds",
                                {"kind": "refresh", "stage": ""}) is None
    assert reg.get_sample_value("vendorkb_sync_cursor_timestamp_seconds") > 0
    sizes = [s for m in reg.collect() if m.name == "vendorkb_table_size_bytes"
             for s in m.samples]
    assert 0 < len(sizes) <= 15
    await conn.close()


async def test_refresh_failure_keeps_last_snapshot(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    await _run(conn, "map", done=4)
    collector = PipelineCollector()
    collector.update(await snapshot(conn), True)
    reg = CollectorRegistry()
    reg.register(collector)
    assert reg.get_sample_value("vendorkb_metrics_refresh_ok") == 1
    assert reg.get_sample_value("vendorkb_metrics_refresh_timestamp_seconds") > 0
    collector.update(None, False)
    assert reg.get_sample_value("vendorkb_metrics_refresh_ok") == 0
    assert [s.value for m in reg.collect() if m.name == "vendorkb_task_done"
            for s in m.samples] == [4]
    await conn.close()


async def test_collect_before_first_snapshot():
    reg = CollectorRegistry()
    reg.register(PipelineCollector())
    assert reg.get_sample_value("vendorkb_metrics_refresh_ok") == 0
    # no good refresh yet: no timestamp (0 would read as "56 years ago")
    assert reg.get_sample_value("vendorkb_metrics_refresh_timestamp_seconds") is None


async def test_no_question_text_in_labels(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    await _run(conn, "map", stage="targets")
    await conn.execute(
        "INSERT INTO taxonomy_meta (key, value) VALUES ('taxonomy_version', 'abc123')")
    reg, _ = await _registry(conn)
    allowed = {
        "kind": {"sync", "refresh", "extract", "map", "support", "capabilities", "propose",
                 "map-target", "map-dimension", "support-check", "support-review"},
        "stage": {"", "targets", "dimensions"},
        "origin": {"cluster", "manual"},
        "status": {"in_flight", "done", "errored", "failed", "submitted", "processed"},
        "what": {"documents", "chunks", "claims", "capabilities"},
        "window": {"24h", "7d"},
    }
    for m in reg.collect():
        for s in m.samples:
            for k, v in s.labels.items():
                if k in allowed:
                    assert v in allowed[k], (m.name, k, v)
                else:
                    assert k in {"run", "table", "version"}, (m.name, k)
    await conn.close()


async def test_refresh_loop_marks_failure_and_recovers(vkb_dsn):
    import asyncio

    from vendorkb.metrics_db import refresh_loop

    collector = PipelineCollector()
    reg = CollectorRegistry()
    reg.register(collector)
    bad = asyncio.create_task(refresh_loop("postgresql://x:y@127.0.0.1:1/none", collector, 0.05))
    await asyncio.sleep(0.5)
    bad.cancel()
    assert reg.get_sample_value("vendorkb_metrics_refresh_ok") == 0
    good = asyncio.create_task(refresh_loop(vkb_dsn, collector, 0.05))
    await asyncio.sleep(1.5)
    good.cancel()
    assert reg.get_sample_value("vendorkb_metrics_refresh_ok") == 1


async def test_refresh_timeout_counts_as_failure(vkb_dsn, monkeypatch):
    import asyncio

    from vendorkb import metrics_db

    async def slow(_conn):
        await asyncio.sleep(30)

    collector = PipelineCollector()
    reg = CollectorRegistry()
    reg.register(collector)
    collector.update(Snapshot(corpus={"documents": 7}), True)
    monkeypatch.setattr(metrics_db, "snapshot", slow)
    task = asyncio.create_task(metrics_db.refresh_loop(vkb_dsn, collector, 0.05, timeout_s=0.1))
    await asyncio.sleep(0.2 + 5.2)  # wait_for allows timeout_s + 5
    task.cancel()
    assert reg.get_sample_value("vendorkb_metrics_refresh_ok") == 0
    assert reg.get_sample_value("vendorkb_corpus", {"what": "documents"}) == 7
