import asyncio
import uuid
from datetime import UTC, datetime

import psycopg
import pytest
from vendorkb.db import connect, init_schema


async def test_init_schema_is_idempotent_and_creates_tables(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        await init_schema(conn, embed_dim=768)
        cur = await conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        )
        tables = {r[0] for r in await cur.fetchall()}
        assert {"documents", "chunks"} <= tables
        doc_id, chunk_id = uuid.uuid4(), uuid.uuid4()
        await conn.execute(
            "INSERT INTO documents(doc_id, vendor, content_hash, fetched_at) "
            "VALUES (%s, 'v', 'h', %s)",
            (doc_id, datetime.now(UTC)),
        )
        await conn.execute(
            "INSERT INTO chunks(chunk_id, doc_id, section_path, ordinal, text, token_count,"
            " embedding) VALUES (%s, %s, %s, 0, 'hello', 1, %s)",
            (chunk_id, doc_id, ["a"], [0.1] * 768),
        )
        cur = await conn.execute("SELECT count(*) FROM chunks WHERE tsv @@ to_tsquery('hello')")
        assert (await cur.fetchone())[0] == 1
    finally:
        await conn.close()


async def test_init_schema_upgrades_a_pre_sync_database(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await conn.execute("DROP TABLE IF EXISTS chunks, documents, sync_state, sources, "
                           "sync_failures CASCADE")
        await conn.execute(
            "CREATE TABLE documents (doc_id uuid PRIMARY KEY, vendor text NOT NULL, "
            "product text, source text, title text, url text, content_hash text NOT NULL, "
            "graph_content_hash text, fetched_at timestamptz NOT NULL)")
        await init_schema(conn, embed_dim=768)
        await init_schema(conn, embed_dim=768)
        cur = await conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='documents'")
        columns = {r[0] for r in await cur.fetchall()}
        assert "graph_content_hash" not in columns
        assert "source_id" in columns
        cur = await conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        tables = {r[0] for r in await cur.fetchall()}
        assert {"sync_state", "sources", "sync_failures"} <= tables
    finally:
        await conn.close()


async def test_init_schema_adds_the_quarantine_hash_to_sync_failures(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        await conn.execute("DROP TABLE sync_failures")
        await conn.execute("CREATE TABLE sync_failures (doc_id uuid PRIMARY KEY, reason text "
                           "NOT NULL, failed_at timestamptz NOT NULL DEFAULT now())")
        await conn.execute("INSERT INTO sync_failures(doc_id, reason) VALUES (%s, 'old')",
                           (uuid.uuid4(),))
        # a database from before this column also predates the current schema marker
        await conn.execute("DELETE FROM taxonomy_meta WHERE key = 'schema_version'")
        await init_schema(conn, embed_dim=768)
        await init_schema(conn, embed_dim=768)
        cur = await conn.execute("SELECT reason, content_hash FROM sync_failures")
        assert await cur.fetchall() == [("old", None)]
    finally:
        await conn.close()


async def test_init_schema_gives_the_superseded_column_planner_statistics(vkb_dsn):
    """BACKLOG 35: a column added to a filled table has no statistics until the next ANALYZE.
    Without them the planner took ``superseded_at IS NULL`` for 0.5% of the chunks and sorted
    every chunk by distance instead of using the HNSW index: 2.2 s a vector search."""
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        await conn.execute("DELETE FROM documents")
        doc = uuid.uuid4()
        await conn.execute(
            "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
            " fetched_at) VALUES (%s, 'V', 'P', 'S', 'T', 'http://u', 'h', now())", (doc,))
        await conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
            " embedding) SELECT gen_random_uuid(), %s, ARRAY['A'], n, 'text', 1,"
            " array_fill(0.1, ARRAY[768])::vector FROM generate_series(1, 50) n", (doc,))
        await conn.execute("DELETE FROM pg_statistic WHERE starelid = 'chunks'::regclass")
        await conn.execute("DELETE FROM taxonomy_meta WHERE key = 'schema_version'")  # an upgrade
        await init_schema(conn, embed_dim=768)
        cur = await conn.execute("SELECT null_frac FROM pg_stats WHERE tablename = 'chunks'"
                                 " AND attname = 'superseded_at'")
        assert await cur.fetchall() == [(1.0,)]
    finally:
        await conn.execute("DELETE FROM documents")
        await conn.close()


async def test_a_current_schema_is_not_rerun_so_a_long_read_does_not_block_it(vkb_dsn):
    from vendorkb import db

    conn, reader = await connect(vkb_dsn), await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        cur = await conn.execute("SELECT value FROM taxonomy_meta WHERE key='schema_version'")
        assert (await cur.fetchone())[0] == db.schema_marker(768)
        # a long read holds ACCESS SHARE on raw_claims; re-running the DDL would queue behind it
        await reader.execute("BEGIN")
        await reader.execute("SELECT count(*) FROM raw_claims")
        await asyncio.wait_for(init_schema(conn, embed_dim=768), timeout=5)  # skipped: no DDL
    finally:
        await reader.execute("ROLLBACK")
        await reader.close()
        await conn.close()


async def test_a_changed_schema_fails_fast_instead_of_queueing(vkb_dsn, monkeypatch):
    from vendorkb import db

    monkeypatch.setattr(db, "SCHEMA_LOCK_TIMEOUT_MS", 300)
    conn, reader = await connect(vkb_dsn), await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        await conn.execute("UPDATE taxonomy_meta SET value='stale' WHERE key='schema_version'")
        await reader.execute("BEGIN")
        await reader.execute("SELECT count(*) FROM raw_claims")
        with pytest.raises(psycopg.errors.LockNotAvailable):
            await asyncio.wait_for(init_schema(conn, embed_dim=768), timeout=10)
        cur = await conn.execute("SELECT value FROM taxonomy_meta WHERE key='schema_version'")
        assert (await cur.fetchone())[0] == "stale"  # nothing half-applied, retried next run
    finally:
        await reader.execute("ROLLBACK")
        await reader.close()
        await conn.close()


async def test_a_dropped_table_is_recreated_even_when_the_marker_is_current(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        await conn.execute("DROP TABLE IF EXISTS capabilities")
        await init_schema(conn, embed_dim=768)
        cur = await conn.execute("SELECT to_regclass('capabilities') IS NOT NULL")
        assert (await cur.fetchone())[0]
    finally:
        await conn.close()


async def test_a_full_upgrade_over_a_populated_database_keeps_every_row(vkb_dsn):
    """M12: the whole DDL re-run (as after a schema change) on a database with rows in every
    table the services write: nothing is dropped, rewritten away or refused by a constraint."""
    from tests.integration.test_capabilities_v2 import _claim, _fresh
    from vendorkb.taxonomy import capabilities

    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)  # schema, fixture taxonomy (dimensions, targets, aliases)
        await _claim(conn, "coverage.backup", "okta", class_id="saas.identity",
                     object_type="user", mapped_version=version)
        await _claim(conn, "retention.maximum_period", mapped_version=version,
                     value=30, unit="days")
        await capabilities.build(conn)
        await conn.execute("INSERT INTO extract_batches (batch_id, extractor_model,"
                           " prompt_version, chunk_ids, status) VALUES ('b1','m','p','{}','processed')")
        await conn.execute("INSERT INTO propose_runs (run_id, kind, model, spent_usd)"
                           " VALUES ('r1','dimension','m',0.5)")
        await conn.execute("INSERT INTO map_decisions (key, kind, phrase, candidates, choice, model,"
                           " source, rules) VALUES ('k1','dimension','x','{a}','a','m','llm','r')")
        await conn.execute("INSERT INTO embedding_cache (model, input_type, text, vec)"
                           " VALUES ('m@8','document','x','\\x00')")
        cur = await conn.execute("SELECT table_name FROM information_schema.tables"
                                 " WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
        tables = sorted(r[0] for r in await cur.fetchall())

        async def counts() -> dict[str, int]:
            out = {}
            for t in tables:
                cur = await conn.execute(f'SELECT count(*) FROM "{t}"')
                out[t] = (await cur.fetchone())[0]
            return out
        before = await counts()
        assert all(before[t] > 0 for t in ("documents", "chunks", "raw_claims", "capabilities",
                                            "dimensions", "aliases", "propose_runs"))
        await conn.execute("DELETE FROM taxonomy_meta WHERE key = 'schema_version'")
        await init_schema(conn, embed_dim=8)  # the full DDL runs again
        assert await counts() == before
        cur = await conn.execute("SELECT count(*) FROM taxonomy_meta WHERE key = 'schema_version'")
        assert (await cur.fetchone())[0] == 1
    finally:
        await conn.close()
