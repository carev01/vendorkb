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
