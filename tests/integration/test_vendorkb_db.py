import uuid
from datetime import UTC, datetime
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
