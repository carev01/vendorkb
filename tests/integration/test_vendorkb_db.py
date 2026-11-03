import uuid
from datetime import UTC, datetime

import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from vendorkb.db import connect, init_schema


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def vkb_dsn():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg.get_connection_url().replace("+psycopg2", "")


async def test_init_schema_is_idempotent_and_creates_tables(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn)
        await init_schema(conn)
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
