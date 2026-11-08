from __future__ import annotations

import hashlib
import re
from importlib import resources

import psycopg
from pgvector.psycopg import register_vector_async


async def connect(dsn: str) -> psycopg.AsyncConnection:
    """Open an autocommit connection with the pgvector adapters registered.

    The vector extension must exist before the adapter can look up its type, so a
    first connection to a fresh database creates it.
    """
    conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector_async(conn)
    except BaseException:
        await conn.close()
        raise
    return conn


# How long the schema DDL may wait for a table lock before it gives up. The DDL takes ACCESS
# EXCLUSIVE locks (ALTER TABLE ... IF NOT EXISTS takes it even when there is nothing to add); a
# request waiting behind a long read makes every later reader of that table queue behind it, so
# failing fast (the command fails, its next run retries) beats waiting.
SCHEMA_LOCK_TIMEOUT_MS = 10_000
_TABLES = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+)")


def _schema_sql(embed_dim: int) -> str:
    sql = resources.files("vendorkb").joinpath("schema.sql").read_text(encoding="utf-8")
    return sql.replace("{embed_dim}", str(int(embed_dim)))


def schema_marker(embed_dim: int) -> str:
    """The version of the schema DDL (stored as taxonomy_meta 'schema_version')."""
    return hashlib.sha256(_schema_sql(embed_dim).encode("utf-8")).hexdigest()[:16]


async def _schema_is_current(conn: psycopg.AsyncConnection, sql: str, marker: str) -> bool:
    """True when the stored marker matches and every table the DDL creates exists (tests and
    operators sometimes drop a table by hand). Plain reads: no lock beyond ACCESS SHARE."""
    tables = sorted(set(_TABLES.findall(sql)))
    cur = await conn.execute("SELECT count(*) FROM unnest(%s::text[]) t"
                             " WHERE to_regclass(t) IS NOT NULL", (tables,))
    if (await cur.fetchone())[0] != len(tables):  # type: ignore[index]
        return False
    cur = await conn.execute("SELECT value FROM taxonomy_meta WHERE key = 'schema_version'")
    row = await cur.fetchone()
    return row is not None and row[0] == marker


async def init_schema(conn: psycopg.AsyncConnection, *, embed_dim: int) -> None:
    """Create or upgrade the schema; ``embed_dim`` sizes ``chunks.embedding`` on a fresh
    database (an existing table keeps its type: change it with ``vendorkb promote-embedding``).

    Every command calls this. When the schema is current (the stored marker matches this DDL
    and its tables exist) it runs no DDL at all. Otherwise the DDL and the new marker commit in
    one transaction, waiting at most SCHEMA_LOCK_TIMEOUT_MS for each lock: on a timeout nothing
    is applied and LockNotAvailable propagates."""
    sql = _schema_sql(embed_dim)
    marker = hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]
    if await _schema_is_current(conn, sql, marker):
        return
    async with conn.transaction():
        await conn.execute(f"SET LOCAL lock_timeout = {int(SCHEMA_LOCK_TIMEOUT_MS)}")
        await conn.execute(sql)
        await conn.execute(
            "INSERT INTO taxonomy_meta (key, value) VALUES ('schema_version', %s)"
            " ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            (marker,))
