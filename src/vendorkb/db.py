from __future__ import annotations

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


async def init_schema(conn: psycopg.AsyncConnection) -> None:
    sql = resources.files("vendorkb").joinpath("schema.sql").read_text(encoding="utf-8")
    await conn.execute(sql)
