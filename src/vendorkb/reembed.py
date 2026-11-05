"""Embedding-model switch without re-fetching: fill a candidate column, then promote it.

`reembed` embeds every chunk's stored text into ``chunks.embedding_next`` (resumable: only
NULL rows are embedded). Searches can read that column for an A/B (`search_docs(...,
vector_column="embedding_next")`). `promote` then replaces ``embedding`` with it.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import psycopg
from pgvector import Vector

from vendorkb.clients import Embedder

logger = logging.getLogger(__name__)

CANDIDATE = "embedding_next"


@dataclass
class ReembedReport:
    chunks: int
    tokens: int
    seconds: float
    remaining: int


async def _column_type(conn: psycopg.AsyncConnection, column: str) -> str | None:
    cur = await conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = %s AND NOT attisdropped", (column,))
    row = await cur.fetchone()
    return row[0] if row else None


async def ensure_candidate_column(conn: psycopg.AsyncConnection, dim: int) -> None:
    want = f"vector({int(dim)})"
    have = await _column_type(conn, CANDIDATE)
    if have is None:
        await conn.execute(f"ALTER TABLE chunks ADD COLUMN {CANDIDATE} {want}")
    elif have != want:
        raise RuntimeError(f"chunks.{CANDIDATE} is {have}, expected {want}: drop it first")


async def _remaining(conn: psycopg.AsyncConnection) -> int:
    cur = await conn.execute(f"SELECT count(*) FROM chunks WHERE {CANDIDATE} IS NULL")
    row = await cur.fetchone()
    return int(row[0]) if row else 0


async def reembed(conn: psycopg.AsyncConnection, embedder: Embedder, *, dim: int,
                  batch: int, concurrency: int, limit: int | None = None) -> ReembedReport:
    """Embed chunks whose candidate vector is NULL, ``batch`` texts per request and
    ``concurrency`` requests at a time; each page is written before the next is read, so
    an interrupted run resumes where it stopped. A request that still fails after the
    embedder's retries aborts the run (nothing half-written: rows stay NULL)."""
    await ensure_candidate_column(conn, dim)
    start, done, tokens0 = time.monotonic(), 0, embedder.tokens_used
    page = batch * concurrency
    while limit is None or done < limit:
        n = page if limit is None else min(page, limit - done)
        cur = await conn.execute(
            f"SELECT chunk_id, text FROM chunks WHERE {CANDIDATE} IS NULL "
            "ORDER BY chunk_id LIMIT %s", (n,))
        rows = await cur.fetchall()
        if not rows:
            break
        groups = [rows[i:i + batch] for i in range(0, len(rows), batch)]
        vectors = await asyncio.gather(
            *(embedder.embed([t for _, t in g], input_type="document") for g in groups))
        async with conn.transaction():
            async with conn.cursor() as wcur:
                await wcur.executemany(
                    f"UPDATE chunks SET {CANDIDATE} = %s WHERE chunk_id = %s",
                    [(Vector(v), cid) for g, vs in zip(groups, vectors, strict=True)
                     for (cid, _), v in zip(g, vs, strict=True)])
        done += len(rows)
        logger.info("reembed: %d chunks, %d tokens", done, embedder.tokens_used - tokens0)
    return ReembedReport(chunks=done, tokens=embedder.tokens_used - tokens0,
                         seconds=round(time.monotonic() - start, 1),
                         remaining=await _remaining(conn))


async def promote(conn: psycopg.AsyncConnection) -> str:
    """Replace ``chunks.embedding`` with the fully filled candidate column and rebuild the
    HNSW index. Refuses while any candidate vector is NULL. Returns the new column type."""
    have = await _column_type(conn, CANDIDATE)
    if have is None:
        raise RuntimeError(f"chunks.{CANDIDATE} does not exist: run reembed first")
    missing = await _remaining(conn)
    if missing:
        raise RuntimeError(f"{missing} chunks have no {CANDIDATE} yet: finish reembed first")
    cur = await conn.execute(
        "SELECT i.indexrelid::regclass::text FROM pg_index i JOIN pg_attribute a "
        "ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
        "WHERE i.indrelid = 'chunks'::regclass AND a.attname = %s", (CANDIDATE,))
    candidate_indexes = [r[0] for r in await cur.fetchall()]
    async with conn.transaction():
        # An index built on the candidate for the A/B would follow the rename and leave
        # two HNSW indexes on `embedding`: drop it, the canonical one is rebuilt below.
        for name in candidate_indexes:
            await conn.execute(f'DROP INDEX "{name}"')
        await conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
        await conn.execute("ALTER TABLE chunks DROP COLUMN embedding")
        await conn.execute(f"ALTER TABLE chunks RENAME COLUMN {CANDIDATE} TO embedding")
        await conn.execute("ALTER TABLE chunks ALTER COLUMN embedding SET NOT NULL")
        await conn.execute("CREATE INDEX chunks_embedding_idx ON chunks "
                           "USING hnsw (embedding vector_cosine_ops)")
    return have
