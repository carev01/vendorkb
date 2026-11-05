"""Document store shared by ``sync``: write, re-tag and delete documents with their chunks.

None of these opens a transaction: the caller wraps a page of writes in
``async with conn.transaction():`` so the page commits or rolls back as one.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg

from vendorkb.chunker import Chunk


@dataclass(frozen=True)
class DocMeta:
    doc_id: uuid.UUID
    vendor: str
    product: str | None
    source: str | None
    source_id: uuid.UUID | None
    title: str | None
    url: str | None


async def stored_hashes(conn: psycopg.AsyncConnection,
                        doc_ids: list[uuid.UUID]) -> dict[uuid.UUID, str]:
    cur = await conn.execute(
        "SELECT doc_id, content_hash FROM documents WHERE doc_id = ANY(%s)", (doc_ids,))
    return {doc_id: h for doc_id, h in await cur.fetchall()}


async def write_document(conn: psycopg.AsyncConnection, meta: DocMeta, content_hash: str,
                         chunks: list[Chunk], vectors: list[list[float]]) -> None:
    await conn.execute("DELETE FROM documents WHERE doc_id=%s", (meta.doc_id,))
    await conn.execute(
        "INSERT INTO documents(doc_id, vendor, product, source, source_id, title, url,"
        " content_hash, fetched_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (meta.doc_id, meta.vendor, meta.product, meta.source, meta.source_id, meta.title,
         meta.url, content_hash, datetime.now(UTC)))
    async with conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO chunks(chunk_id, doc_id, section_path, ordinal, text, token_count,"
            " embedding) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            [(uuid.uuid4(), meta.doc_id, list(c.section_path), c.ordinal, c.text,
              c.token_count, v) for c, v in zip(chunks, vectors, strict=True)])


async def update_metadata(conn: psycopg.AsyncConnection, meta: DocMeta) -> bool:
    """Re-tag a stored document without touching its chunks; True when a row changed."""
    cur = await conn.execute(
        "UPDATE documents SET vendor=%s, product=%s, source=%s, source_id=%s, title=%s, url=%s"
        " WHERE doc_id=%s AND (vendor, product, source, source_id, title, url)"
        " IS DISTINCT FROM (%s, %s, %s, %s, %s, %s)",
        (meta.vendor, meta.product, meta.source, meta.source_id, meta.title, meta.url,
         meta.doc_id, meta.vendor, meta.product, meta.source, meta.source_id, meta.title,
         meta.url))
    return cur.rowcount > 0


async def delete_document(conn: psycopg.AsyncConnection, doc_id: uuid.UUID) -> bool:
    cur = await conn.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))
    return cur.rowcount > 0
