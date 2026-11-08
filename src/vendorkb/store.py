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
    """Store a new version of a page (a first write, or an update).

    The rule: the facts of the last extracted version stay until the next one is mapped. At
    most one superseded set is kept per page: the newest version whose facts may be in the
    capability table. When the page already has one, its current chunks are deleted outright
    (that version never replaced the old facts) and only ``retire_superseded``, once the new
    chunks are extracted and mapped, moves the page on. Otherwise, when any current chunk
    carries a claim, the current chunks are kept with ``superseded_at`` set (their claims, and
    the capabilities built from them, stay in every query). A page with neither is replaced
    outright: a version that was never extracted carries no fact worth keeping."""
    await conn.execute(
        "INSERT INTO documents(doc_id, vendor, product, source, source_id, title, url,"
        " content_hash, fetched_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        " ON CONFLICT (doc_id) DO UPDATE SET vendor = EXCLUDED.vendor,"
        " product = EXCLUDED.product, source = EXCLUDED.source, source_id = EXCLUDED.source_id,"
        " title = EXCLUDED.title, url = EXCLUDED.url, content_hash = EXCLUDED.content_hash,"
        " fetched_at = EXCLUDED.fetched_at",
        (meta.doc_id, meta.vendor, meta.product, meta.source, meta.source_id, meta.title,
         meta.url, content_hash, datetime.now(UTC)))
    cur = await conn.execute(
        "SELECT EXISTS (SELECT 1 FROM chunks WHERE doc_id = %(d)s AND superseded_at IS NOT NULL),"
        " EXISTS (SELECT 1 FROM chunks c JOIN raw_claims r USING (chunk_id)"
        " WHERE c.doc_id = %(d)s AND c.superseded_at IS NULL)", {"d": meta.doc_id})
    has_superseded, has_claims = await cur.fetchone()  # type: ignore[misc]
    if has_claims and not has_superseded:
        await conn.execute("UPDATE chunks SET superseded_at = now() WHERE doc_id = %s"
                           " AND superseded_at IS NULL", (meta.doc_id,))
    else:
        await conn.execute("DELETE FROM chunks WHERE doc_id = %s AND superseded_at IS NULL",
                           (meta.doc_id,))
    async with conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO chunks(chunk_id, doc_id, section_path, ordinal, text, token_count,"
            " embedding) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            [(uuid.uuid4(), meta.doc_id, list(c.section_path), c.ordinal, c.text,
              c.token_count, v) for c, v in zip(chunks, vectors, strict=True)])


async def retire_superseded(conn: psycopg.AsyncConnection, taxonomy_version: str | None) -> int:
    """Delete the superseded chunks of every page whose current version is fully extracted and
    mapped; the cascade removes their extractions, claims and capability rows. Returns the
    number of chunks deleted.

    A page is ready when each of its current chunks has a final extraction (done or failed) of
    the claims-v3 family and every claim of that family on them is mapped under
    ``taxonomy_version`` (an unmapped claim blocks it too; None: the extraction condition
    only)."""
    # Imported here, not at the top: ``sync`` imports this module and must stay free of the
    # claims stack (anthropic) in its 1 Gi pod (tests/unit/test_vendorkb_cli_imports.py).
    from vendorkb.claims.extract import PROMPT_FAMILY
    cur = await conn.execute(
        "DELETE FROM chunks s WHERE s.superseded_at IS NOT NULL AND NOT EXISTS ("
        " SELECT 1 FROM chunks c WHERE c.doc_id = s.doc_id AND c.superseded_at IS NULL AND ("
        "  NOT EXISTS (SELECT 1 FROM extractions e WHERE e.chunk_id = c.chunk_id"
        "   AND e.status IN ('done', 'failed') AND e.prompt_version LIKE %(family)s)"
        "  OR (%(version)s::text IS NOT NULL AND EXISTS (SELECT 1 FROM raw_claims r"
        "   WHERE r.chunk_id = c.chunk_id AND r.prompt_version LIKE %(family)s"
        "   AND r.mapped_version IS DISTINCT FROM %(version)s::text))))",
        {"family": PROMPT_FAMILY, "version": taxonomy_version})
    return cur.rowcount


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
