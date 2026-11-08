"""Document store shared by ``sync``: write, re-tag and delete documents with their chunks.

None of these opens a transaction: the caller wraps a page of writes in
``async with conn.transaction():`` so the page commits or rolls back as one.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg

from vendorkb.chunker import Chunk, reuse_key


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


def _norm(text: str) -> str:
    """Whitespace collapsed and lowercased: the quote check ignores both."""
    return " ".join(text.split()).lower()


async def write_document(conn: psycopg.AsyncConnection, meta: DocMeta, content_hash: str,
                         chunks: list[Chunk], vectors: list[list[float]]) -> int:
    """Store a new version of a page (a first write, or an update); the number of current
    chunks reused.

    Per chunk, so unchanged text keeps its facts. A new chunk reuses a current chunk of the
    page when their ``reuse_key`` is equal and every claim's evidence quote still appears in
    the new text (whitespace and case ignored): it keeps its ``chunk_id``, extraction and
    claims, and takes the new section path, ordinal, text and embedding. Each other current
    chunk is kept, with ``superseded_at`` set, when a claim on it is mapped (its facts are in
    the capability table or soon will be) and deleted otherwise. Superseded chunks are never
    deleted here: only ``retire_superseded`` does, once the new chunks are extracted and
    mapped. The new chunks that reused nothing are inserted."""
    # Locked until the caller's transaction ends: an extraction recording its claims on one of
    # these chunks (FOR KEY SHARE, claims/extract.py) waits, then gates them against the new
    # text, and one that committed first is seen by the quote check below.
    await conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = %s AND superseded_at IS NULL"
                       " ORDER BY chunk_id FOR UPDATE", (meta.doc_id,))
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
        "SELECT c.chunk_id, c.text, coalesce(array_agg(r.evidence_quote)"
        " FILTER (WHERE r.claim_id IS NOT NULL), '{}'), coalesce(bool_or("
        "r.mapped_version IS NOT NULL), false) FROM chunks c LEFT JOIN raw_claims r"
        " ON r.chunk_id = c.chunk_id WHERE c.doc_id = %s AND c.superseded_at IS NULL"
        " GROUP BY c.chunk_id ORDER BY c.ordinal", (meta.doc_id,))
    rows = await cur.fetchall()
    mapped = {chunk_id: has_mapped for chunk_id, _, _, has_mapped in rows}
    by_key: dict[str, list[tuple[uuid.UUID, list[str]]]] = {}  # in ordinal order
    for chunk_id, text, quotes, _ in rows:
        by_key.setdefault(reuse_key(text), []).append((chunk_id, quotes))
    reused: list[tuple[uuid.UUID, Chunk, list[float]]] = []
    fresh: list[tuple[Chunk, list[float]]] = []
    used: set[uuid.UUID] = set()
    for chunk, vector in zip(chunks, vectors, strict=True):
        text = _norm(chunk.text)
        match = next((chunk_id for chunk_id, quotes in by_key.get(reuse_key(chunk.text), [])
                      if chunk_id not in used and all(_norm(q) in text for q in quotes)), None)
        if match is None:
            fresh.append((chunk, vector))
        else:
            used.add(match)
            reused.append((match, chunk, vector))
    dropped = [chunk_id for chunk_id in mapped if chunk_id not in used]
    await conn.execute("UPDATE chunks SET superseded_at = now() WHERE chunk_id = ANY(%s)",
                       ([c for c in dropped if mapped[c]],))
    await conn.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)",
                       ([c for c in dropped if not mapped[c]],))
    # The reused chunks move to their new ordinals through negative ones, so that none
    # collides with another on the partial unique index while a swap is half done.
    async with conn.cursor() as cur:
        await cur.executemany(
            "UPDATE chunks SET section_path = %s, ordinal = %s, text = %s, token_count = %s,"
            " embedding = %s WHERE chunk_id = %s",
            [(list(c.section_path), -1 - c.ordinal, c.text, c.token_count, v, chunk_id)
             for chunk_id, c, v in reused])
        await cur.execute("UPDATE chunks SET ordinal = -1 - ordinal WHERE doc_id = %s"
                          " AND superseded_at IS NULL AND ordinal < 0", (meta.doc_id,))
        await cur.executemany(
            "INSERT INTO chunks(chunk_id, doc_id, section_path, ordinal, text, token_count,"
            " embedding) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            [(uuid.uuid4(), meta.doc_id, list(c.section_path), c.ordinal, c.text,
              c.token_count, v) for c, v in fresh])
    return len(reused)


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
