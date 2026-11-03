"""Resumable corpus ingest: fetch -> chunk -> embed concurrently, write serially."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import psycopg

from vendorkb.chunker import Chunk, chunk_markdown
from vendorkb.clients import CorpusEntry, DocExtractor, Embedder

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    fetched: int = 0
    skipped_unchanged: int = 0
    failed: list[str] = field(default_factory=list)
    changed_since_graph: list[str] = field(default_factory=list)
    chunks_written: int = 0


@dataclass
class _Outcome:
    state: str  # written | skipped | failed
    chunks: int = 0
    changed: bool = False
    reason: str = ""


async def ingest(conn: psycopg.AsyncConnection, corpus: list[CorpusEntry], *,
                 docext: DocExtractor, embedder: Embedder,
                 concurrency: int = 8) -> IngestReport:
    sem = asyncio.Semaphore(max(1, concurrency))
    db_lock = asyncio.Lock()  # one connection cannot run concurrent transactions

    async def one(entry: CorpusEntry) -> _Outcome:
        try:
            async with sem:
                art = await docext.article(entry.article_id)
                md = art.get("content_markdown")
                if not isinstance(md, str):
                    return _Outcome("failed", reason="no content_markdown")
                content_hash = hashlib.sha256(md.encode()).hexdigest()
                changed = (entry.graph_content_hash is not None
                           and content_hash != entry.graph_content_hash)
                chunks = chunk_markdown(entry.title, md)
                if not chunks:
                    return _Outcome("failed", reason="markdown produced 0 chunks")
                async with db_lock:
                    cur = await conn.execute(
                        "SELECT content_hash FROM documents WHERE doc_id=%s",
                        (uuid.UUID(entry.article_id),))
                    row = await cur.fetchone()
                if row is not None and row[0] == content_hash:
                    return _Outcome("skipped", changed=changed)
                vectors = await embedder.embed([c.text for c in chunks])
            async with db_lock:
                await _replace(conn, entry, content_hash, chunks, vectors)
            return _Outcome("written", chunks=len(chunks), changed=changed)
        except Exception as exc:  # recorded in the report, never aborts the run
            logger.warning("ingest of %s failed: %s", entry.article_id, exc)
            return _Outcome("failed", reason=f"{type(exc).__name__}: {exc}")

    outcomes = await asyncio.gather(*(one(e) for e in corpus))
    report = IngestReport()
    for entry, out in zip(corpus, outcomes, strict=True):
        if out.changed:
            report.changed_since_graph.append(entry.article_id)
        if out.state == "written":
            report.fetched += 1
            report.chunks_written += out.chunks
        elif out.state == "skipped":
            report.skipped_unchanged += 1
        else:
            report.failed.append(f"{entry.article_id}: {out.reason}")
    return report


async def _replace(conn: psycopg.AsyncConnection, entry: CorpusEntry, content_hash: str,
                   chunks: list[Chunk], vectors: list[list[float]]) -> None:
    doc_id = uuid.UUID(entry.article_id)
    async with conn.transaction():
        await conn.execute("DELETE FROM documents WHERE doc_id=%s", (doc_id,))
        await conn.execute(
            "INSERT INTO documents(doc_id, vendor, product, source, title, url, content_hash,"
            " graph_content_hash, fetched_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (doc_id, entry.vendor, entry.product, entry.source, entry.title, entry.url,
             content_hash, entry.graph_content_hash, datetime.now(UTC)))
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO chunks(chunk_id, doc_id, section_path, ordinal, text, token_count,"
                " embedding) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                [(uuid.uuid4(), doc_id, list(c.section_path), c.ordinal, c.text,
                  c.token_count, v) for c, v in zip(chunks, vectors, strict=True)])
