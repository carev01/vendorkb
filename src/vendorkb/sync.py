"""``vendorkb sync``: bootstrap, then incremental pulls, from the DocExtractor delta feed.

The stream is consumed in order, in pages. Each page is embedded in one call and written in
one transaction together with the bootstrap watermark, so ``bootstrap_after`` always means
"every id up to here is applied". The cursor moves only on the terminal cursor line.
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from contextlib import aclosing
from dataclasses import dataclass

import psycopg

from vendorkb.chunker import Chunk, chunk_markdown
from vendorkb.clients import DocExtractor, Embedder
from vendorkb.feed import BootstrapStart, Content, CursorLine, Tombstone
from vendorkb.store import DocMeta, delete_document, stored_hashes, update_metadata, write_document

logger = logging.getLogger(__name__)

LOCK_KEY = 0x766B62  # "vkb": single-flight across sync runs


@dataclass
class SyncReport:
    mode: str
    records: int = 0
    written: int = 0
    unchanged: int = 0
    metadata_updated: int = 0
    removed: int = 0
    failed: int = 0
    tokens: int = 0
    seconds: float = 0.0
    complete: bool = False
    stopped_by_budget: bool = False
    locked: bool = False


Item = Content | Tombstone


async def _load_state(conn: psycopg.AsyncConnection) -> dict[str, str]:
    cur = await conn.execute("SELECT key, value FROM sync_state")
    return {k: v for k, v in await cur.fetchall()}


async def _set_state(conn: psycopg.AsyncConnection, key: str, value: str) -> None:
    await conn.execute(
        "INSERT INTO sync_state(key, value) VALUES (%s, %s) ON CONFLICT (key)"
        " DO UPDATE SET value=EXCLUDED.value, updated_at=now()", (key, value))


def _hash(item: Content) -> str | None:
    if item.content_hash:
        return item.content_hash
    if item.markdown is None:
        return None
    return hashlib.sha256(item.markdown.encode("utf-8")).hexdigest()


class _Run:
    def __init__(self, conn: psycopg.AsyncConnection, docext: DocExtractor,
                 embedder: Embedder, report: SyncReport, state: dict[str, str]) -> None:
        self.conn = conn
        self.docext = docext
        self.embedder = embedder
        self.report = report
        self.state = state
        self.names: dict[str, str] = {}
        self.refreshed = False

    async def load_names(self) -> None:
        cur = await self.conn.execute("SELECT source_id, name FROM sources")
        self.names = {str(sid): name for sid, name in await cur.fetchall()}

    async def _source_name(self, source_id: uuid.UUID | None) -> str | None:
        if source_id is None:
            return None
        key = str(source_id)
        if key not in self.names and not self.refreshed:
            # Once per run: a source still unknown after the refresh stays NULL.
            self.refreshed = True
            fetched = await self.docext.source_names()
            async with self.conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO sources(source_id, name) VALUES (%s, %s) ON CONFLICT"
                    " (source_id) DO UPDATE SET name=EXCLUDED.name, refreshed_at=now()",
                    list(fetched.items()))
            self.names.update(fetched)
        return self.names.get(key)

    async def flush(self, page: list[Item]) -> None:
        """Embed and write one page in a single transaction; nothing is written on error."""
        if not page:
            return
        ids = [it.doc_id for it in page]
        stored = await stored_hashes(self.conn, ids)
        deletes: list[uuid.UUID] = []
        unchanged: list[DocMeta] = []
        failures: list[tuple[uuid.UUID, str]] = []
        writes: list[tuple[DocMeta, str, list[Chunk]]] = []
        for it in page:
            if isinstance(it, Tombstone):
                deletes.append(it.doc_id)
                continue
            meta = DocMeta(doc_id=it.doc_id, vendor=it.vendor, product=it.product,
                           source=await self._source_name(it.source_id),
                           source_id=it.source_id, title=it.title, url=it.url)
            h = _hash(it)
            if h is not None and stored.get(it.doc_id) == h:
                unchanged.append(meta)
            elif it.markdown is None or not it.markdown.strip():
                failures.append((it.doc_id, "missing markdown"))
            elif not (chunks := chunk_markdown(it.title or "", it.markdown)):
                failures.append((it.doc_id, "markdown yields 0 chunks"))
            else:
                assert h is not None  # markdown is present
                writes.append((meta, h, chunks))
        texts = [c.text for _, _, chunks in writes for c in chunks]
        vectors = await self.embedder.embed(texts, input_type="document") if texts else []

        removed = metadata_updated = 0
        async with self.conn.transaction():
            for doc_id in deletes:
                removed += await delete_document(self.conn, doc_id)
            for meta in unchanged:
                metadata_updated += await update_metadata(self.conn, meta)
            at = 0
            for meta, h, chunks in writes:
                await write_document(self.conn, meta, h, chunks,
                                     vectors[at:at + len(chunks)])
                at += len(chunks)
            settled = deletes + [m.doc_id for m in unchanged] + [m.doc_id for m, _, _ in writes]
            await self.conn.execute("DELETE FROM sync_failures WHERE doc_id = ANY(%s)",
                                    (settled,))
            for doc_id, reason in failures:
                await self.conn.execute(
                    "INSERT INTO sync_failures(doc_id, reason) VALUES (%s, %s) ON CONFLICT"
                    " (doc_id) DO UPDATE SET reason=EXCLUDED.reason, failed_at=now()",
                    (doc_id, reason))
            if self.report.mode == "bootstrap":
                # Same transaction as the data: the watermark can never run ahead of it.
                await _set_state(self.conn, "bootstrap_after", str(page[-1].doc_id))
        r = self.report
        r.removed += removed
        r.unchanged += len(unchanged)
        r.metadata_updated += metadata_updated
        r.failed += len(failures)
        r.written += len(writes)

    async def finish(self, line: CursorLine) -> None:
        async with self.conn.transaction():
            if self.report.mode == "bootstrap":
                # The contract: incremental pulls start from the FIRST bootstrap_start.
                since = self.state.get("bootstrap_next_since")
                if since is None:
                    raise RuntimeError("bootstrap ended with no stored bootstrap_start")
                await _set_state(self.conn, "cursor", since)
                await _set_state(self.conn, "mode", "incremental")
                await self.conn.execute(
                    "DELETE FROM sync_state WHERE key='bootstrap_after'")
            else:
                await _set_state(self.conn, "cursor", line.next_since)


async def sync(conn: psycopg.AsyncConnection, *, docext: DocExtractor, embedder: Embedder,
               page_size: int = 200, max_tokens: int | None = None) -> SyncReport:
    started = time.monotonic()
    tokens0 = embedder.tokens_used
    cur = await conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
    got = await cur.fetchone()
    state = await _load_state(conn)
    report = SyncReport(mode="incremental" if "cursor" in state else "bootstrap")
    if not (got and got[0]):
        report.locked = True
        return report
    page: list[Item] = []
    try:
        run = _Run(conn, docext, embedder, report, state)
        await run.load_names()
        if report.mode == "bootstrap":
            stream = docext.delta(bootstrap_after=state.get("bootstrap_after"))
        else:
            stream = docext.delta(since=state["cursor"])

        def over_budget() -> bool:
            return (max_tokens is not None
                    and embedder.tokens_used - tokens0 >= max_tokens)

        async with aclosing(stream) as items:
            async for item in items:
                if isinstance(item, BootstrapStart):
                    if report.mode == "bootstrap" and "bootstrap_next_since" not in state:
                        await _set_state(conn, "bootstrap_next_since", item.next_since)
                        await _set_state(conn, "mode", "bootstrap")
                        state["bootstrap_next_since"] = item.next_since
                    continue
                if isinstance(item, CursorLine):
                    await run.flush(page)
                    page = []
                    await run.finish(item)
                    report.complete = True
                    break
                report.records += 1
                if any(p.doc_id == item.doc_id for p in page):
                    # Stream order must win for a repeated id: write the earlier one first.
                    await run.flush(page)
                    page = []
                    if over_budget():
                        report.stopped_by_budget = True
                        break
                page.append(item)
                if len(page) >= page_size:
                    await run.flush(page)
                    page = []
                    if over_budget():
                        report.stopped_by_budget = True
                        break
        if not report.complete and not report.stopped_by_budget:
            logger.warning("delta stream ended without a cursor line; %d record(s) not "
                           "written, resume next run", len(page))
    except Exception:
        logger.exception("sync aborted; page %s..%s not written",
                         page[0].doc_id if page else None, page[-1].doc_id if page else None)
    finally:
        try:
            await conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
        except psycopg.Error:
            logger.warning("could not release the sync advisory lock", exc_info=True)
        report.tokens = embedder.tokens_used - tokens0
        report.seconds = round(time.monotonic() - started, 3)
    return report
