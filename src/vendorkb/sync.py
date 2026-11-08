"""``vendorkb sync``: bootstrap, then incremental pulls, from the DocExtractor delta feed.

The stream is consumed in order, in pages. Each page is embedded in one call (record by record
in quarantine mode, below) and written in one transaction together with the bootstrap
watermark, so ``bootstrap_after`` always means "every id up to here is applied". The cursor
moves only on the terminal cursor line.

Two guards protect the one-shot embedding budget: during a bootstrap ``--max-tokens`` caps
the running total across attempts (``bootstrap_tokens``), and a page that aborts after
embedding ``BREAKER_LIMIT`` times in a row (matched by its first n items, so a partial tail
page that grows with new arrivals still matches) runs once in quarantine mode: each record is
embedded and written on its own, failing records go to ``sync_failures`` (with the content
hash they failed with, so a replay of that content is skipped unembedded) and the rest commit.
If every record (two or more) fails in quarantine the page is refused, unembedded, until it
changes upstream or an operator clears the breaker.

The bootstrap's end prunes documents the stream never served, but refuses (rolls back,
exits non-complete) to prune more than ``max(PRUNE_FLOOR, PRUNE_SHARE * documents)`` unless
``--allow-prune`` covers the count: an empty or partial stream must not wipe the corpus.
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from functools import partial
from typing import cast

import httpx
import psycopg

from vendorkb.chunker import Chunk, chunk_markdown
from vendorkb.clients import DocExtractor, Embedder
from vendorkb.feed import BootstrapStart, Content, CursorLine, MalformedRecord, Tombstone
from vendorkb.store import DocMeta, delete_document, stored_hashes, update_metadata, write_document

logger = logging.getLogger(__name__)

LOCK_KEY = 0x766B62  # "vkb": single-flight across sync runs
BREAKER_LIMIT = 3  # post-embed aborts of one page before it runs in quarantine mode
PRUNE_FLOOR = 500  # the bootstrap prunes at most max(PRUNE_FLOOR, PRUNE_SHARE * documents)
PRUNE_SHARE = 0.02
CANARY_TEXT = "backup"  # embedded once after a per-record rejection: systemic or not?


@dataclass
class SyncReport:
    mode: str
    records: int = 0
    written: int = 0
    chunks_reused: int = 0  # chunks of updated pages that kept their facts (same text)
    unchanged: int = 0
    metadata_updated: int = 0
    removed: int = 0
    failed: int = 0
    pruned: int = 0
    quarantined: int = 0  # also counted in ``failed``
    skipped_quarantined: int = 0  # same content as when quarantined: not embedded again
    prune_refused: int = 0
    tokens: int = 0
    seconds: float = 0.0
    complete: bool = False
    stopped_by_budget: bool = False
    locked: bool = False


Item = Content | Tombstone | MalformedRecord
_Write = tuple[DocMeta, str, list[Chunk]]
_FAILED = object()
# Never per-record: 408/409/425/429 are retried by the Embedder, and 401/403/404 mean a
# bad key or endpoint, which would otherwise mark every record on the page as failed.
_TRANSIENT_4XX = {401, 403, 404, 408, 409, 425, 429}


_BREAKER_KEYS = ("abort_position", "abort_count", "abort_quarantine_failed")
_CLEAR_BREAKER = ("DELETE FROM sync_state WHERE key IN"
                  " ('abort_position', 'abort_count', 'abort_quarantine_failed')")


def _connection_lost(conn: psycopg.AsyncConnection, exc: BaseException) -> bool:
    """Never quarantined: the session is gone, not the record at fault. Other psycopg
    errors (an OperationalError such as ProgramLimitExceeded included) are per-record."""
    return isinstance(exc, psycopg.InterfaceError) or (
        isinstance(exc, psycopg.Error) and (conn.broken or conn.closed))


def _deterministic(exc: httpx.HTTPStatusError) -> bool:
    """A 4xx the provider will repeat for the same input (bad request, too long, ...)."""
    status = exc.response.status_code
    return 400 <= status < 500 and status not in _TRANSIENT_4XX


@dataclass
class _Quarantine:
    """One quarantine run of a tripped page."""
    aborts: int
    to_write: int  # the all-fail guard's denominator: to-write records only
    hashes: dict[uuid.UUID, str]  # kept in sync_failures for a set-aside record (skip key)
    ids: list[uuid.UUID] = field(default_factory=list)  # the records set aside


class _BreakerOpen(Exception):
    """The page's quarantine run failed for every record; it is refused until it changes."""


class _QuarantineFailed(RuntimeError):
    """Every to-write record of a quarantined page (two or more) failed: systemic."""


class _PruneRefused(Exception):
    """The bootstrap's end would prune more documents than the ceiling allows."""

    def __init__(self, count: int, ceiling: int) -> None:
        super().__init__(count, ceiling)
        self.count = count
        self.ceiling = ceiling


def _page_key(page: list[Item]) -> str:
    """The breaker's page identity: every item in order (id plus content hash, or its
    kind), so pages that merely share a first id never alias; new content gets a fresh try."""
    parts = []
    for it in page:
        if isinstance(it, Tombstone):
            tag = "removed"
        elif isinstance(it, MalformedRecord):
            tag = "malformed"
        else:
            tag = _hash(it) or "no-content"
        parts.append(f"{it.doc_id}:{tag}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _short(text: str) -> str:
    return " ".join(text.replace("\x00", "").split())[:200]


def _quarantined(aborts: int, error: str) -> str:
    return f"quarantined after {aborts} page aborts: {error}"


def _error(exc: Exception) -> str:
    return _short(f"{type(exc).__name__}: {exc}")


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
                 embedder: Embedder, report: SyncReport, state: dict[str, str],
                 allow_prune: int | None = None) -> None:
        self.conn = conn
        self.allow_prune = allow_prune
        self.docext = docext
        self.embedder = embedder
        self.report = report
        self.state = state
        self.names: dict[str, str] = {}
        self.refreshed = False
        self.tokens0 = embedder.tokens_used
        # Bootstrap tokens committed by earlier attempts (the cap is cumulative there).
        self.base_tokens = int(state.get("bootstrap_tokens", 0))

    def run_tokens(self) -> int:
        return self.embedder.tokens_used - self.tokens0

    def budget_tokens(self) -> int:
        """What ``--max-tokens`` compares against: the bootstrap's running total across
        attempts, or this run's tokens in incremental mode."""
        if self.report.mode == "bootstrap":
            return self.base_tokens + self.run_tokens()
        return self.run_tokens()

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

    async def _embed(self, writes: list[_Write]
                     ) -> tuple[list[_Write], list[list[float]], list[tuple[uuid.UUID, str]]]:
        """One embed call for the page. A deterministic 4xx is isolated per record so one
        rejected article cannot stall the sync; anything else propagates (page aborts). A
        canary text tells a per-record rejection from a systemic one (wrong model name,
        bad ``output_dimension``): if it is rejected too, the page aborts."""
        texts = [c.text for _, _, chunks in writes for c in chunks]
        if not texts:
            return writes, [], []
        try:
            return writes, await self.embedder.embed(texts, input_type="document"), []
        except httpx.HTTPStatusError as exc:
            if not _deterministic(exc):
                raise
            logger.warning("page embed rejected (%s); embedding its records one by one",
                           exc.response.status_code)
        # Earlier batches of the page call may be billed: reserve the re-embed and canary too.
        await self._reserve_tokens([*texts, CANARY_TEXT])
        kept: list[_Write] = []
        vectors: list[list[float]] = []
        failures: list[tuple[uuid.UUID, str]] = []
        for w in writes:
            try:
                vecs = await self.embedder.embed([c.text for c in w[2]], input_type="document")
            except httpx.HTTPStatusError as exc:
                if not _deterministic(exc):
                    raise
                failures.append((w[0].doc_id, f"embed: {exc.response.status_code} "
                                              f"{_short(exc.response.text)}"))
                continue
            kept.append(w)
            vectors.extend(vecs)
        if not kept and len(writes) >= 2:
            # Every record rejected looks systemic, not per-article: abort the page rather
            # than mark it all failed and move the watermark past it.
            raise RuntimeError(f"embed rejected all {len(writes)} records on the page "
                               f"({failures[0][1]})")
        if failures:
            await self._canary()
        return kept, vectors, failures

    async def flush(self, page: list[Item]) -> None:
        """Embed and write one page in a single transaction. A transient error writes
        nothing; deterministic per-record failures go to ``sync_failures`` and the page
        still commits. A page that tripped the breaker runs record by record (quarantine)."""
        if not page:
            return
        aborts = int(self.state.get("abort_count", 0))
        matches = self._matches_breaker(page)
        if matches and self.state.get("abort_quarantine_failed"):
            raise _BreakerOpen(
                f"page starting at {page[0].doc_id} failed for every record in quarantine "
                f"mode after {aborts} aborts; refusing to embed it again until it changes "
                "upstream. Inspect the errors logged by the earlier runs, fix the cause, then "
                "clear the breaker: DELETE FROM sync_state WHERE key IN ('abort_position',"
                "'abort_count','abort_quarantine_failed')")
        # A tripped page runs once in quarantine mode instead of blocking the feed.
        quarantine = aborts >= BREAKER_LIMIT and matches
        ids = [it.doc_id for it in page]
        stored = await stored_hashes(self.conn, ids)
        cur = await self.conn.execute(
            "SELECT doc_id, content_hash FROM sync_failures WHERE doc_id = ANY(%s)"
            " AND content_hash IS NOT NULL", (ids,))
        held = {doc_id: h for doc_id, h in await cur.fetchall()}
        skipped = 0
        hashes: dict[uuid.UUID, str] = {}  # unchanged records' hashes (quarantine skip key)
        deletes: list[uuid.UUID] = []
        unchanged: list[DocMeta] = []
        failures: list[tuple[uuid.UUID, str]] = []
        writes: list[_Write] = []
        for it in page:
            if isinstance(it, Tombstone):
                deletes.append(it.doc_id)
                continue
            if isinstance(it, MalformedRecord):
                failures.append((it.doc_id, f"malformed: {_short(it.reason)}"))
                continue
            meta = DocMeta(doc_id=it.doc_id, vendor=it.vendor, product=it.product,
                           source=await self._source_name(it.source_id),
                           source_id=it.source_id, title=it.title, url=it.url)
            h = _hash(it)
            # Postgres text rejects NUL; strip it so it never becomes a DataError.
            markdown = it.markdown.replace("\x00", "") if it.markdown is not None else None
            if h is not None and held.get(it.doc_id) == h:
                skipped += 1  # quarantined with this content: its sync_failures row stays
            elif h is not None and stored.get(it.doc_id) == h:
                unchanged.append(meta)
                hashes[it.doc_id] = h
            elif markdown is None or not markdown.strip():
                failures.append((it.doc_id, "missing markdown"))
            elif not (chunks := chunk_markdown(it.title or "", markdown)):
                failures.append((it.doc_id, "markdown yields 0 chunks"))
            else:
                assert h is not None  # markdown is present
                writes.append((meta, h, chunks))
        unchanged = await self._keep_stored_source(unchanged)
        reserved = await self._reserve_tokens([c.text for _, _, cs in writes for c in cs])
        vectors: list[list[float]] = []
        q = _Quarantine(aborts, len(writes),
                        hashes | {meta.doc_id: h for meta, h, _ in writes})
        quarantined = q.ids
        try:
            if quarantine:
                writes, embed_failures = await self._embed_each(writes, vectors, aborts)
                failures += embed_failures
                quarantined += [doc_id for doc_id, _ in embed_failures]
                async with self.conn.transaction():
                    counts = await self._apply(page, deletes, unchanged, writes, vectors,
                                               failures, isolate=True, quarantine=q)
            else:
                writes, vectors, embed_failures = await self._embed(writes)
                failures += embed_failures
                counts = await self._apply_page(page, deletes, unchanged, writes, vectors,
                                                failures)
        except Exception as exc:
            failed = isinstance(exc, _QuarantineFailed)
            if vectors or reserved or failed:
                await self._record_abort(page, embedded=bool(vectors), quarantine_failed=failed)
            raise
        if self._matches_breaker(page):
            self._clear_breaker()
        if quarantined:
            shown = ", ".join(str(d) for d in quarantined[:10])
            more = f" +{len(quarantined) - 10} more" if len(quarantined) > 10 else ""
            logger.error("page %s..%s aborted after embedding %d times; quarantined %d "
                         "record(s) into sync_failures and committed the rest: %s%s. They are "
                         "retried when their content changes upstream.", page[0].doc_id,
                         page[-1].doc_id, aborts, len(quarantined), shown, more)
        r = self.report
        r.quarantined += len(quarantined)
        r.skipped_quarantined += skipped
        r.removed += counts["removed"]
        r.unchanged += counts["unchanged"]
        r.metadata_updated += counts["metadata_updated"]
        r.failed += counts["failed"]
        r.written += counts["written"]
        r.chunks_reused += counts["chunks_reused"]
        logger.info("%s page %s..%s committed: written=%d chunks_reused=%d unchanged=%d "
                    "removed=%d failed=%d tokens=%d", r.mode, page[0].doc_id, page[-1].doc_id,
                    counts["written"], counts["chunks_reused"], counts["unchanged"],
                    counts["removed"], counts["failed"], self.budget_tokens())

    async def _apply_page(self, page: list[Item], deletes: list[uuid.UUID],
                          unchanged: list[DocMeta], writes: list[_Write],
                          vectors: list[list[float]], failures: list[tuple[uuid.UUID, str]]
                          ) -> dict[str, int]:
        """The page in one transaction; on a DataError, again with a savepoint per item."""
        try:
            async with self.conn.transaction():
                return await self._apply(page, deletes, unchanged, writes, vectors,
                                         failures, isolate=False)
        except psycopg.DataError as exc:
            # The transaction rolled back. Redo it with a savepoint per item, reusing the
            # vectors, so only the offending records are set aside.
            logger.warning("page %s..%s hit a data error (%s); applying records one by "
                           "one", page[0].doc_id, page[-1].doc_id, _short(str(exc)))
            async with self.conn.transaction():
                return await self._apply(page, deletes, unchanged, writes, vectors,
                                         failures, isolate=True)

    async def _embed_each(self, writes: list[_Write], vectors: list[list[float]],
                          aborts: int) -> tuple[list[_Write], list[tuple[uuid.UUID, str]]]:
        """Quarantine mode: one embed call per record, appending to ``vectors`` as it goes
        (so a later abort still counts as post-embed). Only a deterministic 4xx sets a
        record aside, and only once the canary shows it is per-record; anything else
        propagates (the page aborts, nothing quarantined)."""
        kept: list[_Write] = []
        failures: list[tuple[uuid.UUID, str]] = []
        for w in writes:
            try:
                vecs = await self.embedder.embed([c.text for c in w[2]], input_type="document")
            except httpx.HTTPStatusError as exc:
                if not _deterministic(exc):
                    raise
                failures.append((w[0].doc_id, _quarantined(
                    aborts, f"embed: {exc.response.status_code} {_short(exc.response.text)}")))
                continue
            kept.append(w)
            vectors.extend(vecs)
        if not kept and len(writes) >= 2:
            # Systemic, not per-record: write nothing; the page is refused until it changes.
            raise _QuarantineFailed(f"quarantine: every record on the page failed to embed "
                                    f"({failures[0][1]})")
        if failures:
            await self._canary()
        return kept, failures

    async def _canary(self) -> None:
        """After a per-record rejection: a deterministic rejection of the canary text too
        means the 4xx is systemic (the page aborts); a transient failure propagates."""
        try:
            await self.embedder.embed([CANARY_TEXT], input_type="document")
        except httpx.HTTPStatusError as exc:
            if _deterministic(exc):
                raise RuntimeError(
                    f"embed rejected the canary text too ({exc.response.status_code} "
                    f"{_short(exc.response.text)}): systemic, not per-record") from exc
            raise

    def _matches_breaker(self, page: list[Item]) -> bool:
        """``abort_position`` is ``n:<page key of the first n items>``: a page matches when
        its first n items hash the same, so a partial last page that new arrivals grow (or
        push past page_size) is still the same page. Any other value never matches. Once the
        page's quarantine has failed only the exact page matches: growth is a fresh try."""
        n_text, sep, stored = self.state.get("abort_position", "").partition(":")
        if not sep or not n_text.isdigit() or int(n_text) < 1 or len(page) < int(n_text):
            return False
        if self.state.get("abort_quarantine_failed") and len(page) != int(n_text):
            return False
        return _page_key(page[:int(n_text)]) == stored

    def _clear_breaker(self) -> None:
        for key in _BREAKER_KEYS:
            self.state.pop(key, None)

    async def _reserve_tokens(self, texts: list[str]) -> bool:
        """Bootstrap only: commit ``bootstrap_tokens`` plus a chars/4 estimate of the texts
        (Voyage counts ~0.84x that) before embedding them, so an abort can never
        under-count."""
        if self.report.mode != "bootstrap" or not texts:
            return False
        estimate = sum(len(t) // 4 for t in texts)
        async with self.conn.transaction():
            await _set_state(self.conn, "bootstrap_tokens",
                             str(self.budget_tokens() + estimate))
        return True

    async def _record_abort(self, page: list[Item], *, embedded: bool,
                            quarantine_failed: bool = False) -> None:
        """Outside the rolled-back page transaction, so it survives the abort. Only a page
        that was embedded counts against the breaker (an embed outage must not trip it);
        in bootstrap mode the actual token total replaces the reservation either way. A
        failed quarantine marks the breaker so the page is refused until it changes."""
        bootstrap = self.report.mode == "bootstrap"
        first = str(page[0].doc_id)
        # The same page, possibly grown, counts on; store the whole aborted page as the new
        # prefix so later growth still matches and a fix to any of its items gets a fresh try.
        count = (int(self.state.get("abort_count", 0)) + 1
                 if self._matches_breaker(page) else 1)
        position = f"{len(page)}:{_page_key(page)}"
        try:
            async with self.conn.transaction():
                # A failed quarantine records this page even if nothing was embedded, so the
                # exact-page refusal fires on the next run.
                if embedded or quarantine_failed:
                    await _set_state(self.conn, "abort_position", position)
                    await _set_state(self.conn, "abort_count", str(count))
                if quarantine_failed:
                    await _set_state(self.conn, "abort_quarantine_failed", "1")
                elif embedded:  # a different page (or a fresh try) starts unmarked
                    await self.conn.execute(
                        "DELETE FROM sync_state WHERE key = 'abort_quarantine_failed'")
                if bootstrap:  # the tokens were spent all the same
                    await _set_state(self.conn, "bootstrap_tokens", str(self.budget_tokens()))
        except psycopg.Error:
            # The reservation stands: an over-count of at most one page, never an under-count.
            logger.warning("could not record the abort of page %s", first, exc_info=True)
            return
        if embedded or quarantine_failed:
            self.state.update(abort_position=position, abort_count=str(count))
        if quarantine_failed:
            self.state["abort_quarantine_failed"] = "1"
        elif embedded:
            self.state.pop("abort_quarantine_failed", None)

    async def _keep_stored_source(self, unchanged: list[DocMeta]) -> list[DocMeta]:
        """An unresolved source name must not blank a stored one for the same source."""
        unresolved = [m.doc_id for m in unchanged if m.source is None and m.source_id]
        if not unresolved:
            return unchanged
        cur = await self.conn.execute(
            "SELECT doc_id, source, source_id FROM documents WHERE doc_id = ANY(%s)",
            (unresolved,))
        stored = {d: (s, sid) for d, s, sid in await cur.fetchall()}
        out = []
        for m in unchanged:
            src, sid = stored.get(m.doc_id, (None, None))
            out.append(replace(m, source=src) if m.source is None and sid == m.source_id
                       and src is not None else m)
        return out

    async def _apply(self, page: list[Item], deletes: list[uuid.UUID],
                     unchanged: list[DocMeta], writes: list[_Write],
                     vectors: list[list[float]], failures: list[tuple[uuid.UUID, str]], *,
                     isolate: bool,
                     quarantine: _Quarantine | None = None) -> dict[str, int]:
        """In quarantine mode any record (delete, metadata update or write) whose operation
        raises anything but a lost connection is set aside (``quarantine.ids``; its content
        hash, if any, kept in ``sync_failures``) rather than aborting the page."""
        conn = self.conn
        failures = list(failures)
        settled: list[uuid.UUID] = []
        counts = dict.fromkeys(
            ("removed", "unchanged", "metadata_updated", "written", "chunks_reused"), 0)

        async def one(doc_id: uuid.UUID, op: Callable[[], Awaitable[object]]) -> object:
            if not isolate:
                result = await op()
            else:
                try:
                    async with conn.transaction():  # a savepoint
                        result = await op()
                except Exception as exc:
                    data_error = isinstance(exc, psycopg.DataError)
                    if _connection_lost(conn, exc) or not (data_error or quarantine):
                        raise
                    reason = f"db: {_short(str(exc))}" if data_error else _error(exc)
                    if quarantine:
                        reason = _quarantined(quarantine.aborts, reason)
                        quarantine.ids.append(doc_id)
                    failures.append((doc_id, reason))
                    return _FAILED
            settled.append(doc_id)
            return result

        for doc_id in deletes:
            res = await one(doc_id, partial(delete_document, conn, doc_id))
            if res is not _FAILED:
                counts["removed"] += bool(res)
        for meta in unchanged:
            res = await one(meta.doc_id, partial(update_metadata, conn, meta))
            if res is not _FAILED:
                counts["unchanged"] += 1
                counts["metadata_updated"] += bool(res)
        at = 0
        for meta, h, chunks in writes:
            vecs = vectors[at:at + len(chunks)]
            at += len(chunks)
            res = await one(meta.doc_id, partial(write_document, conn, meta, h, chunks, vecs))
            if res is not _FAILED:
                counts["written"] += 1
                counts["chunks_reused"] += cast(int, res)
        if quarantine and quarantine.to_write >= 2 and not counts["written"]:
            # Systemic, not per-record: roll the page back; it is refused until it changes.
            raise _QuarantineFailed(f"quarantine: every record on the page failed "
                                    f"({failures[-1][1]})")
        await conn.execute("DELETE FROM sync_failures WHERE doc_id = ANY(%s)", (settled,))
        # Quarantined records keep the hash they failed with (none for a tombstone).
        held = ({d: quarantine.hashes.get(d) for d in quarantine.ids} if quarantine else {})
        for doc_id, reason in failures:
            await conn.execute(
                "INSERT INTO sync_failures(doc_id, reason, content_hash) VALUES (%s, %s, %s)"
                " ON CONFLICT (doc_id) DO UPDATE SET reason=EXCLUDED.reason,"
                " content_hash=EXCLUDED.content_hash, failed_at=now()",
                (doc_id, reason, held.get(doc_id)))
        counts["failed"] = len(failures)
        if self._matches_breaker(page):
            # Only the page that aborted resets the breaker: an incremental run replays
            # the unchanged pages before it, and their commits must not reset its count.
            await conn.execute(_CLEAR_BREAKER)
        if self.report.mode == "bootstrap":
            # Same transaction as the data: the watermark can never run ahead of it.
            await conn.execute("INSERT INTO sync_seen(doc_id) SELECT unnest(%s::uuid[])"
                               " ON CONFLICT DO NOTHING", ([it.doc_id for it in page],))
            await _set_state(conn, "bootstrap_after", str(page[-1].doc_id))
            await _set_state(conn, "bootstrap_tokens", str(self.budget_tokens()))
        return counts

    async def finish(self, line: CursorLine) -> None:
        async with self.conn.transaction():
            if self.report.mode == "bootstrap":
                # The contract: incremental pulls start from the FIRST bootstrap_start.
                since = self.state.get("bootstrap_next_since")
                if since is None:
                    raise RuntimeError("bootstrap ended with no stored bootstrap_start")
                # Documents the bootstrap never served were deleted upstream after the
                # local ingest; their tombstones predate the watermark, so prune them here.
                await self._check_prune()
                cur = await self.conn.execute(
                    "DELETE FROM documents d WHERE NOT EXISTS"
                    " (SELECT 1 FROM sync_seen s WHERE s.doc_id = d.doc_id)")
                self.report.pruned = cur.rowcount
                await self.conn.execute("TRUNCATE sync_seen")
                await _set_state(self.conn, "cursor", since)
                await _set_state(self.conn, "mode", "incremental")
                await self.conn.execute(
                    "DELETE FROM sync_state WHERE key IN"
                    " ('bootstrap_after', 'bootstrap_next_since', 'bootstrap_tokens')")
            else:
                await _set_state(self.conn, "cursor", line.next_since)
            await self.conn.execute(_CLEAR_BREAKER)  # the stream has moved past every page
        self._clear_breaker()

    async def _check_prune(self) -> None:
        """Raise (rolling the finish back) when the prune exceeds the ceiling and the
        ``allow_prune`` override does not cover it."""
        cur = await self.conn.execute(
            "SELECT count(*) FILTER (WHERE NOT EXISTS (SELECT 1 FROM sync_seen s"
            " WHERE s.doc_id = d.doc_id)), count(*) FROM documents d")
        row = await cur.fetchone()
        assert row is not None
        count, total = row
        ceiling = max(PRUNE_FLOOR, math.ceil(PRUNE_SHARE * total))
        if count > ceiling and (self.allow_prune is None or count > self.allow_prune):
            raise _PruneRefused(count, ceiling)


async def sync(conn: psycopg.AsyncConnection, *, docext: DocExtractor, embedder: Embedder,
               page_size: int = 200, max_tokens: int | None = None,
               allow_prune: int | None = None) -> SyncReport:
    started = time.monotonic()
    tokens0 = embedder.tokens_used
    report = SyncReport(mode="bootstrap")
    cur = await conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
    got = await cur.fetchone()
    if not (got and got[0]):
        if "cursor" in await _load_state(conn):
            report.mode = "incremental"
        report.locked = True
        logger.warning("sync advisory lock is held by another run; nothing done")
        return report
    page: list[Item] = []
    try:  # everything after the lock is acquired, so every path releases it
        state = await _load_state(conn)
        if "cursor" in state:
            report.mode = "incremental"
        run = _Run(conn, docext, embedder, report, state, allow_prune)

        def over_budget() -> bool:
            return max_tokens is not None and run.budget_tokens() >= max_tokens

        if report.mode == "bootstrap" and over_budget():
            # Earlier attempts already spent the cap: stop before any feed or embed call.
            logger.warning("bootstrap has spent %d embedding tokens, at or over the cap of "
                           "%s; nothing done", run.budget_tokens(), max_tokens)
            report.stopped_by_budget = True
            return report
        await run.load_names()
        if report.mode == "bootstrap":
            stream = docext.delta(bootstrap_after=state.get("bootstrap_after"))
        else:
            stream = docext.delta(since=state["cursor"])
        async with aclosing(stream) as items:
            async for item in items:
                if isinstance(item, BootstrapStart):
                    if report.mode == "bootstrap" and "bootstrap_next_since" not in state:
                        async with conn.transaction():  # a fresh bootstrap: forget old ids
                            await conn.execute("TRUNCATE sync_seen")
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
                if report.mode == "bootstrap" and "bootstrap_next_since" not in state:
                    # Without the first watermark the bootstrap can never hand over to
                    # incremental pulls; write nothing and retry from scratch.
                    raise RuntimeError("bootstrap stream did not start with bootstrap_start")
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
    except _BreakerOpen as exc:
        logger.error("sync refused: %s", exc)
    except _PruneRefused as exc:
        report.prune_refused = exc.count
        logger.error("bootstrap finish refused: it would prune %d documents the stream never "
                     "served, over the ceiling of %d; nothing pruned, cursor not set. If the "
                     "count is expected, re-run with: vendorkb sync --allow-prune %d",
                     exc.count, exc.ceiling, exc.count)
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
