import hashlib
import json
import logging
import re
import uuid

import httpx
import psycopg

from vendorkb import sync as sync_mod
from vendorkb.chunker import chunk_markdown
from vendorkb.clients import DocExtractor, Embedder
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect, init_schema
from vendorkb.feed import parse_line
from vendorkb.sync import LOCK_KEY, sync

A = [uuid.UUID(int=n) for n in range(1, 6)]  # stream order == id order


def _settings() -> VendorkbSettings:
    return VendorkbSettings(
        _env_file=None, embed_base_url="http://e/v1", embed_dim=768, embed_max_batch=128,
        embed_input_type=False, embed_output_dimension=None, docext_base_url="http://d")


def _md(body: str) -> str:
    return f"# Title\n\n{body}"


def _rec(doc_id, *, change="added", body=None, title="Title", source_id=None,
         content_hash="auto", markdown="auto") -> dict:
    md = _md(body or f"body of {doc_id}") if markdown == "auto" else markdown
    h = hashlib.sha256(md.encode()).hexdigest() if content_hash == "auto" else content_hash
    return {"id": str(doc_id), "change_type": change, "vendor": "V", "product": "P",
            "title": title, "source_url": f"https://x/{doc_id}", "content_hash": h,
            "content_markdown": md, "source_id": str(source_id) if source_id else None}


def _tomb(doc_id) -> dict:
    return {"id": str(doc_id), "change_type": "removed", "run_id": None}


def _start(since):
    return {"control": "bootstrap_start", "next_since": since}


def _cursor(since, count=0):
    return {"control": "cursor", "next_since": since, "count": count}


class FakeDocExt:
    """Serves one queued NDJSON body per delta request and records the params."""

    def __init__(self, bodies: list[list[dict]], sources: dict[str, str] | None = None):
        self.bodies = list(bodies)
        self.params: list[dict] = []
        self.products_calls = 0
        self.sources = sources or {}

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/articles/delta":
            self.params.append(dict(req.url.params))
            lines = self.bodies.pop(0)
            return httpx.Response(200, text="".join(
                (x if isinstance(x, str) else json.dumps(x)) + "\n" for x in lines))
        if req.url.path == "/api/products":
            self.products_calls += 1
            return httpx.Response(200, json={"products": [{"id": "p1"}]})
        if req.url.path == "/api/sources":
            return httpx.Response(200, json={"sources": [
                {"id": k, "name": v} for k, v in self.sources.items()]})
        return httpx.Response(404)

    def client(self) -> DocExtractor:
        return DocExtractor(_settings(), transport=httpx.MockTransport(self.handler))


class FakeEmbedder:
    """Counts requests and texts; ``status`` fails every request, ``reject`` returns 400
    for any request containing that marker text, ``canary_status`` fails the canary."""

    def __init__(self, status: int = 200, reject: str | None = None,
                 canary_status: int = 200, *, max_batch: int = 128,
                 fail_after: int | None = None, on_call=None):
        self.status = status
        self.reject = reject
        self.canary_status = canary_status
        self.max_batch = max_batch
        self.fail_after = fail_after  # requests after this many return 503
        self.on_call = on_call
        self.calls = 0
        self.texts: list[str] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.on_call:
            self.on_call()
        if self.fail_after is not None and self.calls > self.fail_after:
            return httpx.Response(503, text="unavailable")
        if self.status != 200:
            return httpx.Response(self.status, text="bad request")
        inputs = json.loads(req.content)["input"]
        if self.canary_status != 200 and inputs == [sync_mod.CANARY_TEXT]:
            return httpx.Response(self.canary_status, text="canary failed")
        if self.reject and any(self.reject in t for t in inputs):
            return httpx.Response(400, text="input too long")
        self.texts.extend(inputs)
        return httpx.Response(200, json={
            "data": [{"index": i, "embedding": [0.0] * 767 + [1.0]} for i in range(len(inputs))],
            "usage": {"total_tokens": len(inputs)}})

    def client(self) -> Embedder:
        settings = _settings().model_copy(update={"embed_max_batch": self.max_batch})
        return Embedder(settings, transport=httpx.MockTransport(self.handler),
                        retry_base_delay=0.0)


async def _fresh(dsn):
    conn = await connect(dsn)
    await init_schema(conn, embed_dim=768)
    await conn.execute("TRUNCATE documents, chunks, sync_state, sources, sync_failures,"
                       " sync_seen CASCADE")
    return conn


async def _run(conn, docext: FakeDocExt, emb: FakeEmbedder, **kw):
    async with docext.client() as d, emb.client() as e:
        return await sync(conn, docext=d, embedder=e, **{"page_size": 2, **kw})


async def _state(conn) -> dict[str, str]:
    cur = await conn.execute("SELECT key, value FROM sync_state")
    return dict(await cur.fetchall())


async def _failures(conn) -> dict[uuid.UUID, str]:
    cur = await conn.execute("SELECT doc_id, reason FROM sync_failures")
    return dict(await cur.fetchall())


async def _lock_free(dsn) -> bool:
    """Session advisory locks are re-entrant, so probe from a different connection."""
    other = await connect(dsn)
    try:
        cur = await other.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
        got = (await cur.fetchone())[0]
        if got:
            await other.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
        return got
    finally:
        await other.close()


async def _doc_ids(conn) -> set[uuid.UUID]:
    cur = await conn.execute("SELECT doc_id FROM documents")
    return {r[0] for r in await cur.fetchall()}


async def test_full_bootstrap_writes_all_and_switches_to_incremental(vkb_dsn):
    # stream: bootstrap_start(S0), 3 added, cursor(S9). Assert 3 documents, the cursor
    # equals S0 (not S9), mode=incremental, bootstrap_after deleted, report.complete.
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]), _rec(A[2]), _cursor("S9", 3)]])
        report = await _run(conn, fd, FakeEmbedder())
        assert report.complete and report.mode == "bootstrap"
        assert (report.records, report.written) == (3, 3)
        assert await _doc_ids(conn) == set(A[:3])
        state = await _state(conn)
        assert state["cursor"] == "S0"
        assert state["mode"] == "incremental"
        assert "bootstrap_after" not in state
        assert fd.params == [{}]
    finally:
        await conn.close()


async def test_bootstrap_drop_resumes_after_last_committed_page(vkb_dsn):
    # attempt 1: bootstrap_start(S0), a1, a2, a3, then EOF (no cursor line). Assert
    # a1, a2 written, a3 NOT written, bootstrap_after == a2, complete False.
    # attempt 2: bootstrap_start(S1), a3, cursor(S9). Assert request 2 has
    # bootstrap_after=a2 and no since; cursor == S0 (the first watermark kept).
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]), _rec(A[2])],
                         [_start("S1"), _rec(A[2]), _cursor("S9", 1)]])
        first = await _run(conn, fd, FakeEmbedder())
        assert not first.complete
        assert await _doc_ids(conn) == set(A[:2])
        assert (await _state(conn))["bootstrap_after"] == str(A[1])
        second = await _run(conn, fd, FakeEmbedder())
        assert second.complete
        assert fd.params[1] == {"bootstrap_after": str(A[1])}
        assert await _doc_ids(conn) == set(A[:3])
        assert (await _state(conn))["cursor"] == "S0"
    finally:
        await conn.close()


async def test_incremental_update_unchanged_replay_and_tombstone(vkb_dsn):
    # seeded by a bootstrap. Incremental: updated a1 (new hash), added a2 (same hash),
    # removed a3, cursor(S10). Assert texts embedded only for a1, a3 deleted with its
    # chunks cascaded, cursor == S10, request had since=<old cursor>.
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([
            [_start("S0"), _rec(A[0]), _rec(A[1]), _rec(A[2]), _cursor("S9", 3)],
            [_rec(A[0], change="updated", body="rewritten alpha"), _rec(A[1]), _tomb(A[2]),
             _cursor("S10", 3)]])
        await _run(conn, fd, FakeEmbedder())
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb)
        assert report.complete and report.mode == "incremental"
        assert emb.texts and all("rewritten alpha" in t for t in emb.texts)
        assert (report.written, report.unchanged, report.removed) == (1, 1, 1)
        assert await _doc_ids(conn) == set(A[:2])
        cur = await conn.execute("SELECT count(*) FROM chunks WHERE doc_id=%s", (A[2],))
        assert (await cur.fetchone())[0] == 0
        assert (await _state(conn))["cursor"] == "S10"
        assert fd.params[1] == {"since": "S0"}
    finally:
        await conn.close()


async def test_dropped_incremental_leaves_cursor_unchanged(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)],
                         [_rec(A[1]), _rec(A[2]), _rec(A[3])]])
        await _run(conn, fd, FakeEmbedder())
        report = await _run(conn, fd, FakeEmbedder())
        assert not report.complete
        assert (await _state(conn))["cursor"] == "S0"
        assert A[3] not in await _doc_ids(conn)  # the unfinished page is not written
    finally:
        await conn.close()


async def test_unchanged_hash_updates_metadata_without_embedding(vkb_dsn):
    # the same hash with a new title: the title is updated, 0 texts embedded,
    # report.metadata_updated == 1
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)],
                         [_rec(A[0], change="updated", title="Renamed"), _cursor("S10", 1)]])
        await _run(conn, fd, FakeEmbedder())
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb)
        assert report.complete
        assert emb.texts == []
        assert report.metadata_updated == 1
        cur = await conn.execute("SELECT title FROM documents WHERE doc_id=%s", (A[0],))
        assert (await cur.fetchone())[0] == "Renamed"
    finally:
        await conn.close()


async def test_transient_embed_failure_aborts_without_moving_watermark(vkb_dsn):
    # a 503 that exhausts the Embedder's retries: 0 documents written, bootstrap_after
    # unset, report.complete False, sync_failures empty, lock released, conn usable
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]), _cursor("S9", 2)]])
        report = await _run(conn, fd, FakeEmbedder(status=503))
        assert not report.complete
        assert await _doc_ids(conn) == set()
        state = await _state(conn)
        assert "bootstrap_after" not in state and "cursor" not in state
        assert await _failures(conn) == {}
        assert await _lock_free(vkb_dsn)
        await conn.execute("SELECT 1")
    finally:
        await conn.close()


async def test_embed_400_for_every_record_aborts_the_page(vkb_dsn):
    # every record on a 2-record page 400s: that looks systemic, so the page aborts --
    # nothing written, no sync_failures rows, watermark unmoved
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]), _cursor("S9", 2)]])
        report = await _run(conn, fd, FakeEmbedder(status=400))
        assert not report.complete
        assert await _doc_ids(conn) == set()
        assert await _failures(conn) == {}
        state = await _state(conn)
        assert "bootstrap_after" not in state and "cursor" not in state
    finally:
        await conn.close()


async def test_embed_400_for_a_lone_record_that_the_canary_repeats_aborts(vkb_dsn):
    # one to-embed record and every call 400s (a wrong model name, say): the canary is
    # rejected too, so the 400 is systemic -- the page aborts, nothing marked failed
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)]])
        emb = FakeEmbedder(status=400)
        report = await _run(conn, fd, emb)
        assert not report.complete
        assert emb.calls == 3  # page, record, canary
        assert await _doc_ids(conn) == set()
        assert await _failures(conn) == {}
        state = await _state(conn)
        assert "bootstrap_after" not in state and "cursor" not in state
    finally:
        await conn.close()


async def test_transient_canary_failure_aborts_the_page(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1], body="POISON text"),
                          _cursor("S9", 2)]])
        report = await _run(conn, fd, FakeEmbedder(reject="POISON", canary_status=503))
        assert not report.complete
        assert await _doc_ids(conn) == set()
        assert await _failures(conn) == {}
        assert "bootstrap_after" not in await _state(conn)
    finally:
        await conn.close()


async def test_embed_401_aborts_the_page(vkb_dsn):
    # an expired key: transient, so nothing is written and nothing is marked failed
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)]])
        emb = FakeEmbedder(status=401)
        report = await _run(conn, fd, emb)
        assert not report.complete
        assert emb.calls == 1  # not retried, not isolated
        assert await _doc_ids(conn) == set()
        assert await _failures(conn) == {}
        assert "bootstrap_after" not in await _state(conn)
    finally:
        await conn.close()


async def test_embed_4xx_for_one_record_isolates_it(vkb_dsn):
    # 400 only for requests containing the marker: that record goes to sync_failures,
    # the other record on the page is written, the watermark moves
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1], body="POISON text")]])
        emb = FakeEmbedder(reject="POISON")
        report = await _run(conn, fd, emb)
        # the page, each record, then the canary once (it succeeds: per-record rejection)
        assert emb.calls == 4 and emb.texts.count(sync_mod.CANARY_TEXT) == 1
        assert (report.written, report.failed) == (1, 1)
        assert await _doc_ids(conn) == {A[0]}
        assert list(await _failures(conn)) == [A[1]]
        assert (await _state(conn))["bootstrap_after"] == str(A[1])
    finally:
        await conn.close()


async def test_nul_characters_are_stripped_before_chunking(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0], markdown=_md("null\x00byte here")),
                          _cursor("S9", 1)]])
        report = await _run(conn, fd, FakeEmbedder())
        assert report.complete and report.written == 1
        cur = await conn.execute("SELECT text FROM chunks WHERE doc_id=%s", (A[0],))
        texts = [r[0] for r in await cur.fetchall()]
        assert texts and all("\x00" not in t for t in texts)
        assert any("nullbyte here" in t for t in texts)
    finally:
        await conn.close()


async def test_data_error_isolates_the_record_without_re_embedding(vkb_dsn, monkeypatch):
    # write_document raises DataError for one doc: only it goes to sync_failures, the
    # rest of the page commits, and the page was embedded exactly once
    real = sync_mod.write_document

    async def flaky(conn, meta, *args):
        if meta.doc_id == A[1]:
            raise psycopg.DataError("value out of range")
        await real(conn, meta, *args)

    monkeypatch.setattr(sync_mod, "write_document", flaky)
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1])]])
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb)
        assert emb.calls == 1
        assert (report.written, report.failed) == (1, 1)
        assert await _doc_ids(conn) == {A[0]}
        fails = await _failures(conn)
        assert list(fails) == [A[1]] and fails[A[1]].startswith("db: ")
        assert (await _state(conn))["bootstrap_after"] == str(A[1])
    finally:
        await conn.close()


async def test_zero_chunk_record_goes_to_sync_failures_and_page_commits(vkb_dsn):
    # markdown "   ": the failure row exists, the other record on the page is written,
    # and the watermark moved
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0], markdown="   "), _rec(A[1])]])
        report = await _run(conn, fd, FakeEmbedder())
        assert (report.failed, report.written) == (1, 1)
        cur = await conn.execute("SELECT doc_id, reason FROM sync_failures")
        rows = await cur.fetchall()
        assert [r[0] for r in rows] == [A[0]] and rows[0][1]
        assert await _doc_ids(conn) == {A[1]}
        assert (await _state(conn))["bootstrap_after"] == str(A[1])
    finally:
        await conn.close()


async def test_unknown_source_refreshes_names_once(vkb_dsn):
    # two records with unknown source ids: /api/products called exactly once,
    # documents.source is filled where known and NULL where still unknown
    known, unknown = uuid.uuid4(), uuid.uuid4()
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0], source_id=known),
                          _rec(A[1], source_id=unknown), _rec(A[2], source_id=unknown),
                          _cursor("S9", 3)]],
                        sources={str(known): "Known Guide"})
        report = await _run(conn, fd, FakeEmbedder())
        assert report.complete
        assert fd.products_calls == 1
        cur = await conn.execute("SELECT doc_id, source, source_id FROM documents")
        rows = {r[0]: (r[1], r[2]) for r in await cur.fetchall()}
        assert rows[A[0]] == ("Known Guide", known)
        assert rows[A[1]] == (None, unknown) and rows[A[2]] == (None, unknown)
        cur = await conn.execute("SELECT name FROM sources WHERE source_id=%s", (known,))
        assert (await cur.fetchone())[0] == "Known Guide"
    finally:
        await conn.close()


async def test_max_tokens_stops_after_the_current_page(vkb_dsn):
    # 4 added, page_size 2, max_tokens=1. One page is written, stopped_by_budget,
    # bootstrap_after == the 2nd id, cursor unset
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), *(_rec(a) for a in A[:4]), _cursor("S9", 4)]])
        report = await _run(conn, fd, FakeEmbedder(), max_tokens=1)
        assert report.stopped_by_budget and not report.complete
        assert await _lock_free(vkb_dsn)
        assert report.tokens >= 1
        assert await _doc_ids(conn) == set(A[:2])
        state = await _state(conn)
        assert state["bootstrap_after"] == str(A[1])
        assert "cursor" not in state
    finally:
        await conn.close()


async def test_second_concurrent_run_reports_locked(vkb_dsn):
    # hold pg_advisory_lock(LOCK_KEY) on another connection; sync returns locked=True
    # and makes no feed request
    conn = await _fresh(vkb_dsn)
    holder = await connect(vkb_dsn)
    try:
        await holder.execute("SELECT pg_advisory_lock(%s)", (LOCK_KEY,))
        fd = FakeDocExt([[_start("S0"), _cursor("S9", 0)]])
        report = await _run(conn, fd, FakeEmbedder())
        assert report.locked and not report.complete
        assert fd.params == []
    finally:
        await holder.close()
        await conn.close()


async def test_duplicate_id_in_a_page_applies_in_stream_order(vkb_dsn):
    # page_size 2: a1 updated then a1 removed. The early flush writes the update, the
    # tombstone then deletes it; reversed order would leave a1 present.
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)],
                         [_rec(A[0], change="updated", body="new"), _tomb(A[0]),
                          _cursor("S10", 2)]])
        await _run(conn, fd, FakeEmbedder())
        report = await _run(conn, fd, FakeEmbedder())
        assert report.complete
        assert (report.written, report.removed) == (1, 1)
        assert await _doc_ids(conn) == set()
    finally:
        await conn.close()


async def test_malformed_line_mid_stream_keeps_committed_pages(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]), _rec(A[2]), "{not json",
                          _rec(A[3]), _cursor("S9", 4)]])
        report = await _run(conn, fd, FakeEmbedder())
        assert not report.complete
        assert await _doc_ids(conn) == set(A[:2])
        assert (await _state(conn))["bootstrap_after"] == str(A[1])
        assert await _lock_free(vkb_dsn)
    finally:
        await conn.close()


async def test_missing_markdown_goes_to_sync_failures(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0], markdown=None, content_hash="h1"),
                          _rec(A[1]), _cursor("S9", 2)]])
        report = await _run(conn, fd, FakeEmbedder())
        assert report.complete and (report.failed, report.written) == (1, 1)
        assert await _failures(conn) == {A[0]: "missing markdown"}
        assert await _doc_ids(conn) == {A[1]}
    finally:
        await conn.close()


async def test_bootstrap_without_bootstrap_start_writes_nothing(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_rec(A[0]), _rec(A[1]), _cursor("S9", 2)]])
        report = await _run(conn, fd, FakeEmbedder())
        assert not report.complete
        assert await _doc_ids(conn) == set()
        assert await _state(conn) == {}
    finally:
        await conn.close()


async def test_unchanged_doc_keeps_stored_source_when_name_unresolved(vkb_dsn):
    sid = uuid.uuid4()
    conn = await _fresh(vkb_dsn)
    try:
        await _run(conn, FakeDocExt([[_start("S0"), _rec(A[0], source_id=sid),
                                      _cursor("S9", 1)]], sources={str(sid): "Guide"}),
                   FakeEmbedder())
        await conn.execute("TRUNCATE sources")  # the name is no longer resolvable
        fd = FakeDocExt([[_rec(A[0], change="updated", source_id=sid, title="Renamed"),
                          _cursor("S10", 1)]])
        report = await _run(conn, fd, FakeEmbedder())
        assert report.complete and report.metadata_updated == 1
        cur = await conn.execute("SELECT source, title FROM documents WHERE doc_id=%s",
                                 (A[0],))
        assert await cur.fetchone() == ("Guide", "Renamed")
    finally:
        await conn.close()


async def test_full_bootstrap_clears_bootstrap_next_since(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)]])
        await _run(conn, fd, FakeEmbedder())
        assert await _state(conn) == {"cursor": "S0", "mode": "incremental"}
    finally:
        await conn.close()


async def _seed_doc(conn, doc_id, content_hash="h-old") -> None:
    await conn.execute(
        "INSERT INTO documents(doc_id, vendor, content_hash, fetched_at)"
        " VALUES (%s, 'V', %s, now())", (doc_id, content_hash))


async def test_malformed_record_goes_to_sync_failures_and_the_stream_continues(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        bad = {**_rec(A[1]), "vendor": None}
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), bad, _rec(A[2]), _cursor("S9", 3)]])
        report = await _run(conn, fd, FakeEmbedder())
        assert report.complete
        assert (report.records, report.written, report.failed) == (3, 2, 1)
        assert await _doc_ids(conn) == {A[0], A[2]}
        fails = await _failures(conn)
        assert list(fails) == [A[1]] and fails[A[1]].startswith("malformed: ")
    finally:
        await conn.close()


async def test_held_lock_logs_a_warning(vkb_dsn, caplog):
    conn = await _fresh(vkb_dsn)
    holder = await connect(vkb_dsn)
    try:
        await holder.execute("SELECT pg_advisory_lock(%s)", (LOCK_KEY,))
        with caplog.at_level(logging.WARNING, logger="vendorkb.sync"):
            report = await _run(conn, FakeDocExt([]), FakeEmbedder())
        assert report.locked
        assert any(r.levelno == logging.WARNING and "lock" in r.getMessage()
                   for r in caplog.records)
    finally:
        await holder.close()
        await conn.close()


async def test_each_committed_page_logs_progress(vkb_dsn, caplog):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), *(_rec(a) for a in A[:3]), _cursor("S9", 3)]])
        with caplog.at_level(logging.INFO, logger="vendorkb.sync"):
            await _run(conn, fd, FakeEmbedder())
        pages = [r.getMessage() for r in caplog.records
                 if r.levelno == logging.INFO and "committed" in r.getMessage()]
        assert len(pages) == 2
        assert str(A[0]) in pages[0] and str(A[1]) in pages[0] and "written=2" in pages[0]
        assert str(A[2]) in pages[1] and "tokens=" in pages[1]
    finally:
        await conn.close()


async def test_bootstrap_budget_is_cumulative_across_runs(vkb_dsn):
    # run 1 stops on budget after one page; run 2 with the same cap stops before any
    # embed call because the stored total already reaches it. report.tokens is per run.
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), *(_rec(a) for a in A[:4]), _cursor("S9", 4)],
                         [_start("S1"), _rec(A[2]), _rec(A[3]), _cursor("S9", 2)]])
        first = await _run(conn, fd, FakeEmbedder(), max_tokens=1)
        assert first.stopped_by_budget and first.tokens >= 1
        assert int((await _state(conn))["bootstrap_tokens"]) == first.tokens
        emb = FakeEmbedder()
        second = await _run(conn, fd, emb, max_tokens=1)
        assert second.stopped_by_budget and not second.complete
        assert emb.calls == 0 and second.tokens == 0
        assert await _doc_ids(conn) == set(A[:2])
        # a higher cap carries on; finishing the bootstrap deletes the running total
        third = await _run(conn, fd, FakeEmbedder(), max_tokens=first.tokens + 100)
        assert third.complete and third.tokens >= 1
        assert "bootstrap_tokens" not in await _state(conn)
    finally:
        await conn.close()


async def test_incremental_budget_is_per_run(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)],
                         [_rec(A[1]), _rec(A[2]), _rec(A[3]), _cursor("S10", 3)],
                         [_rec(A[3]), _cursor("S11", 1)]])
        await _run(conn, fd, FakeEmbedder())
        first = await _run(conn, fd, FakeEmbedder(), max_tokens=1)
        assert first.stopped_by_budget and await _doc_ids(conn) == set(A[:3])
        second = await _run(conn, fd, FakeEmbedder(), max_tokens=1)
        assert second.complete and A[3] in await _doc_ids(conn)
    finally:
        await conn.close()


def _is_page_key(value: str) -> bool:
    """The breaker keys a page as ``n:<sha256 of its first n items>``, not by its first id."""
    return re.fullmatch(r"[1-9][0-9]*:[0-9a-f]{64}", value) is not None


def _failing_write(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("bug in the write path")
    monkeypatch.setattr(sync_mod, "write_document", boom)


async def test_three_post_embed_aborts_trip_the_breaker_into_quarantine(vkb_dsn, monkeypatch,
                                                                       caplog):
    # run 4 runs the page in quarantine mode (one embed call per record); every write
    # still fails, so that is systemic: the page aborts and the breaker stays tripped
    _failing_write(monkeypatch)
    page = [_start("S0"), _rec(A[0]), _rec(A[1]), _cursor("S9", 2)]
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([page] * 4)
        for n in (1, 2, 3):
            emb = FakeEmbedder()
            report = await _run(conn, fd, emb)
            assert not report.complete and emb.calls == 1
            state = await _state(conn)
            assert _is_page_key(state["abort_position"]) and state["abort_count"] == str(n)
        emb = FakeEmbedder()
        with caplog.at_level(logging.ERROR, logger="vendorkb.sync"):
            report = await _run(conn, fd, emb)
        assert not report.complete and emb.calls == 2
        assert (await _state(conn))["abort_count"] == "4"
        assert await _doc_ids(conn) == set() and await _failures(conn) == {}
        assert await _lock_free(vkb_dsn)
        assert any("quarantine: every record" in r.getMessage() or
                   (r.exc_info and "quarantine: every record" in str(r.exc_info[1]))
                   for r in caplog.records if r.levelno == logging.ERROR)
    finally:
        await conn.close()


async def test_a_committed_page_resets_the_breaker(vkb_dsn, monkeypatch):
    # the page that aborted is the page that commits; the stream then drops, so only the
    # page commit (not finish) can have cleared the keys
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]), _cursor("S9", 2)],
                         [_start("S0"), _rec(A[0]), _rec(A[1])]])
        with monkeypatch.context() as m:
            _failing_write(m)
            await _run(conn, fd, FakeEmbedder())
        assert (await _state(conn))["abort_count"] == "1"
        report = await _run(conn, fd, FakeEmbedder())
        assert not report.complete and report.written == 2
        state = await _state(conn)
        assert "abort_position" not in state and "abort_count" not in state
    finally:
        await conn.close()


async def test_replayed_pages_before_the_failing_page_keep_the_breaker(vkb_dsn, monkeypatch):
    # incremental replay: page 1 (unchanged) commits on every run, page 2 aborts after
    # embedding. The page-1 commits must not reset page 2's count.
    conn = await _fresh(vkb_dsn)
    try:
        await _run(conn, FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]),
                                      _cursor("S9", 2)]]), FakeEmbedder())
        _failing_write(monkeypatch)
        body = [_rec(A[0]), _rec(A[1]), _rec(A[2]), _rec(A[3]), _cursor("S10", 4)]
        fd = FakeDocExt([body] * 4)
        for n in (1, 2, 3):
            emb = FakeEmbedder()
            report = await _run(conn, fd, emb)
            assert not report.complete and emb.calls == 1
            state = await _state(conn)
            assert _is_page_key(state["abort_position"]) and state["abort_count"] == str(n)
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb)
        # quarantine mode (one call per record) proves page 2 still matched; every write
        # fails, so it aborts
        assert not report.complete and emb.calls == 2
        assert (await _state(conn))["cursor"] == "S0"
    finally:
        await conn.close()


async def test_finish_clears_the_breaker(vkb_dsn, monkeypatch):
    # the aborted page is no longer served (the stream changed upstream): the cursor
    # moving past it clears the keys
    conn = await _fresh(vkb_dsn)
    try:
        await _run(conn, FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)]]),
                   FakeEmbedder())
        fd = FakeDocExt([[_rec(A[1]), _rec(A[2]), _cursor("S10", 2)],
                         [_rec(A[0]), _cursor("S11", 1)]])
        with monkeypatch.context() as m:
            _failing_write(m)
            await _run(conn, fd, FakeEmbedder())
        assert (await _state(conn))["abort_count"] == "1"
        report = await _run(conn, fd, FakeEmbedder())
        assert report.complete
        state = await _state(conn)
        assert "abort_position" not in state and "abort_count" not in state
        assert state["cursor"] == "S11"
    finally:
        await conn.close()


async def test_abort_before_embedding_does_not_count(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)]])
        await _run(conn, fd, FakeEmbedder(status=503))
        assert "abort_count" not in await _state(conn)
    finally:
        await conn.close()


async def test_bootstrap_finish_prunes_documents_not_in_the_stream(vkb_dsn):
    gone, kept = uuid.uuid4(), A[0]
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_doc(conn, gone)
        await _seed_doc(conn, kept, content_hash=_rec(kept)["content_hash"])
        # a stale id from an earlier bootstrap must not protect anything
        await conn.execute("INSERT INTO sync_seen(doc_id) VALUES (%s)", (gone,))
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]), _cursor("S9", 2)]])
        report = await _run(conn, fd, FakeEmbedder())
        assert report.complete and report.pruned == 1
        assert await _doc_ids(conn) == {A[0], A[1]}
        cur = await conn.execute("SELECT count(*) FROM sync_seen")
        assert (await cur.fetchone())[0] == 0
    finally:
        await conn.close()


async def test_dropped_bootstrap_does_not_prune(vkb_dsn):
    gone = uuid.uuid4()
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_doc(conn, gone)
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]), _rec(A[2])],
                         [_start("S1"), _rec(A[2]), _cursor("S9", 1)]])
        first = await _run(conn, fd, FakeEmbedder())
        assert not first.complete and first.pruned == 0
        assert gone in await _doc_ids(conn)
        # the resumed attempt keeps what the first one saw: A[0], A[1] are not pruned
        second = await _run(conn, fd, FakeEmbedder())
        assert second.complete and second.pruned == 1
        assert await _doc_ids(conn) == set(A[:3])
    finally:
        await conn.close()


async def test_incremental_never_prunes(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)],
                         [_rec(A[1]), _cursor("S10", 1)]])
        await _run(conn, fd, FakeEmbedder())
        report = await _run(conn, fd, FakeEmbedder())
        assert report.complete and report.pruned == 0
        assert await _doc_ids(conn) == set(A[:2])
    finally:
        await conn.close()


async def _seed_many(conn, n: int) -> None:
    await conn.execute(
        "INSERT INTO documents(doc_id, vendor, content_hash, fetched_at)"
        " SELECT gen_random_uuid(), 'V', 'h-old', now() FROM generate_series(1, %s)", (n,))


async def _doc_count(conn) -> int:
    cur = await conn.execute("SELECT count(*) FROM documents")
    return (await cur.fetchone())[0]


async def test_bootstrap_prune_over_the_ceiling_is_refused(vkb_dsn, caplog):
    # an empty bootstrap stream that still ends with a cursor line would prune all 600
    # restored documents: refused, everything rolled back
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_many(conn, 600)
        fd = FakeDocExt([[_start("S0"), _cursor("S9", 0)]])
        with caplog.at_level(logging.ERROR, logger="vendorkb.sync"):
            report = await _run(conn, fd, FakeEmbedder())
        assert not report.complete
        assert (report.prune_refused, report.pruned) == (600, 0)
        assert await _doc_count(conn) == 600
        state = await _state(conn)
        assert "cursor" not in state and state["bootstrap_next_since"] == "S0"
        assert state["mode"] == "bootstrap"  # sync_seen kept: see the re-run test
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("600" in m and "500" in m and "--allow-prune" in m for m in msgs)
        assert await _lock_free(vkb_dsn)
    finally:
        await conn.close()


async def test_allow_prune_overrides_the_ceiling(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_many(conn, 600)
        fd = FakeDocExt([[_start("S0"), _cursor("S9", 0)]])
        report = await _run(conn, fd, FakeEmbedder(), allow_prune=600)
        assert report.complete
        assert (report.pruned, report.prune_refused) == (600, 0)
        assert await _doc_count(conn) == 0
        assert (await _state(conn))["cursor"] == "S0"
    finally:
        await conn.close()


async def test_allow_prune_below_the_count_still_refuses(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_many(conn, 600)
        fd = FakeDocExt([[_start("S0"), _cursor("S9", 0)]])
        report = await _run(conn, fd, FakeEmbedder(), allow_prune=599)
        assert not report.complete and report.prune_refused == 600
        assert await _doc_count(conn) == 600
    finally:
        await conn.close()


async def test_rerun_after_a_prune_refusal_embeds_nothing(vkb_dsn):
    # run 1 writes one new article then is refused at the end; the re-run resumes after
    # it, streams nothing, embeds nothing, and is refused again until overridden
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_many(conn, 600)
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)],
                         [_start("S1"), _cursor("S9", 0)],
                         [_start("S1"), _cursor("S9", 0)]])
        first = await _run(conn, fd, FakeEmbedder())
        assert first.written == 1 and first.prune_refused == 600
        state = await _state(conn)
        assert state["mode"] == "bootstrap" and "cursor" not in state
        cur = await conn.execute("SELECT doc_id FROM sync_seen")
        assert [r[0] for r in await cur.fetchall()] == [A[0]]  # the finish's TRUNCATE rolled back
        emb = FakeEmbedder()
        second = await _run(conn, fd, emb)
        assert fd.params[1] == {"bootstrap_after": str(A[0])}
        assert emb.calls == 0 and second.prune_refused == 600 and not second.complete
        third = await _run(conn, fd, FakeEmbedder(), allow_prune=600)
        assert third.complete and third.pruned == 600
        assert await _doc_ids(conn) == {A[0]}
        assert (await _state(conn))["cursor"] == "S0"
    finally:
        await conn.close()


async def test_tokens_spent_before_a_failed_embed_are_kept(vkb_dsn):
    # batch 1 succeeds (1 token reported), batch 2 exhausts its 503 retries: the page
    # aborts before anything is written, but bootstrap_tokens keeps the spent token
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]), _cursor("S9", 2)]])
        report = await _run(conn, fd, FakeEmbedder(max_batch=1, fail_after=1))
        assert not report.complete and report.tokens == 1
        state = await _state(conn)
        assert state["bootstrap_tokens"] == "1"  # the reservation replaced by the actual
        assert "abort_count" not in state  # an embed failure never trips the breaker
    finally:
        await conn.close()


async def test_bootstrap_tokens_are_reserved_before_the_embed_call(vkb_dsn):
    seen: list[int] = []

    def observe():
        with psycopg.connect(vkb_dsn) as other:
            row = other.execute(
                "SELECT value FROM sync_state WHERE key='bootstrap_tokens'").fetchone()
            seen.append(int(row[0]))

    conn = await _fresh(vkb_dsn)
    try:
        await conn.execute(
            "INSERT INTO sync_state(key, value) VALUES ('bootstrap_tokens', '100')")
        recs = [_rec(A[0]), _rec(A[1])]
        estimate = sum(len(c.text) // 4 for r in recs
                       for c in chunk_markdown(r["title"], r["content_markdown"]))
        assert estimate > 0
        fd = FakeDocExt([[_start("S0"), *recs]])  # drops: the total is not deleted
        report = await _run(conn, fd, FakeEmbedder(on_call=observe))
        assert seen and seen[0] >= 100 + estimate
        assert (await _state(conn))["bootstrap_tokens"] == str(100 + report.tokens)
    finally:
        await conn.close()


async def test_incremental_makes_no_reservation(vkb_dsn):
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _cursor("S9", 1)],
                         [_rec(A[1]), _cursor("S10", 1)]])
        await _run(conn, fd, FakeEmbedder())
        report = await _run(conn, fd, FakeEmbedder(fail_after=0))
        assert not report.complete
        assert "bootstrap_tokens" not in await _state(conn)
    finally:
        await conn.close()


async def test_pages_sharing_a_first_id_do_not_reset_each_others_breaker(vkb_dsn, monkeypatch):
    # incremental: X appears twice (created, then updated); the duplicate-id rule flushes
    # [X v1] and starts the next page with X v2, so both pages start with X. Replaying the
    # unchanged [X v1] must not reset the count of the failing [X v2, Y].
    conn = await _fresh(vkb_dsn)
    try:
        await _run(conn, FakeDocExt([[_start("S0"), _rec(A[4]), _cursor("S9", 1)]]),
                   FakeEmbedder())
        real = sync_mod.write_document

        async def boom(conn_, meta, *args, **kwargs):
            if meta.doc_id == A[2]:
                raise RuntimeError("bug in the write path")
            return await real(conn_, meta, *args, **kwargs)

        monkeypatch.setattr(sync_mod, "write_document", boom)
        body = [_rec(A[0], body="v1"), _rec(A[0], body="v2"), _rec(A[2]), _cursor("S10", 3)]
        fd = FakeDocExt([body] * 4)
        calls, counts = [], []
        for _ in range(3):
            emb = FakeEmbedder()
            report = await _run(conn, fd, emb)
            assert not report.complete
            calls.append(emb.calls)
            counts.append((await _state(conn)).get("abort_count"))
        assert counts == ["1", "2", "3"]
        assert calls == [2, 1, 1]  # run 1 also embeds [X v1]
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb)
        # quarantine of [X v2, Y]: one call per record, Y set aside, X v2 written
        assert report.complete and emb.calls == 2 and report.quarantined == 1
        assert list(await _failures(conn)) == [A[2]]
    finally:
        await conn.close()


async def test_a_stale_breaker_key_does_not_refuse_a_different_page(vkb_dsn, monkeypatch):
    # [A0, A1] trips the breaker; upstream then drops A1, so the page is [A0, A2]: same
    # first id, different page. It is embedded and committed, and finish clears the keys.
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]), _cursor("S9", 2)]] * 3
                        + [[_start("S0"), _rec(A[0]), _rec(A[2]), _cursor("S9", 2)]])
        with monkeypatch.context() as m:
            _failing_write(m)
            for _ in range(3):
                await _run(conn, fd, FakeEmbedder())
        assert (await _state(conn))["abort_count"] == "3"
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb)
        assert report.complete and emb.calls == 1
        assert await _doc_ids(conn) == {A[0], A[2]}
        assert "abort_count" not in await _state(conn)
    finally:
        await conn.close()


async def test_bootstrap_tokens_are_reserved_again_before_isolation(vkb_dsn):
    # batch size 1: A0's batch is billed (1 token), A1's is rejected, so isolation
    # re-embeds both records plus the canary. Before the first isolation call the
    # reservation covers what was spent plus everything about to be re-embedded.
    seen: list[int] = []

    def observe():
        with psycopg.connect(vkb_dsn) as other:
            row = other.execute(
                "SELECT value FROM sync_state WHERE key='bootstrap_tokens'").fetchone()
            seen.append(int(row[0]))

    conn = await _fresh(vkb_dsn)
    try:
        recs = [_rec(A[0]), _rec(A[1], body="POISON text")]
        estimate = sum(len(c.text) // 4 for r in recs
                       for c in chunk_markdown(r["title"], r["content_markdown"]))
        fd = FakeDocExt([[_start("S0"), *recs]])
        report = await _run(conn, fd, FakeEmbedder(reject="POISON", max_batch=1,
                                                   on_call=observe))
        assert report.failed == 1
        # calls: page A0 (ok), page A1 (400), isolation A0, isolation A1, canary
        assert len(seen) == 5
        assert seen[2] >= 1 + estimate + 1
        assert (await _state(conn))["bootstrap_tokens"] == str(report.tokens)
    finally:
        await conn.close()


async def test_a_failed_token_write_after_an_abort_keeps_the_reservation(vkb_dsn, monkeypatch,
                                                                         caplog):
    # the embed fails after one billed batch, and the best-effort write of the actual total
    # fails too: the run still aborts cleanly and the (larger) reservation stands
    real = sync_mod._set_state
    armed = []

    async def flaky(conn, key, value):
        if armed and key == "bootstrap_tokens":
            raise psycopg.OperationalError("connection lost")
        await real(conn, key, value)

    monkeypatch.setattr(sync_mod, "_set_state", flaky)
    conn = await _fresh(vkb_dsn)
    try:
        recs = [_rec(A[0]), _rec(A[1])]
        estimate = sum(len(c.text) // 4 for r in recs
                       for c in chunk_markdown(r["title"], r["content_markdown"]))
        fd = FakeDocExt([[_start("S0"), *recs, _cursor("S9", 2)]])
        emb = FakeEmbedder(max_batch=1, fail_after=1, on_call=lambda: armed.append(1))
        with caplog.at_level(logging.WARNING, logger="vendorkb.sync"):
            report = await _run(conn, fd, emb)
        assert not report.complete and report.tokens == 1
        state = await _state(conn)
        assert state["bootstrap_tokens"] == str(estimate) and estimate > report.tokens
        assert "abort_count" not in state
        assert any("could not record the abort" in r.getMessage() for r in caplog.records)
        assert await _lock_free(vkb_dsn)
        await conn.execute("SELECT 1")
    finally:
        await conn.close()


N = [uuid.UUID(int=n) for n in range(100, 110)]  # articles arriving between runs


def _poison_write(monkeypatch, doc_id):
    real = sync_mod.write_document

    async def boom(conn, meta, *args, **kwargs):
        if meta.doc_id == doc_id:
            raise RuntimeError("bug in the write path")
        return await real(conn, meta, *args, **kwargs)

    monkeypatch.setattr(sync_mod, "write_document", boom)


async def _seed_incremental(conn) -> None:
    await _run(conn, FakeDocExt([[_start("S0"), _rec(A[4]), _cursor("S9", 1)]]),
               FakeEmbedder())


async def test_a_growing_tail_page_still_trips_the_breaker(vkb_dsn, monkeypatch):
    # the poisoned article is on the partial last page; one new article joins that page
    # before each run, so the page changes every time but its first item does not
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_incremental(conn)
        _poison_write(monkeypatch, A[0])
        fd = FakeDocExt([[_rec(A[0]), *(_rec(n) for n in N[:k]), _cursor("S10", k + 1)]
                         for k in range(4)])
        for n in (1, 2, 3):
            emb = FakeEmbedder()
            report = await _run(conn, fd, emb, page_size=10)
            assert not report.complete and emb.calls == 1
            assert (await _state(conn))["abort_count"] == str(n)
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb, page_size=10)
        # the grown page still matches: quarantine, one call per record
        assert report.complete and emb.calls == 4 and report.quarantined == 1
        assert set(N[:3]) <= await _doc_ids(conn)
        assert (await _state(conn))["cursor"] == "S10"
    finally:
        await conn.close()


async def test_a_tail_page_that_overflows_still_trips_the_breaker(vkb_dsn, monkeypatch):
    # page_size 2: [P] grows to the full page [P, N0]; later arrivals spill onto the next
    # page. The poisoned page keeps its prefix, so it still trips.
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_incremental(conn)
        _poison_write(monkeypatch, A[0])
        fd = FakeDocExt([[_rec(A[0]), *(_rec(n) for n in N[:k]), _cursor("S10", k + 1)]
                         for k in range(4)])
        for n in (1, 2, 3):
            emb = FakeEmbedder()
            report = await _run(conn, fd, emb)
            assert not report.complete and emb.calls == 1
            assert (await _state(conn))["abort_count"] == str(n)
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb)
        # [A0, N0] quarantined (2 calls), then [N1, N2] as a normal page (1 call)
        assert report.complete and emb.calls == 3 and report.quarantined == 1
        assert set(N[:3]) <= await _doc_ids(conn)
    finally:
        await conn.close()


async def _healthy_run_with_breaker(conn, position: str):
    await conn.execute("INSERT INTO sync_state(key, value) VALUES ('abort_position', %s),"
                       " ('abort_count', '3')", (position,))
    emb = FakeEmbedder()
    report = await _run(conn, FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]),
                                           _cursor("S9", 2)]]), emb)
    return report, emb


async def test_a_stored_prefix_that_differs_does_not_refuse_the_page(vkb_dsn):
    # "1:<hash>" of some other first item, at the limit: the healthy page still runs
    conn = await _fresh(vkb_dsn)
    try:
        report, emb = await _healthy_run_with_breaker(conn, "1:" + "ab" * 32)
        assert report.complete and emb.calls == 1
        assert "abort_count" not in await _state(conn)  # finish cleared it
    finally:
        await conn.close()


async def test_an_old_format_breaker_value_never_matches(vkb_dsn):
    for position in ("ab" * 32, str(A[0]), "x:y", "0:" + "ab" * 32, ":"):
        conn = await _fresh(vkb_dsn)
        try:
            report, emb = await _healthy_run_with_breaker(conn, position)
            assert report.complete and emb.calls == 1, position
        finally:
            await conn.close()


async def test_an_update_inside_the_stored_prefix_gets_a_fresh_try(vkb_dsn, monkeypatch):
    # [A0 v1] trips the breaker; upstream then updates A0 (new content): same first id,
    # different prefix hash, so the page is embedded again rather than refused
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_incremental(conn)
        fd = FakeDocExt([[_rec(A[0], body="v1"), _cursor("S10", 1)]] * 3
                        + [[_rec(A[0], body="v2"), _rec(N[0]), _cursor("S11", 2)]])
        with monkeypatch.context() as m:
            _poison_write(m, A[0])
            for _ in range(3):
                await _run(conn, fd, FakeEmbedder(), page_size=10)
        assert (await _state(conn))["abort_count"] == "3"
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb, page_size=10)
        assert report.complete and emb.calls == 1
        assert "abort_count" not in await _state(conn)
    finally:
        await conn.close()


async def test_a_fix_to_an_article_that_joined_after_the_first_abort_gets_a_fresh_try(
        vkb_dsn, monkeypatch):
    # run 1: tail [H] aborts once (a one-off write error); P (poisoned) then joins: [H, P]
    # aborts in runs 2 and 3 (count 3). Upstream fixes P: [H, P'] is embedded again.
    bad = _rec(N[1], body="BAD")
    fixed = _rec(N[1], body="fixed")
    one_off = [True]
    real = sync_mod.write_document

    async def boom(conn, meta, h, *args, **kwargs):
        if meta.doc_id == N[0] and one_off:
            one_off.clear()
            raise RuntimeError("connection dropped")
        if h == bad["content_hash"]:
            raise RuntimeError("bug in the write path")
        return await real(conn, meta, h, *args, **kwargs)

    monkeypatch.setattr(sync_mod, "write_document", boom)
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_incremental(conn)
        fd = FakeDocExt([[_rec(N[0]), _cursor("S10", 1)],
                         [_rec(N[0]), bad, _cursor("S10", 2)],
                         [_rec(N[0]), bad, _cursor("S10", 2)],
                         [_rec(N[0]), fixed, _cursor("S11", 2)]])
        for n in (1, 2, 3):
            report = await _run(conn, fd, FakeEmbedder(), page_size=10)
            assert not report.complete
            assert (await _state(conn))["abort_count"] == str(n)
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb, page_size=10)
        assert report.complete and emb.calls == 1
        assert {N[0], N[1]} <= await _doc_ids(conn)
    finally:
        await conn.close()


def _key(*recs) -> str:
    return sync_mod._page_key([parse_line(json.dumps(r)) for r in recs])


async def test_a_near_miss_prefix_does_not_refuse_the_page(vkb_dsn):
    # same first item, different second item; and a stored prefix longer than the page
    for position in (f"2:{_key(_rec(A[0]), _rec(A[3]))}",
                     f"3:{_key(_rec(A[0]), _rec(A[1]), _rec(A[2]))}"):
        conn = await _fresh(vkb_dsn)
        try:
            report, emb = await _healthy_run_with_breaker(conn, position)
            assert report.complete and emb.calls == 1, position
        finally:
            await conn.close()


async def test_the_exact_prefix_selects_quarantine_mode(vkb_dsn):
    # the positive control for the near misses: the same helper with the matching prefix
    # runs the page in quarantine mode (one embed call per record, not one per page)
    conn = await _fresh(vkb_dsn)
    try:
        report, emb = await _healthy_run_with_breaker(conn, f"1:{_key(_rec(A[0]))}")
        assert report.complete and emb.calls == 2 and report.quarantined == 0
        assert "abort_count" not in await _state(conn)
    finally:
        await conn.close()


async def test_a_replayed_full_page_does_not_reset_a_growing_tail(vkb_dsn, monkeypatch):
    # page_size 2: [A0, A1] is replayed unchanged each run and commits; the poisoned tail
    # [P] grows with arrivals. The replayed commit must not reset the tail's count.
    conn = await _fresh(vkb_dsn)
    try:
        await _run(conn, FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]),
                                      _cursor("S9", 2)]]), FakeEmbedder())
        _poison_write(monkeypatch, A[2])
        fd = FakeDocExt([[_rec(A[0]), _rec(A[1]), _rec(A[2]), *(_rec(n) for n in N[:k]),
                          _cursor("S10", k + 3)] for k in range(4)])
        for n in (1, 2, 3):
            emb = FakeEmbedder()
            report = await _run(conn, fd, emb)
            assert not report.complete and emb.calls == 1
            assert (await _state(conn))["abort_count"] == str(n)
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb)
        # [A0, A1] unchanged, [A2, N0] quarantined (2 calls), [N1, N2] normal (1 call)
        assert report.complete and emb.calls == 3 and report.quarantined == 1
    finally:
        await conn.close()


# --- quarantine: a tripped breaker sets records aside instead of blocking the feed ---

def _poison(monkeypatch, doc_id, exc=None):
    """write_document raises for ``doc_id`` (a RuntimeError unless ``exc`` is given)."""
    real = sync_mod.write_document

    async def boom(conn, meta, *args, **kwargs):
        if meta.doc_id == doc_id:
            raise exc or RuntimeError("bug in the write path")
        return await real(conn, meta, *args, **kwargs)

    monkeypatch.setattr(sync_mod, "write_document", boom)


async def _trip(conn, fd, **kw) -> None:
    """Three runs that abort after embedding the same page: the breaker trips."""
    for n in (1, 2, 3):
        report = await _run(conn, fd, FakeEmbedder(), **kw)
        assert not report.complete
        assert (await _state(conn))["abort_count"] == str(n)


async def test_quarantine_writes_the_healthy_records_and_sets_the_poisoned_one_aside(
        vkb_dsn, monkeypatch, caplog):
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_incremental(conn)
        _poison(monkeypatch, N[1])
        fd = FakeDocExt([[_rec(N[0]), _rec(N[1]), _rec(N[2]), _cursor("S10", 3)]] * 4)
        await _trip(conn, fd, page_size=10)
        emb = FakeEmbedder()
        with caplog.at_level(logging.ERROR, logger="vendorkb.sync"):
            report = await _run(conn, fd, emb, page_size=10)
        assert report.complete and emb.calls == 3  # one embed call per record
        assert (report.written, report.quarantined, report.failed) == (2, 1, 1)
        assert {N[0], N[2]} <= await _doc_ids(conn) and N[1] not in await _doc_ids(conn)
        fails = await _failures(conn)
        assert list(fails) == [N[1]]
        assert fails[N[1]].startswith("quarantined after 3 page aborts: ")
        assert "bug in the write path" in fails[N[1]]
        state = await _state(conn)
        assert state["cursor"] == "S10"
        assert "abort_position" not in state and "abort_count" not in state
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any(str(N[1]) in m and "content changes" in m for m in msgs)
    finally:
        await conn.close()


async def test_an_article_that_joined_the_tripped_tail_page_is_written(vkb_dsn, monkeypatch):
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_incremental(conn)
        _poison(monkeypatch, N[1])
        fd = FakeDocExt([[_rec(N[0]), _rec(N[1]), _cursor("S10", 2)]] * 3
                        + [[_rec(N[0]), _rec(N[1]), _rec(N[2]), _cursor("S11", 3)]])
        await _trip(conn, fd, page_size=10)
        report = await _run(conn, fd, FakeEmbedder(), page_size=10)
        assert report.complete and report.quarantined == 1
        assert {N[0], N[2]} <= await _doc_ids(conn)
        assert (await _state(conn))["cursor"] == "S11"
    finally:
        await conn.close()


async def test_pages_after_the_quarantined_page_are_written_in_the_same_run(vkb_dsn,
                                                                            monkeypatch):
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_incremental(conn)
        _poison(monkeypatch, N[1])
        body = [_rec(N[0]), _rec(N[1]), _rec(N[2]), _rec(N[3]), _cursor("S10", 4)]
        fd = FakeDocExt([body] * 4)
        await _trip(conn, fd)  # page_size 2: [N0, N1] trips
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb)
        assert report.complete and emb.calls == 3  # 2 quarantine calls + the next page's
        assert {N[0], N[2], N[3]} <= await _doc_ids(conn)
        assert list(await _failures(conn)) == [N[1]]
    finally:
        await conn.close()


async def test_quarantine_aborts_when_every_record_fails(vkb_dsn, monkeypatch):
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_incremental(conn)
        _failing_write(monkeypatch)
        fd = FakeDocExt([[_rec(N[0]), _rec(N[1]), _cursor("S10", 2)]] * 4)
        await _trip(conn, fd, page_size=10)
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb, page_size=10)
        assert not report.complete and emb.calls == 2
        assert report.quarantined == 0
        assert not ({N[0], N[1]} & await _doc_ids(conn))
        assert await _failures(conn) == {}
        state = await _state(conn)
        assert state["cursor"] == "S0" and int(state["abort_count"]) >= 3
    finally:
        await conn.close()


async def test_a_connection_error_during_quarantine_aborts_the_page(vkb_dsn, monkeypatch):
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_incremental(conn)
        fd = FakeDocExt([[_rec(N[0]), _rec(N[1]), _rec(N[2]), _cursor("S10", 3)]] * 4)
        with monkeypatch.context() as m:
            _poison(m, N[1])
            await _trip(conn, fd, page_size=10)
        _poison(monkeypatch, N[1], psycopg.OperationalError("server closed the connection"))
        report = await _run(conn, fd, FakeEmbedder(), page_size=10)
        assert not report.complete and report.quarantined == 0
        assert not ({N[0], N[1], N[2]} & await _doc_ids(conn))
        assert await _failures(conn) == {}
        assert (await _state(conn))["cursor"] == "S0"
        await conn.execute("SELECT 1")
    finally:
        await conn.close()


async def test_a_quarantined_article_is_written_once_its_content_changes(vkb_dsn,
                                                                         monkeypatch):
    conn = await _fresh(vkb_dsn)
    try:
        await _seed_incremental(conn)
        fd = FakeDocExt([[_rec(N[0]), _rec(N[1]), _cursor("S10", 2)]] * 4
                        + [[_rec(N[1], change="updated", body="fixed"), _cursor("S11", 1)]])
        with monkeypatch.context() as m:
            _poison(m, N[1])
            await _trip(conn, fd, page_size=10)
            report = await _run(conn, fd, FakeEmbedder(), page_size=10)
            assert report.complete and report.quarantined == 1
        report = await _run(conn, fd, FakeEmbedder(), page_size=10)
        assert report.complete and report.written == 1
        assert N[1] in await _doc_ids(conn)
        assert await _failures(conn) == {}
    finally:
        await conn.close()


async def test_bootstrap_quarantine_commits_the_page_and_counts_every_token(vkb_dsn,
                                                                           monkeypatch):
    conn = await _fresh(vkb_dsn)
    try:
        _poison(monkeypatch, N[1])
        # no cursor line: page 1 is [N0, N1, N2], N3 is left pending when the stream ends
        fd = FakeDocExt([[_start("S0"), *(_rec(n) for n in N[:4])]] * 4)
        spent = 0
        for n in (1, 2, 3):
            report = await _run(conn, fd, FakeEmbedder(), page_size=3)
            spent += report.tokens
            assert (await _state(conn))["abort_count"] == str(n)
        report = await _run(conn, fd, FakeEmbedder(), page_size=3)
        spent += report.tokens
        assert report.quarantined == 1 and report.written == 2
        state = await _state(conn)
        assert state["bootstrap_after"] == str(N[2])
        assert "abort_count" not in state
        assert int(state["bootstrap_tokens"]) >= spent
    finally:
        await conn.close()
