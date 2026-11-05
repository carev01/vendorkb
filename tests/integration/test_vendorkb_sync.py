import hashlib
import json
import logging
import uuid

import httpx
import psycopg

from vendorkb import sync as sync_mod
from vendorkb.clients import DocExtractor, Embedder
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect, init_schema
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
                 canary_status: int = 200):
        self.status = status
        self.reject = reject
        self.canary_status = canary_status
        self.calls = 0
        self.texts: list[str] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.calls += 1
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
        return Embedder(_settings(), transport=httpx.MockTransport(self.handler),
                        retry_base_delay=0.0)


async def _fresh(dsn):
    conn = await connect(dsn)
    await init_schema(conn, embed_dim=768)
    await conn.execute("TRUNCATE documents, chunks, sync_state, sources, sync_failures,"
                       " sync_seen CASCADE")
    return conn


async def _run(conn, docext: FakeDocExt, emb: FakeEmbedder, **kw):
    async with docext.client() as d, emb.client() as e:
        return await sync(conn, docext=d, embedder=e, page_size=2, **kw)


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


def _failing_write(monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("bug in the write path")
    monkeypatch.setattr(sync_mod, "write_document", boom)


async def test_three_post_embed_aborts_trip_the_breaker(vkb_dsn, monkeypatch, caplog):
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
            assert (state["abort_position"], state["abort_count"]) == (str(A[0]), str(n))
        emb = FakeEmbedder()
        with caplog.at_level(logging.ERROR, logger="vendorkb.sync"):
            report = await _run(conn, fd, emb)
        assert not report.complete and emb.calls == 0
        assert await _lock_free(vkb_dsn)
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any(str(A[0]) in m and "abort_count" in m for m in msgs)
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
            assert (state["abort_position"], state["abort_count"]) == (str(A[2]), str(n))
        emb = FakeEmbedder()
        report = await _run(conn, fd, emb)
        assert not report.complete and emb.calls == 0
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
