import hashlib
import json
import uuid

import httpx

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
            return httpx.Response(200, text="".join(json.dumps(x) + "\n" for x in lines))
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
    def __init__(self, status: int = 200):
        self.status = status
        self.texts: list[str] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        if self.status != 200:
            return httpx.Response(self.status, text="bad request")
        inputs = json.loads(req.content)["input"]
        self.texts.extend(inputs)
        return httpx.Response(200, json={
            "data": [{"index": i, "embedding": [0.0] * 767 + [1.0]} for i in range(len(inputs))],
            "usage": {"total_tokens": len(inputs)}})

    def client(self) -> Embedder:
        return Embedder(_settings(), transport=httpx.MockTransport(self.handler))


async def _fresh(dsn):
    conn = await connect(dsn)
    await init_schema(conn, embed_dim=768)
    await conn.execute("TRUNCATE documents, chunks, sync_state, sources, sync_failures CASCADE")
    return conn


async def _run(conn, docext: FakeDocExt, emb: FakeEmbedder, **kw):
    async with docext.client() as d, emb.client() as e:
        return await sync(conn, docext=d, embedder=e, page_size=2, **kw)


async def _state(conn) -> dict[str, str]:
    cur = await conn.execute("SELECT key, value FROM sync_state")
    return dict(await cur.fetchall())


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


async def test_embed_failure_aborts_without_moving_watermark(vkb_dsn):
    # the embedder transport returns 400 (not retried). Assert 0 documents written,
    # bootstrap_after unset, report.complete False, sync_failures empty
    conn = await _fresh(vkb_dsn)
    try:
        fd = FakeDocExt([[_start("S0"), _rec(A[0]), _rec(A[1]), _cursor("S9", 2)]])
        report = await _run(conn, fd, FakeEmbedder(status=400))
        assert not report.complete
        assert await _doc_ids(conn) == set()
        state = await _state(conn)
        assert "bootstrap_after" not in state and "cursor" not in state
        cur = await conn.execute("SELECT count(*) FROM sync_failures")
        assert (await cur.fetchone())[0] == 0
        # the lock was released and the connection is still usable
        cur = await conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,))
        assert (await cur.fetchone())[0] is True
        await conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
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
