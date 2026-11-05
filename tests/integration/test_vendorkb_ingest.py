import hashlib
import uuid

import httpx

from vendorkb.clients import CorpusEntry, DocExtractor, Embedder
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect, init_schema
from vendorkb.ingest import ingest

MD = "# Guide\n\nIntro text about backups.\n\n## Install\n\nRun the installer now.\n"


def _h(md):
    return hashlib.sha256(md.encode()).hexdigest()


def _clients(articles):
    s = VendorkbSettings(_env_file=None, embed_dim=768, embed_input_type=False,
                         embed_output_dimension=None, docext_base_url="http://d", docext_read_key="k",
                         embed_base_url="http://e/v1")

    def dx(req: httpx.Request) -> httpx.Response:
        aid = req.url.path.rsplit("/", 1)[1]
        if aid not in articles:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"id": aid, "content_markdown": articles[aid]})

    def em(req: httpx.Request) -> httpx.Response:
        import json
        inputs = json.loads(req.content)["input"]
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [0.5] * 768} for i in range(len(inputs))]})

    return (DocExtractor(s, transport=httpx.MockTransport(dx)),
            Embedder(s, transport=httpx.MockTransport(em)))


def _entry(aid, md=MD, graph_hash="same"):
    return CorpusEntry(aid, "V", "P", "S", "Guide", f"http://u/{aid}",
                       _h(md) if graph_hash == "same" else graph_hash)


async def _wipe(conn):
    await conn.execute("DELETE FROM documents")


async def test_ingest_writes_chunks_and_resumes(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        await _wipe(conn)
        a, b = str(uuid.uuid4()), str(uuid.uuid4())
        dx, em = _clients({a: MD, b: MD})
        corpus = [_entry(a), _entry(b)]
        r1 = await ingest(conn, corpus, docext=dx, embedder=em, concurrency=4)
        assert r1.fetched == 2 and r1.failed == [] and r1.chunks_written >= 2
        cur = await conn.execute("SELECT count(*) FROM chunks")
        n = (await cur.fetchone())[0]
        assert n == r1.chunks_written
        r2 = await ingest(conn, corpus, docext=dx, embedder=em, concurrency=4)
        assert r2.skipped_unchanged == 2 and r2.fetched == 0 and r2.chunks_written == 0
        cur = await conn.execute("SELECT count(*) FROM chunks")
        assert (await cur.fetchone())[0] == n
        cur = await conn.execute("SELECT vendor, title, url FROM documents WHERE doc_id=%s",
                                 (uuid.UUID(a),))
        assert await cur.fetchone() == ("V", "Guide", f"http://u/{a}")
    finally:
        await conn.close()


async def test_ingest_flags_changed_since_graph_and_records_failures(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await init_schema(conn, embed_dim=768)
        await _wipe(conn)
        a, bad, empty = (str(uuid.uuid4()) for _ in range(3))
        dx, em = _clients({a: MD, empty: ""})
        corpus = [_entry(a, graph_hash="old"), _entry(bad), _entry(empty)]
        r = await ingest(conn, corpus, docext=dx, embedder=em)
        assert r.changed_since_graph == [a]
        assert bad in r.failed[0] and empty in r.failed[1]
        assert r.fetched == 1
        cur = await conn.execute("SELECT count(*) FROM documents")
        assert (await cur.fetchone())[0] == 1
    finally:
        await conn.close()
