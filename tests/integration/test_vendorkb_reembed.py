import uuid
from datetime import UTC, datetime

import pytest

from vendorkb.db import connect, init_schema
from vendorkb.reembed import promote, reembed
from vendorkb.search import search_docs


class FakeEmbedder:
    """Deterministic 4-dim vectors; counts tokens like a provider would."""

    def __init__(self, fail_on=None):
        self.tokens_used = 0
        self.calls = []
        self.fail_on = fail_on

    async def embed(self, texts, *, input_type="document"):
        self.calls.append((list(texts), input_type))
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise RuntimeError("provider down")
        self.tokens_used += len(texts)
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]


async def _seed(conn, n):
    await conn.execute("DROP TABLE IF EXISTS chunks, documents CASCADE")
    await init_schema(conn, embed_dim=3)
    doc = uuid.uuid4()
    await conn.execute("INSERT INTO documents(doc_id, vendor, title, content_hash, fetched_at) "
                       "VALUES (%s, 'V', 'T', 'h', %s)", (doc, datetime.now(UTC)))
    for i in range(n):
        await conn.execute(
            "INSERT INTO chunks(chunk_id, doc_id, section_path, ordinal, text, token_count, "
            "embedding) VALUES (%s, %s, %s, %s, %s, 1, %s)",
            (uuid.uuid4(), doc, ["T"], i, "x" * (i + 1), [0.0, 0.0, 1.0]))


async def test_reembed_fills_candidate_resumably_then_promote_swaps(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _seed(conn, 7)
        emb = FakeEmbedder()
        first = await reembed(conn, emb, dim=4, batch=2, concurrency=2, limit=3)
        assert (first.chunks, first.remaining, first.tokens) == (3, 4, 3)
        rest = await reembed(conn, emb, dim=4, batch=2, concurrency=2)
        assert (rest.chunks, rest.remaining) == (4, 0)
        assert {t for _, t in emb.calls} == {"document"}
        await conn.execute("CREATE INDEX chunks_embedding_next_idx ON chunks "
                           "USING hnsw (embedding_next vector_cosine_ops)")
        assert max(len(texts) for texts, _ in emb.calls) == 2
        assert await promote(conn) == "vector(4)"
        cur = await conn.execute("SELECT format_type(atttypid, atttypmod), attnotnull "
                                 "FROM pg_attribute WHERE attrelid='chunks'::regclass "
                                 "AND attname='embedding'")
        assert await cur.fetchone() == ("vector(4)", True)
        cur = await conn.execute("SELECT indexname FROM pg_indexes WHERE tablename='chunks' "
                                 "AND indexdef LIKE '%%hnsw%%'")
        assert [r[0] for r in await cur.fetchall()] == ["chunks_embedding_idx"]
    finally:
        await conn.close()


async def test_promote_refuses_while_candidate_incomplete_or_missing(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _seed(conn, 3)
        with pytest.raises(RuntimeError, match="run reembed first"):
            await promote(conn)
        await reembed(conn, FakeEmbedder(), dim=4, batch=2, concurrency=1, limit=1)
        with pytest.raises(RuntimeError, match="2 chunks"):
            await promote(conn)
    finally:
        await conn.close()


async def test_reembed_failure_leaves_unwritten_rows_null_and_dim_mismatch_refused(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _seed(conn, 4)
        with pytest.raises(RuntimeError, match="provider down"):
            await reembed(conn, FakeEmbedder(fail_on="xxx"), dim=4, batch=2, concurrency=2)
        cur = await conn.execute("SELECT count(*) FROM chunks WHERE embedding_next IS NULL")
        assert (await cur.fetchone())[0] == 4
        with pytest.raises(RuntimeError, match="expected vector"):
            await reembed(conn, FakeEmbedder(), dim=8, batch=2, concurrency=1)
    finally:
        await conn.close()


async def test_search_reads_candidate_column_and_embeds_query_as_query(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _seed(conn, 3)
        await reembed(conn, FakeEmbedder(), dim=4, batch=8, concurrency=1)
        emb = FakeEmbedder()
        res = await search_docs(conn, "zzz", embedder=emb, reranker=None, use_rerank=False,
                                vector_column="embedding_next")
        assert len(res.hits) == 3 and emb.calls == [(["zzz"], "query")]
        with pytest.raises(ValueError):
            await search_docs(conn, "zzz", embedder=emb, reranker=None,
                              vector_column="embedding; DROP TABLE chunks")
    finally:
        await conn.close()
