import uuid

from vendorkb.chunker import Chunk
from vendorkb.db import connect, init_schema
from vendorkb.store import DocMeta, delete_document, stored_hashes, update_metadata, write_document

VEC = [0.0] * 767 + [1.0]


def _chunk(ordinal: int) -> Chunk:
    return Chunk(section_path=("T",), ordinal=ordinal, text=f"text {ordinal}", token_count=2)


def _meta(doc_id: uuid.UUID, **kw) -> DocMeta:
    base = dict(doc_id=doc_id, vendor="V", product="P", source="S",
                source_id=uuid.uuid4(), title="T", url="https://x/y")
    return DocMeta(**{**base, **kw})


async def _fresh(conn) -> None:
    await conn.execute("DROP TABLE IF EXISTS chunks, documents CASCADE")
    await init_schema(conn, embed_dim=768)


async def _count(conn, table: str) -> int:
    cur = await conn.execute(f"SELECT count(*) FROM {table}")
    return (await cur.fetchone())[0]


async def test_write_document_replaces_chunks(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc_id = uuid.uuid4()
        async with conn.transaction():
            await write_document(conn, _meta(doc_id), "h1", [_chunk(0), _chunk(1)], [VEC, VEC])
        assert await _count(conn, "chunks") == 2
        async with conn.transaction():
            await write_document(conn, _meta(doc_id), "h2", [_chunk(0)], [VEC])
        assert await _count(conn, "chunks") == 1
        assert await stored_hashes(conn, [doc_id, uuid.uuid4()]) == {doc_id: "h2"}
    finally:
        await conn.close()


async def test_update_metadata_reports_change_only_when_different(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        await write_document(conn, meta, "h", [_chunk(0)], [VEC])
        renamed = _meta(doc_id, title="New", source_id=meta.source_id)
        assert await update_metadata(conn, renamed) is True
        assert await update_metadata(conn, renamed) is False
        cur = await conn.execute("SELECT title FROM documents WHERE doc_id=%s", (doc_id,))
        assert (await cur.fetchone())[0] == "New"
    finally:
        await conn.close()


async def test_update_metadata_treats_null_to_value_as_a_change(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc_id = uuid.uuid4()
        stored = _meta(doc_id, product=None)
        await write_document(conn, stored, "h", [_chunk(0)], [VEC])
        assert await update_metadata(conn, stored) is False  # NULL equals NULL
        tagged = _meta(doc_id, product="P", source_id=stored.source_id)
        assert await update_metadata(conn, tagged) is True
        assert await update_metadata(conn, tagged) is False
        assert await update_metadata(conn, _meta(uuid.uuid4())) is False  # no such row
    finally:
        await conn.close()


async def test_delete_document_cascades_chunks(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc_id = uuid.uuid4()
        await write_document(conn, _meta(doc_id), "h", [_chunk(0), _chunk(1)], [VEC, VEC])
        assert await delete_document(conn, doc_id) is True
        assert await _count(conn, "chunks") == 0
        assert await delete_document(conn, doc_id) is False
    finally:
        await conn.close()
