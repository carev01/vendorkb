"""A re-synced page keeps the chunks that carry facts (superseded) until its new version is
extracted and mapped (``store.write_document`` / ``store.retire_superseded``).

Every name and text here is invented for the tests."""
import uuid

from pgvector import Vector

from tests.integration.test_capabilities_v2 import _fresh
from vendorkb.chunker import Chunk
from vendorkb.db import connect, init_schema
from vendorkb.store import (DocMeta, delete_document, retire_superseded, update_metadata,
                            write_document)

VEC = Vector([0.1] * 8)
FAMILY = "claims-v3.1"


def _chunks(n: int, tag: str) -> list[Chunk]:
    return [Chunk(section_path=("T",), ordinal=i, text=f"{tag} text {i}", token_count=3)
            for i in range(n)]


def _meta(doc_id: uuid.UUID, **kw) -> DocMeta:
    base = dict(doc_id=doc_id, vendor="V", product="P", source="S",
                source_id=uuid.uuid4(), title="T", url="https://x/y")
    return DocMeta(**{**base, **kw})


async def _write(conn, meta: DocMeta, n: int, tag: str) -> list[uuid.UUID]:
    """Write a version of ``n`` chunks; the ids of the document's current chunks, by ordinal."""
    await write_document(conn, meta, f"hash-{tag}", _chunks(n, tag), [VEC] * n)
    return await _current(conn, meta.doc_id)


async def _current(conn, doc_id: uuid.UUID) -> list[uuid.UUID]:
    cur = await conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = %s"
                             " AND superseded_at IS NULL ORDER BY ordinal", (doc_id,))
    return [r[0] for r in await cur.fetchall()]


async def _superseded(conn, doc_id: uuid.UUID) -> list[uuid.UUID]:
    cur = await conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = %s"
                             " AND superseded_at IS NOT NULL ORDER BY ordinal", (doc_id,))
    return [r[0] for r in await cur.fetchall()]


async def _extract(conn, chunk_id: uuid.UUID, *, claim_mapped: str | None | bool = False,
                   status: str = "done", prompt_version: str = FAMILY) -> uuid.UUID | None:
    """An extraction of the chunk; ``claim_mapped`` False adds no claim, else one claim whose
    ``mapped_version`` is that value (None = not mapped yet). Returns the claim id."""
    ext_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status) VALUES (%s,%s,'m',%s,%s)", (ext_id, chunk_id, prompt_version, status))
    if claim_mapped is False:
        return None
    claim_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO raw_claims (claim_id, extraction_id, chunk_id, vendor, product,"
        " capability_raw, support, evidence_quote, gate, extractor_model, prompt_version,"
        " mapped_version) VALUES (%s,%s,%s,'V','P','a capability','supported','quote',"
        "'exact','m',%s,%s)", (claim_id, ext_id, chunk_id, prompt_version, claim_mapped))
    return claim_id


async def _count(conn, sql: str, *params) -> int:
    cur = await conn.execute(sql, params)
    return (await cur.fetchone())[0]


async def test_an_update_keeps_the_extracted_version_until_the_new_one_is_mapped(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        v1 = await _write(conn, meta, 2, "v1")
        claim = await _extract(conn, v1[0], claim_mapped=version)
        await _extract(conn, v1[1])  # done, no claim

        # v2: v1's chunks stay, superseded, with their claim; v2's chunks are current
        v2 = await _write(conn, meta, 2, "v2")
        assert await _superseded(conn, doc_id) == v1
        assert len(v2) == 2 and not set(v1) & set(v2)
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s",
                            claim) == 1

        # v3 before v2 was extracted: v2 carried no fact, so it is simply replaced
        v3 = await _write(conn, meta, 1, "v3")
        assert await _superseded(conn, doc_id) == v1
        assert await _count(conn, "SELECT count(*) FROM chunks WHERE chunk_id = ANY(%s)",
                            v2) == 0
        assert len(v3) == 1
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s",
                            claim) == 1

        # nothing retires while the new version is unextracted
        assert await retire_superseded(conn, version) == 0

        # extracted, but its claim is mapped under another taxonomy version: still nothing
        new_claim = await _extract(conn, v3[0], claim_mapped="old-version")
        assert await retire_superseded(conn, version) == 0

        # mapped under the current version: v1 is retired, claim and all
        await conn.execute("UPDATE raw_claims SET mapped_version = %s WHERE claim_id = %s",
                           (version, new_claim))
        assert await retire_superseded(conn, version) == len(v1)
        assert await _superseded(conn, doc_id) == []
        assert await _current(conn, doc_id) == v3
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s",
                            claim) == 0
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s",
                            new_claim) == 1
        assert await retire_superseded(conn, version) == 0  # nothing left to retire
    finally:
        await conn.close()


async def test_fewer_chunks_after_an_update(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc_id = uuid.uuid4()
        v1 = await _write(conn, _meta(doc_id), 3, "v1")
        await _extract(conn, v1[1], claim_mapped=None)  # one chunk of three carries a claim

        # no unique violation: ordinals 0..2 superseded and 0 current coexist
        await _write(conn, _meta(doc_id, title="New title"), 1, "v2")
        cur = await conn.execute(
            "SELECT ordinal, superseded_at IS NOT NULL FROM chunks WHERE doc_id = %s"
            " ORDER BY superseded_at IS NOT NULL, ordinal", (doc_id,))
        assert await cur.fetchall() == [(0, False), (0, True), (1, True), (2, True)]
        # the document row is the new version
        cur = await conn.execute("SELECT title, content_hash FROM documents WHERE doc_id = %s",
                                 (doc_id,))
        assert await cur.fetchone() == ("New title", "hash-v2")
    finally:
        await conn.close()


async def test_a_retag_supersedes_nothing(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        v1 = await _write(conn, meta, 2, "v1")
        await _extract(conn, v1[0], claim_mapped=version)
        assert await update_metadata(conn, _meta(doc_id, vendor="W", title="Other",
                                                 source_id=meta.source_id)) is True
        assert await _count(conn, "SELECT count(*) FROM chunks WHERE superseded_at IS NOT NULL") == 0
        assert await _current(conn, doc_id) == v1
        assert await _count(conn, "SELECT count(*) FROM raw_claims") == 1
    finally:
        await conn.close()


async def test_removal_still_deletes_everything(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        v1 = await _write(conn, meta, 2, "v1")
        await _extract(conn, v1[0], claim_mapped=version)
        v2 = await _write(conn, meta, 1, "v2")
        await _extract(conn, v2[0], claim_mapped=version)
        v3 = await _write(conn, meta, 1, "v3")  # v2 carried a claim: v1 gone, v2 superseded
        assert len(await _superseded(conn, doc_id)) == 1
        assert await _current(conn, doc_id) == v3
        # a second document is untouched
        other = uuid.uuid4()
        other_chunks = await _write(conn, _meta(other), 1, "o")
        await _extract(conn, other_chunks[0], claim_mapped=version)

        assert await delete_document(conn, doc_id) is True
        assert await _count(conn, "SELECT count(*) FROM chunks WHERE doc_id = %s", doc_id) == 0
        # only the other document's extraction and claim remain
        assert await _count(conn, "SELECT count(*) FROM extractions") == 1
        assert await _count(conn, "SELECT count(*) FROM raw_claims") == 1
        assert await _current(conn, other) == other_chunks
    finally:
        await conn.close()


async def test_retiring_waits_for_a_final_extraction_of_every_current_chunk(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        v1 = await _write(conn, meta, 1, "v1")
        await _extract(conn, v1[0], claim_mapped=version)
        v2 = await _write(conn, meta, 2, "v2")
        assert len(await _superseded(conn, doc_id)) == 1

        await _extract(conn, v2[0])  # done, no claim
        # the second chunk is unextracted; so is one with only an errored attempt
        assert await retire_superseded(conn, version) == 0
        await _extract(conn, v2[1], status="errored")
        assert await retire_superseded(conn, version) == 0
        # a 'done' extraction of another prompt family does not count
        await _extract(conn, v2[1], prompt_version="claims-v2")
        assert await retire_superseded(conn, version) == 0
        # a failed one is final: no claims will ever come, so the page's old facts go
        await _extract(conn, v2[1], status="failed", prompt_version="claims-v3")
        assert await retire_superseded(conn, version) == 1
        assert await _count(conn, "SELECT count(*) FROM raw_claims") == 0
    finally:
        await conn.close()


async def test_retiring_without_a_taxonomy_version_checks_extraction_only(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        v1 = await _write(conn, meta, 1, "v1")
        await _extract(conn, v1[0], claim_mapped=version)
        v2 = await _write(conn, meta, 1, "v2")
        await _extract(conn, v2[0], claim_mapped=None)  # a claim not mapped yet
        assert await retire_superseded(conn, version) == 0
        assert await retire_superseded(conn, None) == 1
    finally:
        await conn.close()


async def test_the_schema_upgrade_lets_superseded_ordinals_coexist(vkb_dsn):
    """A database from before this change has UNIQUE (doc_id, ordinal) on chunks and no
    ``superseded_at``: the upgrade drops the constraint and adds the column, over its rows."""
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn)
        doc_id = uuid.uuid4()
        await _write(conn, _meta(doc_id), 1, "v1")
        await conn.execute("DROP INDEX chunks_doc_ordinal_current_idx")
        await conn.execute("DROP INDEX chunks_superseded_idx")
        await conn.execute("ALTER TABLE chunks DROP COLUMN superseded_at")
        await conn.execute("ALTER TABLE chunks ADD CONSTRAINT chunks_doc_id_ordinal_key"
                           " UNIQUE (doc_id, ordinal)")
        await conn.execute("DELETE FROM taxonomy_meta WHERE key = 'schema_version'")
        await init_schema(conn, embed_dim=8)
        assert await _count(conn, "SELECT count(*) FROM chunks WHERE superseded_at IS NULL") == 1
        first = (await _current(conn, doc_id))[0]
        await _extract(conn, first, claim_mapped=None)
        await _write(conn, _meta(doc_id), 1, "v2")  # ordinal 0 twice: superseded + current
        assert await _count(conn, "SELECT count(*) FROM chunks WHERE doc_id = %s", doc_id) == 2
    finally:
        await conn.close()
