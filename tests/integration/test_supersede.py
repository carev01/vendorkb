"""A re-synced page keeps a chunk whose text is unchanged, with its facts, and keeps the dropped
chunks that carry mapped facts (superseded) until its new version is extracted and mapped
(``store.write_document`` / ``store.retire_superseded``).

Every name and text here is invented for the tests."""
import asyncio
import uuid

from pgvector import Vector

from tests.integration.test_capabilities_v2 import _fresh
from tests.integration.test_claims_extract import GOOD, _settings
from vendorkb.chunker import Chunk
from vendorkb.claims import extract
from vendorkb.db import connect, init_schema
from vendorkb.search import read_section
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


async def _write_texts(conn, meta: DocMeta, texts: list[str], vec: Vector = VEC,
                       token_count: int = 3, path: tuple[str, ...] = ("T",)
                       ) -> tuple[int, list[uuid.UUID]]:
    """Write a version whose chunks have exactly these texts; the number ``write_document``
    reused, and the document's current chunk ids by ordinal."""
    chunks = [Chunk(section_path=path, ordinal=i, text=t, token_count=token_count)
              for i, t in enumerate(texts)]
    written = await write_document(conn, meta, f"hash-{texts}", chunks, [vec] * len(texts))
    return written.reused, await _current(conn, meta.doc_id)


async def _current(conn, doc_id: uuid.UUID) -> list[uuid.UUID]:
    cur = await conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = %s"
                             " AND superseded_at IS NULL ORDER BY ordinal", (doc_id,))
    return [r[0] for r in await cur.fetchall()]


async def _superseded(conn, doc_id: uuid.UUID) -> list[uuid.UUID]:
    cur = await conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = %s"
                             " AND superseded_at IS NOT NULL ORDER BY ordinal", (doc_id,))
    return [r[0] for r in await cur.fetchall()]


async def _extract(conn, chunk_id: uuid.UUID, *, claim_mapped: str | None | bool = False,
                   status: str = "done", prompt_version: str = FAMILY,
                   quote: str = "quote") -> uuid.UUID | None:
    """An extraction of the chunk; ``claim_mapped`` False adds no claim, else one claim quoting
    ``quote`` whose ``mapped_version`` is that value (None = not mapped yet). Returns the
    claim id."""
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
        " mapped_version) VALUES (%s,%s,%s,'V','P','a capability','supported',%s,"
        "'exact','m',%s,%s)", (claim_id, ext_id, chunk_id, quote, prompt_version, claim_mapped))
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

        # v2: v1's chunk with a mapped claim stays, superseded, with its claim; the one
        # without a claim is deleted; v2's chunks are current
        v2 = await _write(conn, meta, 2, "v2")
        assert await _superseded(conn, doc_id) == [v1[0]]
        assert await _count(conn, "SELECT count(*) FROM chunks WHERE chunk_id = %s", v1[1]) == 0
        assert len(v2) == 2 and not set(v1) & set(v2)
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s",
                            claim) == 1

        # v3 before v2 was extracted: v2 carried no fact, so it is simply replaced
        v3 = await _write(conn, meta, 1, "v3")
        assert await _superseded(conn, doc_id) == [v1[0]]
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
        assert await retire_superseded(conn, version) == 1
        assert await _superseded(conn, doc_id) == []
        assert await _current(conn, doc_id) == v3
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s",
                            claim) == 0
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s",
                            new_claim) == 1
        assert await retire_superseded(conn, version) == 0  # nothing left to retire
    finally:
        await conn.close()


async def test_a_second_update_never_drops_the_facts_in_the_table(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        v1 = await _write(conn, meta, 2, "v1")
        claim = await _extract(conn, v1[0], claim_mapped=version)

        # v2 is only partly extracted, its claim not mapped; v3 arrives
        v2 = await _write(conn, meta, 2, "v2")
        await _extract(conn, v2[0], claim_mapped=None)
        v3 = await _write(conn, meta, 1, "v3")
        assert await _superseded(conn, doc_id) == [v1[0]]
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s",
                            claim) == 1
        assert await _count(conn, "SELECT count(*) FROM chunks WHERE chunk_id = ANY(%s)",
                            v2) == 0
        assert await _current(conn, doc_id) == v3
        # the facts in the table are still v1's, never an unmapped v2's
        assert await _count(conn, "SELECT count(*) FROM raw_claims") == 1

        # when v2 was fully extracted and mapped but not retired before v3, v2's chunk stays
        # superseded too: both versions' facts stay until v3 is mapped
        doc2 = uuid.uuid4()
        meta2 = _meta(doc2)
        w1 = await _write(conn, meta2, 1, "w1")
        w1_claim = await _extract(conn, w1[0], claim_mapped=version)
        w2 = await _write(conn, meta2, 1, "w2")
        w2_claim = await _extract(conn, w2[0], claim_mapped=version)
        w3 = await _write(conn, meta2, 1, "w3")
        assert set(await _superseded(conn, doc2)) == {*w1, *w2}
        assert await _current(conn, doc2) == w3
        for claim_id in (w1_claim, w2_claim):
            assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s",
                                claim_id) == 1
    finally:
        await conn.close()


async def test_fewer_chunks_after_an_update(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        v1 = await _write(conn, _meta(doc_id), 3, "v1")
        await _extract(conn, v1[0], claim_mapped=version)  # one chunk of three carries a fact

        # no unique violation: the superseded ordinal 0 and the current ordinal 0 coexist;
        # the two chunks without a fact are gone
        await _write(conn, _meta(doc_id, title="New title"), 1, "v2")
        cur = await conn.execute(
            "SELECT ordinal, superseded_at IS NOT NULL FROM chunks WHERE doc_id = %s"
            " ORDER BY superseded_at IS NOT NULL, ordinal", (doc_id,))
        assert await cur.fetchall() == [(0, False), (0, True)]
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
        assert await _count(
            conn, "SELECT count(*) FROM chunks WHERE superseded_at IS NOT NULL") == 0
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
        v3 = await _write(conn, meta, 1, "v3")  # v1 and v2 are both superseded
        assert set(await _superseded(conn, doc_id)) == {v1[0], v2[0]}
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
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        await _write(conn, _meta(doc_id), 1, "v1")
        await conn.execute("DROP INDEX chunks_doc_ordinal_current_idx")
        await conn.execute("DROP INDEX chunks_superseded_idx")
        await conn.execute("DROP INDEX chunks_doc_idx")
        await conn.execute("ALTER TABLE chunks DROP COLUMN superseded_at")
        await conn.execute("ALTER TABLE chunks ADD UNIQUE (doc_id, ordinal)")
        await conn.execute("DELETE FROM taxonomy_meta WHERE key = 'schema_version'")
        await init_schema(conn, embed_dim=8)
        assert await _count(conn, "SELECT count(*) FROM chunks WHERE superseded_at IS NULL") == 1
        # the upgrade removed the unique constraint PostgreSQL named for the old inline one
        assert await _count(conn, "SELECT count(*) FROM pg_constraint WHERE contype = 'u'"
                                  " AND conrelid = 'chunks'::regclass") == 0
        # and the unique constraint's index leading with doc_id is replaced by a plain one
        assert await _count(conn, "SELECT count(*) FROM pg_indexes WHERE tablename = 'chunks'"
                                  " AND indexname = 'chunks_doc_idx'") == 1
        first = (await _current(conn, doc_id))[0]
        await _extract(conn, first, claim_mapped=version)
        await _write(conn, _meta(doc_id), 1, "v2")  # ordinal 0 twice: superseded + current
        assert await _count(conn, "SELECT count(*) FROM chunks WHERE doc_id = %s", doc_id) == 2
    finally:
        await conn.close()


# --- unchanged text keeps its chunk, and its facts (a per-chunk rule) ---

def _body(label: str) -> str:
    """A chunk text; the word "quote" is what ``_extract``'s claims quote."""
    return f"T\n\n{label} quote of the chunk"


async def test_identical_chunks_keep_their_facts(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        _, v1 = await _write_texts(conn, meta, [_body("a"), _body("b"), _body("c")])
        claims = [await _extract(conn, c, claim_mapped=version) for c in v1]

        reused, v2 = await _write_texts(conn, meta, [_body("a"), _body("B changed"), _body("c")])
        assert reused == 2
        assert v2[0] == v1[0] and v2[2] == v1[2] and v2[1] != v1[1]
        assert await _superseded(conn, doc_id) == [v1[1]]
        # the kept chunks keep their extraction and claim; the new chunk has neither
        for chunk_id, claim_id in ((v1[0], claims[0]), (v1[2], claims[2])):
            assert await _count(conn, "SELECT count(*) FROM extractions WHERE chunk_id = %s",
                                chunk_id) == 1
            assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s"
                                " AND chunk_id = %s", claim_id, chunk_id) == 1
        assert await _count(conn, "SELECT count(*) FROM extractions WHERE chunk_id = %s",
                            v2[1]) == 0

        # once the new chunk is extracted and mapped, only the old chunk 1 is retired
        await _extract(conn, v2[1], claim_mapped=version)
        assert await retire_superseded(conn, version) == 1
        assert await _superseded(conn, doc_id) == []
        assert await _current(conn, doc_id) == v2
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s",
                            claims[1]) == 0
        assert await _count(conn, "SELECT count(*) FROM raw_claims") == 3
    finally:
        await conn.close()


async def test_cosmetic_changes_reuse_the_chunk_with_the_new_text(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        old = ("T\n\nSee [List Media](list_media.html).\nLast Updated: 2026\\-03\\-26\n"
               "quote one two")
        new = ("T\n\nSee [List Media](./list_media.html).\n\nLast Updated: 2026\\-07\\-31\n"
               "quote one\ntwo")
        _, v1 = await _write_texts(conn, meta, [old])
        claim = await _extract(conn, v1[0], claim_mapped=version, quote="quote one two")

        new_vec = Vector([0.9] * 8)
        reused, v2 = await _write_texts(conn, meta, [new], vec=new_vec, token_count=77)
        assert reused == 1 and v2 == v1
        assert await _superseded(conn, doc_id) == []
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE claim_id = %s",
                            claim) == 1
        cur = await conn.execute("SELECT text, token_count, embedding = %s FROM chunks"
                                 " WHERE chunk_id = %s", (new_vec, v1[0]))
        assert await cur.fetchone() == (new, 77, True)
    finally:
        await conn.close()


async def test_a_quote_no_longer_in_the_text_blocks_reuse(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        _, v1 = await _write_texts(conn, meta, ["T\n\nUse [Guide](guide.html) for the quote."])
        await _extract(conn, v1[0], claim_mapped=version,
                       quote="Use [Guide](guide.html) FOR THE   quote")

        # equal keys (only the link target differs), but the claim's quote is not in the text
        reused, v2 = await _write_texts(
            conn, meta, ["T\n\nUse [Guide](./guide.html) for the quote."])
        assert reused == 0
        assert await _superseded(conn, doc_id) == v1
        assert len(v2) == 1 and v2 != v1

        # the quote check ignores whitespace and case: this version is reused
        await _extract(conn, v2[0], claim_mapped=version, quote="USE [guide](./guide.html)\nFor")
        reused, v3 = await _write_texts(
            conn, meta, ["T\n\nUse [Guide](./guide.html)\n for the quote."])
        assert reused == 1 and v3 == v2
    finally:
        await conn.close()


async def test_reordered_chunks_are_reused(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        _, (a, b) = await _write_texts(conn, meta, [_body("a"), _body("b")])
        for chunk_id in (a, b):
            await _extract(conn, chunk_id, claim_mapped=version)

        # swapped ordinals: no clash on the partial unique index while they move
        reused, v2 = await _write_texts(conn, meta, [_body("b"), _body("a")])
        assert reused == 2 and v2 == [b, a]
        assert await _superseded(conn, doc_id) == []
        assert await _count(conn, "SELECT count(*) FROM raw_claims") == 2
    finally:
        await conn.close()


async def test_a_dropped_chunk_is_kept_only_with_a_mapped_claim(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        _, v1 = await _write_texts(conn, meta, [_body(x) for x in "abcde"])
        await _extract(conn, v1[0], claim_mapped=version)  # mapped: kept
        await _extract(conn, v1[1], claim_mapped=None)  # unmapped only: deleted
        await _extract(conn, v1[2])  # no claim: deleted
        # v1[3] was never extracted: deleted
        await _extract(conn, v1[4], claim_mapped=None)  # one claim unmapped, one mapped: kept
        await _extract(conn, v1[4], claim_mapped=version, prompt_version="claims-v3.0")

        reused, v2 = await _write_texts(conn, meta, [_body("z")])
        assert reused == 0
        assert set(await _superseded(conn, doc_id)) == {v1[0], v1[4]}
        assert await _count(conn, "SELECT count(*) FROM chunks WHERE doc_id = %s", doc_id) == 3
        # the deleted chunks took their extractions and unmapped claims with them
        assert await _count(conn, "SELECT count(*) FROM raw_claims WHERE chunk_id = ANY(%s)",
                            v1[1:4]) == 0

        # a superseded chunk is never deleted by a write: v2 is mapped but not retired
        await _extract(conn, v2[0], claim_mapped=version)
        _, v3 = await _write_texts(conn, meta, [_body("y")])
        assert set(await _superseded(conn, doc_id)) == {v1[0], v1[4], v2[0]}

        # until the current version is extracted and mapped
        await _extract(conn, v3[0], claim_mapped=version)
        assert await retire_superseded(conn, version) == 3
        assert await _superseded(conn, doc_id) == []
        assert await _current(conn, doc_id) == v3
    finally:
        await conn.close()


async def test_write_document_counts_the_chunks_it_reused_and_superseded(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        meta = _meta(uuid.uuid4())
        _, v1 = await _write_texts(conn, meta, [_body(x) for x in "abcd"])
        await _extract(conn, v1[0], claim_mapped=version)  # reused
        await _extract(conn, v1[1], claim_mapped=version)  # dropped, mapped: superseded
        await _extract(conn, v1[2], claim_mapped=None)  # dropped, unmapped: deleted
        # v1[3] was never extracted: deleted

        chunks = [Chunk(section_path=("T",), ordinal=i, text=t, token_count=3)
                  for i, t in enumerate([_body("a"), _body("z")])]
        written = await write_document(conn, meta, "h2", chunks, [VEC] * 2)
        assert (written.reused, written.superseded) == (1, 1)
        # a first write has nothing to reuse or supersede
        first = await write_document(conn, _meta(uuid.uuid4()), "h", chunks, [VEC] * 2)
        assert (first.reused, first.superseded) == (0, 0)
    finally:
        await conn.close()


async def test_a_reused_chunk_takes_the_new_section_path(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        # a heading whose only change is an escaped hyphen: equal keys, another section_path
        _, v1 = await _write_texts(conn, meta, ["T\nPlan 2026\\-03\n\nthe quote text"],
                                   path=("T", "Plan 2026\\-03"))
        await _extract(conn, v1[0], claim_mapped=version)
        reused, v2 = await _write_texts(conn, meta, ["T\nPlan 2026-03\n\nthe quote text"],
                                        path=("T", "Plan 2026-03"))
        assert reused == 1 and v2 == v1
        cur = await conn.execute("SELECT section_path FROM chunks WHERE chunk_id = %s", (v1[0],))
        assert await cur.fetchone() == (["T", "Plan 2026-03"],)
    finally:
        await conn.close()


async def _xmins(conn, doc_id: uuid.UUID) -> dict[uuid.UUID, str]:
    """The row version (``xmin``) of each of the page's chunks: it changes when a row is updated."""
    cur = await conn.execute("SELECT chunk_id, xmin::text FROM chunks WHERE doc_id = %s",
                             (doc_id,))
    return {chunk_id: xmin for chunk_id, xmin in await cur.fetchall()}


async def test_an_exact_reuse_in_the_same_position_leaves_the_row_untouched(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        _, v1 = await _write_texts(conn, meta, [_body("a"), _body("b")])
        for chunk_id in v1:
            await _extract(conn, chunk_id, claim_mapped=version)
        before = await _xmins(conn, doc_id)

        # same text, section path and ordinal; the new embedding differs and is not stored
        new_vec = Vector([0.9] * 8)
        reused, v2 = await _write_texts(conn, meta, [_body("a"), _body("b")], vec=new_vec)
        assert reused == 2 and v2 == v1  # still counted as reused
        assert await _xmins(conn, doc_id) == before
        assert await _count(conn, "SELECT count(*) FROM chunks WHERE doc_id = %s"
                                  " AND embedding = %s", doc_id, VEC) == 2
    finally:
        await conn.close()


async def _updated(conn) -> int:
    """The chunk rows updated so far by the connection's open transaction."""
    return await _count(conn, "SELECT n_tup_upd FROM pg_stat_xact_user_tables"
                              " WHERE relname = 'chunks'")


async def test_a_reused_chunk_is_updated_once_and_only_a_moved_one_takes_the_hop(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        old = "T\n\nSee [the guide](g.html) for the quote."
        new = "T\n\nSee [the guide](./g.html) for the quote."  # same reuse_key, other text
        _, v1 = await _write_texts(conn, meta, [_body("a"), old, _body("c")])
        for chunk_id in v1:
            await _extract(conn, chunk_id, claim_mapped=version)
        before = await _xmins(conn, doc_id)

        # the middle chunk changes in place: one UPDATE; its neighbours are not touched
        new_vec = Vector([0.9] * 8)
        async with conn.transaction():
            reused, v2 = await _write_texts(conn, meta, [_body("a"), new, _body("c")],
                                            vec=new_vec)
            assert await _updated(conn) == 1
        assert reused == 3 and v2 == v1
        after = await _xmins(conn, doc_id)
        assert (after[v1[0]], after[v1[2]]) == (before[v1[0]], before[v1[2]])
        assert after[v1[1]] != before[v1[1]]
        cur = await conn.execute("SELECT text, embedding = %s FROM chunks WHERE chunk_id = %s",
                                 (new_vec, v1[1]))
        assert await cur.fetchone() == (new, True)

        # two chunks swap places: only they are updated
        reused, v3 = await _write_texts(conn, meta, [_body("a"), _body("c"), new])
        assert reused == 3 and v3 == [v1[0], v1[2], v1[1]]
        swapped = await _xmins(conn, doc_id)
        assert swapped[v1[0]] == after[v1[0]]
        assert swapped[v1[1]] != after[v1[1]] and swapped[v1[2]] != after[v1[2]]
    finally:
        await conn.close()


async def test_a_reused_chunk_moves_onto_a_dropped_ordinal_and_a_fresh_one_takes_its_old(
        vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        # the dropped chunk B is superseded (mapped claim) in one document, deleted
        # (unmapped claim) in the other: either way A takes its ordinal 1 and N takes ordinal 0
        for b_claim in (version, None):
            doc_id = uuid.uuid4()
            meta = _meta(doc_id)
            _, (a, b) = await _write_texts(conn, meta, [_body("a"), _body("b")])
            await _extract(conn, a, claim_mapped=version)
            await _extract(conn, b, claim_mapped=b_claim)

            reused, v2 = await _write_texts(conn, meta, [_body("n"), _body("a")])
            assert reused == 1 and v2[1] == a and v2[0] not in (a, b)
            cur = await conn.execute("SELECT chunk_id, ordinal FROM chunks WHERE doc_id = %s"
                                     " AND superseded_at IS NULL ORDER BY ordinal", (doc_id,))
            assert await cur.fetchall() == [(v2[0], 0), (a, 1)]
            assert await _superseded(conn, doc_id) == ([b] if b_claim else [])
    finally:
        await conn.close()


async def test_read_section_of_a_superseded_chunk_stays_in_its_own_write(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        # v1 and v2 are both mapped and neither is retired when v3 arrives
        v1 = await _write(conn, meta, 2, "v1")
        for chunk_id in v1:
            await _extract(conn, chunk_id, claim_mapped=version)
        v2 = await _write(conn, meta, 2, "v2")
        for chunk_id in v2:
            await _extract(conn, chunk_id, claim_mapped=version)
        v3 = await _write(conn, meta, 2, "v3")
        assert set(await _superseded(conn, doc_id)) == {*v1, *v2}

        async def ids(chunk_id: uuid.UUID, expand) -> list[str]:
            return [h.chunk_id for h in await read_section(conn, str(chunk_id), expand)]

        for version_ids, other in ((v1, v2), (v2, v1)):
            assert await ids(version_ids[0], "0") == [str(version_ids[0])]
            assert await ids(version_ids[0], "1") == [str(c) for c in version_ids]
            assert await ids(version_ids[1], "full") == [str(c) for c in version_ids]
            assert not {str(c) for c in other} & set(await ids(version_ids[0], "full"))
        assert await ids(v3[0], "full") == [str(c) for c in v3]
    finally:
        await conn.close()


async def test_read_section_marks_the_chunks_of_an_earlier_version(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        meta = _meta(uuid.uuid4())
        v1 = await _write(conn, meta, 2, "v1")
        for chunk_id in v1:
            await _extract(conn, chunk_id, claim_mapped=version)
        v2 = await _write(conn, meta, 2, "v2")
        # the text of a superseded chunk is no longer on the page: its reader is told
        assert [h.superseded for h in await read_section(conn, str(v1[0]), "full")] == [True] * 2
        assert [h.superseded for h in await read_section(conn, str(v1[1]), "0")] == [True]
        assert [h.superseded for h in await read_section(conn, str(v2[0]), "full")] == [False] * 2
    finally:
        await conn.close()


# --- an extraction and a re-sync of the same page wait for each other ---

URL = "https://docs.example.com/long/path/to/the/guide/page-with-the-details-about-it.html"
OLD_TEXT = f"T\n\nSee [the guide]({URL}) for every repository."
NEW_TEXT = "T\n\nSee [the guide](/g/it.html) for every repository."  # same reuse_key
SPANNING_QUOTE = f"[the guide]({URL}) for every repository"  # in OLD_TEXT, not in NEW_TEXT


async def _wait_for_lock(watcher, waiting) -> None:
    """Until ``waiting``'s backend is blocked on a lock (seen through ``watcher``)."""
    for _ in range(200):
        cur = await watcher.execute("SELECT wait_event_type FROM pg_stat_activity"
                                    " WHERE pid = %s", (waiting.info.backend_pid,))
        if (await cur.fetchone())[0] == "Lock":
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the connection never waited for a lock")


async def test_a_recorded_extraction_waits_for_a_resync_and_gates_on_its_text(vkb_dsn):
    writer, recorder = await connect(vkb_dsn), await connect(vkb_dsn)
    try:
        await _fresh(writer)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        _, [chunk_id] = await _write_texts(writer, meta, [OLD_TEXT])
        # the extraction loaded the chunk before the re-sync below
        loaded = (await extract._load_chunks(recorder, [str(chunk_id)]))[str(chunk_id)]
        outcome = extract._Outcome("done", claims=[{**GOOD, "evidence_quote": SPANNING_QUOTE}])

        async with writer.transaction():  # the re-sync holds the page's chunks until it commits
            reused, _ = await _write_texts(writer, meta, [NEW_TEXT])
            assert reused == 1
            record = asyncio.create_task(extract._record(
                recorder, str(chunk_id), loaded, outcome, settings=_settings(),
                ledger_id="b", batch=True, report=extract.ExtractReport()))
            await _wait_for_lock(writer, recorder)
        await record

        # recorded after the commit, against the new text: the quote is not in it
        assert await _count(recorder, "SELECT count(*) FROM raw_claims") == 0
        assert await _count(recorder, "SELECT count(*) FROM rejected_claims") == 1
    finally:
        await writer.close()
        await recorder.close()


async def test_a_resync_waits_for_an_extraction_in_flight_and_sees_its_claims(vkb_dsn):
    writer, recorder = await connect(vkb_dsn), await connect(vkb_dsn)
    try:
        version = await _fresh(writer)
        doc_id = uuid.uuid4()
        meta = _meta(doc_id)
        _, [chunk_id] = await _write_texts(writer, meta, [OLD_TEXT])

        async with recorder.transaction():  # an extraction result being recorded
            await recorder.execute("SELECT text FROM chunks WHERE chunk_id = %s FOR KEY SHARE",
                                   (chunk_id,))
            async def resync():
                async with writer.transaction():
                    return await _write_texts(writer, meta, [NEW_TEXT])

            resync_task = asyncio.create_task(resync())
            await _wait_for_lock(recorder, writer)
            await _extract(recorder, chunk_id, claim_mapped=version, quote=SPANNING_QUOTE)
        reused, [current] = await resync_task

        # the claim committed first: its quote is not in the new text, so no reuse
        assert reused == 0 and current != chunk_id
        assert await _superseded(writer, doc_id) == [chunk_id]
        assert await _count(writer, "SELECT count(*) FROM raw_claims WHERE chunk_id = %s",
                            chunk_id) == 1
    finally:
        await writer.close()
        await recorder.close()


class _Recorder:
    """A connection that remembers the statements it was asked to run."""

    def __init__(self, conn):
        self.conn, self.statements = conn, []

    async def execute(self, sql, params=None):
        self.statements.append((sql, params))
        return await self.conn.execute(sql, params)


async def test_read_section_context_does_not_scan_chunks(vkb_dsn):
    """``chunks`` holds every chunk of the corpus: the context query of either kind of anchor
    must be able to use an index (``chunks_doc_idx`` or the partial index that matches it),
    never scan the table."""
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        meta = _meta(uuid.uuid4())
        v1 = await _write(conn, meta, 2, "v1")
        for chunk_id in v1:
            await _extract(conn, chunk_id, claim_mapped=version)
        v2 = await _write(conn, meta, 2, "v2")  # v1 is superseded, v2 current
        await conn.execute("SET enable_seqscan = off")  # a tiny table would not need an index
        recorder = _Recorder(conn)
        for anchor in (v2[0], v1[0]):
            for expand in ("1", "full"):
                await read_section(recorder, str(anchor), expand)
                sql, params = recorder.statements[-1]
                cur = await conn.execute("EXPLAIN " + sql, params)
                plan = "\n".join(row[0] for row in await cur.fetchall())
                assert "Seq Scan on chunks" not in plan, (anchor, expand, plan)
    finally:
        await conn.close()


async def test_removing_a_page_does_not_scan_chunks(vkb_dsn):
    """The foreign key cascade from ``documents`` runs ``DELETE FROM ONLY chunks WHERE $1 =
    doc_id`` per removed page: only a full index leading with ``doc_id`` can serve it (the
    partial ones cannot be proven for it), or every removal scans the whole table."""
    conn = await connect(vkb_dsn)  # its own session: the settings below are not pooled away
    try:
        await _fresh(conn)
        await conn.execute("SET enable_seqscan = off")  # a tiny table would not need an index
        await conn.execute("SET plan_cache_mode = force_generic_plan")  # as the cascade plans it
        await conn.execute("PREPARE d(uuid) AS DELETE FROM ONLY chunks WHERE $1 = doc_id")
        cur = await conn.execute(f"EXPLAIN EXECUTE d('{uuid.uuid4()}')")
        plan = "\n".join(row[0] for row in await cur.fetchall())
        assert "Seq Scan on chunks" not in plan, plan
    finally:
        await conn.close()
