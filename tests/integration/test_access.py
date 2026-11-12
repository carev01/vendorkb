"""A role that hides vendors (access, schema.sql row-level security): what it sees, what the
MCP start-up check proves, and that the read paths of the MCP tools still work through it.
Every vendor and claim here is invented, but for the hidden vendor's name."""
import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from tests.integration.test_capabilities_v2 import _claim, _fresh
from vendorkb.access import check_restricted, restrict
from vendorkb.db import connect
from vendorkb.search import common_lexemes, search_docs
from vendorkb.taxonomy import capabilities

ROLE = "kb_restricted_test"


class FakeEmbedder:
    async def embed(self, texts, *, input_type="document"):
        return [[0.1] * 8 for _ in texts]


@pytest.fixture
async def kb(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        version = await _fresh(conn)
        for vendor in ("Cohesity", "Examplevendor"):
            await _claim(conn, "retention.maximum_period", "okta", vendor=vendor, mapped_version=version,
                         quote="synthetic quote")
        await conn.execute("UPDATE chunks SET text = 'immutable backup copies synthetic'")
        await capabilities.build(conn)
        await conn.execute("INSERT INTO offerings (offering_id, vendor, name) VALUES"
                           " ('hid.box', 'Cohesity', 'Hidden Box'), ('ex.box', 'Examplevendor',"
                           " 'Example Box') ON CONFLICT DO NOTHING")
        await conn.execute("INSERT INTO editions (offering_id, edition_id, name) VALUES"
                           " ('hid.box', 'pro', 'Pro'), ('ex.box', 'pro', 'Pro')"
                           " ON CONFLICT DO NOTHING")
        await conn.execute(
            "INSERT INTO aliases (alias, entity_type, scope, entity_id) VALUES"
            " ('hidden box', 'offering', 'Cohesity', 'hid.box'),"
            " ('example box', 'offering', 'Examplevendor', 'ex.box'),"
            " ('hidden pro', 'edition', 'hid.box', 'pro'),"
            " ('example pro', 'edition', 'ex.box', 'pro') ON CONFLICT DO NOTHING")
        await restrict(conn, ROLE, ["cohesity"], "pw-test")
        yield conn, make_conninfo(vkb_dsn, user=ROLE, password="pw-test")
    finally:
        await conn.close()


async def _rows(conn, sql):
    cur = await conn.execute(sql)
    return [r[0] for r in await cur.fetchall()]


async def test_the_role_sees_no_row_of_the_hidden_vendor(kb):
    owner, dsn = kb
    assert set(await _rows(owner, "SELECT vendor FROM documents")) == {"Cohesity",
                                                                     "Examplevendor"}
    conn = await connect(dsn)
    try:
        for table in ("documents", "raw_claims", "capabilities", "offerings"):
            assert set(await _rows(conn, f"SELECT DISTINCT lower(vendor) FROM {table}")) == {
                "examplevendor"}, table
        assert await _rows(conn, "SELECT count(*) FROM chunks") == [1]
        editions = await _rows(conn, "SELECT offering_id FROM editions")
        assert "ex.box" in editions and "hid.box" not in editions
        aliases = set(await _rows(conn, "SELECT alias FROM aliases WHERE entity_type IN"
                                        " ('offering', 'edition')"))
        assert {"example box", "example pro"} <= aliases
        assert not aliases & {"hidden box", "hidden pro"}
        assert await _rows(conn, "SELECT count(*) FROM aliases WHERE entity_type"
                                 " = 'dimension'") != [0]  # the taxonomy itself is shown
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await conn.execute("SELECT 1 FROM extractions")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            await conn.execute("DELETE FROM access_hidden_vendors")
    finally:
        await conn.close()


async def test_the_mcp_read_paths_work_through_the_role(kb):
    _, dsn = kb
    conn = await connect(dsn)
    try:
        await common_lexemes(conn)  # through vendorkb_chunk_lexeme_freqs
        res = await search_docs(conn, "immutable backup", embedder=FakeEmbedder(),
                                reranker=None, use_rerank=False)
        assert [h.vendor for h in res.hits] == ["Examplevendor"]
        res = await search_docs(conn, "immutable backup", embedder=FakeEmbedder(),
                                reranker=None, use_rerank=False, vendor=["Cohesity"])
        assert res.hits == []
        caps = await capabilities.query(conn, capabilities.CapFilter(
            dimension="retention.maximum_period"), limit=10)
        assert {f.vendor for f in caps.facts} == {"Examplevendor"}
    finally:
        await conn.close()


async def test_the_start_up_check(kb):
    owner, dsn = kb
    assert await check_restricted(dsn, ["Cohesity"]) == ROLE
    with pytest.raises(RuntimeError, match="does not hide"):
        await check_restricted(dsn, ["Cohesity", "Examplevendor"])
    owner_dsn = owner.info.dsn + " password=" + owner.info.password
    with pytest.raises(RuntimeError, match="bypasses row-level security"):
        await check_restricted(owner_dsn, ["Cohesity"])


async def test_restrict_replaces_the_list_and_keeps_the_password(kb):
    owner, dsn = kb
    await restrict(owner, ROLE, ["Examplevendor"], None)
    conn = await connect(dsn)
    try:
        assert set(await _rows(conn, "SELECT vendor FROM documents")) == {"Cohesity"}
    finally:
        await conn.close()
    with pytest.raises(ValueError, match="at least one vendor"):
        await restrict(owner, ROLE, [" "], None)
    with pytest.raises(ValueError, match="give it a password"):
        await restrict(owner, "kb_never_made", ["Cohesity"], None)
    with pytest.raises(ValueError, match="plain lowercase identifier"):
        await restrict(owner, "Bad; DROP", ["Cohesity"], None)


async def test_a_which_vendors_question_through_the_role(kb):
    """The scope index and the cross-vendor retrieval: asked for the hidden vendor by name,
    the role has no rows of it to give."""
    from tests.integration.test_enum_facts import FakeReranker
    from vendorkb.modes import enumerate_vendors
    from vendorkb.scope import scope_index_from_db

    _, dsn = kb
    conn = await connect(dsn)
    try:
        index = await scope_index_from_db(conn)
        assert "Cohesity" not in index.vendors and "Examplevendor" in index.vendors
        res = await enumerate_vendors(conn, "Which vendors keep backups longest?",
                                      embedder=FakeEmbedder(),
                                      reranker=FakeReranker(lambda q, t: 0.9),
                                      vendors=["Cohesity", "Examplevendor"], min_score=0.5)
        assert res.kept == ["Examplevendor"] and res.empty == ["Cohesity"]
        assert {h.vendor for h in res.hits} == {"Examplevendor"}
    finally:
        await conn.close()
