"""DbBackend (the MCP tools' data access) against a real Postgres, on the fixture taxonomy.

Every claim, quote and name here is invented for the tests."""
import pytest

from tests.integration.test_capabilities_v2 import _claim, _fresh
from vendorkb import mcp_backend
from vendorkb.config import VendorkbSettings
from vendorkb.db import connect
from vendorkb.taxonomy import capabilities
from vendorkb.taxonomy.capabilities import CapFilter


class _Fake:
    """An Embedder/Reranker stand-in: async context manager, fixed vectors and scores."""

    def __init__(self, settings):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def embed(self, texts, *, input_type="document"):
        return [[0.1] * 8 for _ in texts]

    async def rerank(self, query, documents, top_k):
        return [(i, 1.0 - i / 10) for i in range(min(top_k, len(documents)))]


@pytest.fixture
def backend(vkb_dsn, monkeypatch):
    monkeypatch.setattr(mcp_backend, "Embedder", _Fake)
    monkeypatch.setattr(mcp_backend, "Reranker", _Fake)
    return mcp_backend.DbBackend(VendorkbSettings(_env_file=None, vendorkb_dsn=vkb_dsn,
                                                  embed_dim=8))


async def test_browse_query_search_and_read(vkb_dsn, backend):
    conn = await connect(vkb_dsn)
    try:
        v = await _fresh(conn)
        await _claim(conn, "coverage.backup", "okta", class_id="saas.identity",
                     object_type="user", mapped_version=v, quote="Okta users are backed up",
                     url="http://docs/okta", title="Okta backup")
        await _claim(conn, "retention.maximum_period", vendor="othervendor",
                     product="Other Suite", mapped_version=v)
        await capabilities.build(conn)
    finally:
        await conn.close()

    vendors = {x["vendor"]: x for x in await backend.vendors()}
    assert vendors["examplevendor"]["documents"] == 1
    assert vendors["examplevendor"]["capability_facts"] == 1
    assert vendors["othervendor"]["products"] == [{"product": "Other Suite", "documents": 1}]

    dims = await backend.dimensions("Backed  Up Objects", None, 10)  # an alias, normalized
    assert dims[0]["id"] == "coverage.backup" and "backed up objects" in dims[0]["aliases"]
    family = await backend.dimensions(None, "coverage", 50)
    assert family and {d["family"] for d in family} == {"coverage"}
    assert [d["id"] for d in await backend.dimensions("retention", None, 10)] == [
        "retention.maximum_period"]  # a substring of the id / name
    targets = await backend.targets("okta", 10)
    assert targets[0]["id"] == "okta" and targets[0]["class"] == "saas.identity"
    assert await backend.targets("no such workload", 10) == []

    res = await backend.capabilities(CapFilter(dimension="backed up objects", target="Okta"))
    [fact] = res.facts
    assert (fact.vendor, fact.url, fact.evidence_quote) == (
        "examplevendor", "http://docs/okta", "Okta users are backed up")

    found = await backend.search("synthetic text", ["examplevendor"], None, 5)
    assert found.rerank_ok and [h.vendor for h in found.hits] == ["examplevendor"]
    [hit] = await backend.section(found.hits[0].chunk_id, "1")
    assert hit.chunk_id == found.hits[0].chunk_id and hit.url == "http://docs/okta"
    assert await backend.section("00000000-0000-0000-0000-000000000000", "0") == []
