"""The MCP tools (mcp_server) through an in-process client, over a fake backend."""
import pytest
from mcp import Client

from vendorkb import mcp_server as ms
from vendorkb.search import Hit, SearchResult
from vendorkb.taxonomy.capabilities import CapResult, Conflict, Fact


def _fact(**kw) -> Fact:
    base = dict(claim_id="c1", vendor="Examplevendor", product="Example Backup",
                offering_id=None, edition_id=None, dimension_id="security.immutability",
                dimension_family="security", class_id="cloud.database", target_id="aws.rds",
                variant_id=None, object_type_id=None, kind="capability", support="supported",
                availability="unstated", default_state=None, value=None, unit=None, bound=None,
                conditions=None, scope="product", conflict_group=None, is_current=True,
                capability_raw="immutable snapshots", target_raw="Amazon RDS",
                evidence_quote="Snapshots of  RDS\ninstances are immutable", chunk_id="k1",
                url="https://docs.example/rds", title="Protecting RDS",
                doc_version="2026-11-01T00:00:00+00:00", taxonomy_version="2-abc")
    return Fact(**{**base, **kw})


def _hit(n: int, text: str = "Passage text.") -> Hit:
    return Hit(chunk_id=f"k{n}", doc_id=f"d{n}", text=text, section_path=["Guide", "RDS"],
               vendor="Examplevendor", product="Example Backup", title=f"Page {n}",
               url=f"https://docs.example/{n}", rrf_rank=n, rerank_score=0.9)


class FakeBackend:
    def __init__(self, *, caps=None, hits=None, answer=None, fail=None):
        self.calls: list[tuple] = []
        self.caps = caps or CapResult()
        self.hits = hits or []
        self._answer = answer or {}
        self.fail = fail

    async def vendors(self):
        return [{"vendor": "Examplevendor", "documents": 3, "capability_facts": 7,
                 "products": [{"product": "Example Backup", "documents": 3}]}]

    async def dimensions(self, query, family, limit):
        self.calls.append(("dimensions", query, family, limit))
        return [{"id": "security.immutability", "family": "security", "name": "Immutability",
                 "definition": "Copies that cannot be changed.", "applies_to": None,
                 "value_unit": None, "aliases": ["immutable backups"]}]

    async def targets(self, query, limit):
        self.calls.append(("targets", query, limit))
        return [{"id": "aws.rds", "name": "Amazon RDS", "class": "cloud.database",
                 "parent": None, "aliases": ["RDS"]}]

    async def capabilities(self, f, limit):
        self.calls.append(("capabilities", f, limit))
        if self.fail:
            raise self.fail
        return self.caps

    async def search(self, query, vendor, product, k):
        self.calls.append(("search", query, vendor, product, k))
        return SearchResult(self.hits, True)

    async def section(self, chunk_id, expand):
        self.calls.append(("section", chunk_id, expand))
        return [_hit(1, "Before."), _hit(2, "The chunk."), _hit(3, "After.")] \
            if chunk_id == "k2" else []

    async def answer(self, question):
        self.calls.append(("answer", question))
        return self._answer


async def test_the_tools_are_listed_read_only():
    async with Client(ms.build_server(FakeBackend())) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == {"list_vendors", "list_dimensions", "list_targets",
                          "query_capabilities", "search_docs", "read_section",
                          "answer_question"}
    assert all(t.annotations.read_only_hint for t in tools.values())


async def test_query_capabilities_cites_every_fact():
    caps = CapResult(
        facts=[_fact(), _fact(claim_id="c2", support="partial", value=35, unit="days",
                              bound="max", conditions="Enterprise edition", is_current=False,
                              conflict_group="g1")],
        product_wide=[_fact(claim_id="c3", target_id=None, dimension_id="retention.maximum")],
        conflicts=[Conflict("g1", [_fact(claim_id="c2"), _fact(claim_id="c9")], size=7)])
    backend = FakeBackend(caps=caps)
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {"dimension": "immutability",
                                                     "target": "RDS"})
    assert not r.is_error
    text = r.content[0].text
    assert "[1] Examplevendor · Example Backup | security.immutability | aws.rds | SUPPORTED" \
        in text
    assert 'quote: "Snapshots of RDS instances are immutable"' in text  # whitespace collapsed
    assert "source: Protecting RDS — https://docs.example/rds (chunk k1;" in text
    assert "value max 35 days; conditions: Enterprise edition" in text
    assert "a newer document says otherwise" in text
    assert "Product-wide facts (no specific workload): 1" in text and "[3]" in text
    assert "1 conflict group(s)" in text
    assert r.structured_content["facts_total"] == 2
    assert r.structured_content["conflicts"] == [{"conflict_group": "g1", "size": 7,
                                                  "claim_ids": ["c2", "c9"]}]
    [(_, f, limit)] = backend.calls
    assert limit == 25
    assert (f.dimension, f.target, f.vendor) == ("immutability", "RDS", None)


async def test_query_capabilities_limit_and_hint():
    # the backend already returns at most `limit` rows per group; totals keep the full counts
    capped = CapResult(facts=[_fact(claim_id=f"c{i}") for i in range(2)],
                       totals={"facts": 5000, "product_level": 0, "product_wide": 0})
    backend = FakeBackend(caps=capped)
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {"vendor": "Examplevendor", "limit": 2})
    assert "Facts: 5000 (first 2 shown)" in r.content[0].text
    assert len(r.structured_content["facts"]) == 2
    assert r.structured_content["facts_total"] == 5000 and backend.calls[-1][2] == 2
    empty = CapResult(hint="no fact for target 'tape'; search the documentation instead")
    async with Client(ms.build_server(FakeBackend(caps=empty))) as c:
        r = await c.call_tool("query_capabilities", {"target": "tape"})
    assert r.content[0].text.startswith("No facts. no fact for target 'tape'")


async def test_query_capabilities_needs_a_filter_and_reports_failures():
    backend = FakeBackend(fail=RuntimeError("db down"))
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {})
        assert r.is_error and "at least one filter" in r.content[0].text
        assert backend.calls == []
        r = await c.call_tool("query_capabilities", {"dimension": "x"})
        assert r.is_error and "RuntimeError: db down" in r.content[0].text


async def test_search_docs_and_read_section():
    backend = FakeBackend(hits=[_hit(1), _hit(2, "x" * (ms.PASSAGE_CHARS + 50))])
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("search_docs", {"query": "rds immutability",
                                              "vendor": ["Examplevendor"], "k": 99})
        text = r.content[0].text
        assert "[1] Examplevendor · Example Backup — Page 1 › Guide › RDS" in text
        assert "source: https://docs.example/1 (chunk k1)" in text
        assert text.endswith(" […] (read_section k2 for the rest)")  # trimmed in the text only
        assert len(r.structured_content["hits"][1]["text"]) == ms.PASSAGE_CHARS + 50
        assert backend.calls[-1] == ("search", "rds immutability", ["Examplevendor"], None,
                                     ms.MAX_K)
        r = await c.call_tool("read_section", {"chunk_id": "k2"})
        assert "Before.\n\nThe chunk.\n\nAfter." in r.content[0].text
        assert backend.calls[-1] == ("section", "k2", "1")
        r = await c.call_tool("read_section", {"chunk_id": "gone"})
        assert r.is_error and "search again" in r.content[0].text


async def test_answer_question_lists_its_sources():
    out = {"answer": "Snapshots are immutable [1].", "shape": "single", "scope": {},
           "uncited": False, "error": None,
           "citations": [{"marker": 1, "chunk_id": "k1", "url": "https://docs.example/1",
                          "title": "Page 1", "section_path": ["Guide"], "text": "..."}]}
    async with Client(ms.build_server(FakeBackend(answer=out))) as c:
        r = await c.call_tool("answer_question", {"question": "Is RDS immutable?"})
    assert r.content[0].text == ("Snapshots are immutable [1].\n\nSources:\n"
                                 "[1] Page 1 — https://docs.example/1")
    assert r.structured_content["citations"][0]["chunk_id"] == "k1"
    assert "text" not in r.structured_content["citations"][0]


@pytest.mark.parametrize("tool, args", [("list_dimensions", {"query": "immutable"}),
                                        ("list_targets", {"query": "rds"}),
                                        ("list_vendors", {})])
async def test_browse_tools(tool, args):
    async with Client(ms.build_server(FakeBackend())) as c:
        r = await c.call_tool(tool, args)
    assert not r.is_error
    text = r.content[0].text
    assert {"list_dimensions": "security.immutability — Immutability: Copies that cannot be"
                               " changed. (aliases: immutable backups)",
            "list_targets": "aws.rds — Amazon RDS [class cloud.database] (aliases: RDS)",
            "list_vendors": "Examplevendor — 3 documents, 7 capability facts; products:"
                            " Example Backup (3)"}[tool] == text
