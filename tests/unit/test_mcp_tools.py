"""The MCP tools (mcp_server) through an in-process client, over a fake backend."""
from pathlib import Path

import pytest
from mcp import Client

from vendorkb import mcp_server as ms
from vendorkb.answer import citation
from vendorkb.search import Hit, SearchResult
from vendorkb.taxonomy import load
from vendorkb.taxonomy.capabilities import CapResult, Conflict, Fact
from vendorkb.topics import Cell, Grid, render_grid

SHIPPED = Path(__file__).parent.parent.parent / "taxonomy" / "taxonomy.yaml"


@pytest.fixture(scope="module")
def ransomware() -> load.Topic:
    """The shipped taxonomy's ransomware protection topic."""
    [topic] = load.validate(load.read_doc(SHIPPED)).topics
    return topic


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
                doc_version="2026-11-01T00:00:00+00:00", taxonomy_version="2-abc",
                evidence_superseded=False)
    return Fact(**{**base, **kw})


def _hit(n: int, text: str = "Passage text.", superseded: bool = False) -> Hit:
    return Hit(chunk_id=f"k{n}", doc_id=f"d{n}", text=text, section_path=["Guide", "RDS"],
               vendor="Examplevendor", product="Example Backup", title=f"Page {n}",
               url=f"https://docs.example/{n}", rrf_rank=n, rerank_score=0.9,
               superseded=superseded)


class FakeBackend:
    def __init__(self, *, caps=None, hits=None, answer=None, fail=None, topics=()):
        self.calls: list[tuple] = []
        self.caps = caps or CapResult()  # or a function of the filter
        self.known_topics = list(topics)
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
        return self.caps(f) if callable(self.caps) else self.caps

    async def capabilities_many(self, filters, limit):
        self.calls.append(("capabilities_many", filters, limit))
        if self.fail:
            raise self.fail
        out = []
        for f in filters:
            out.append(self.caps(f) if callable(self.caps) else self.caps)
            if out[-1].unresolved:  # like DbBackend: the rest would fail the same way
                break
        return out

    async def topics(self):
        self.calls.append(("topics",))
        return self.known_topics

    async def search(self, query, vendor, product, k):
        self.calls.append(("search", query, vendor, product, k))
        return SearchResult(self.hits, True)

    async def section(self, chunk_id, expand):
        self.calls.append(("section", chunk_id, expand))
        if chunk_id == "old":  # a chunk of an earlier version of its page
            return [_hit(1, "Before.", True), _hit(2, "The chunk.", True)]
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
    caps = tools["query_capabilities"]
    assert "topic" in caps.input_schema["properties"]
    assert "`topic`" in caps.description and "grouped by aspect" in caps.description
    assert "exclusive with `dimension`" in caps.description
    assert "topic" in tools["list_dimensions"].description


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
    assert limit == 20
    assert (f.dimension, f.target, f.vendor) == ("immutability", "RDS", None)


async def test_long_quotes_are_trimmed_in_the_text_only():
    long = "x" * (ms.QUOTE_SHOWN + 100)
    async with Client(ms.build_server(FakeBackend(caps=CapResult(
            facts=[_fact(evidence_quote=long)])))) as c:
        r = await c.call_tool("query_capabilities", {"vendor": "Examplevendor"})
    assert f'quote: "{"x" * ms.QUOTE_SHOWN} […]"' in r.content[0].text
    assert r.structured_content["facts"][0]["evidence_quote"] == long


async def test_unresolved_targets_show_the_documentation_wording():
    caps = CapResult(unresolved_target=[_fact(target_id=None, scope="target_unmapped",
                                              target_raw="Lotus Notes", support="not_supported")])
    async with Client(ms.build_server(FakeBackend(caps=caps))) as c:
        r = await c.call_tool("query_capabilities", {"vendor": "Examplevendor"})
    text = r.content[0].text
    assert "Facts about a workload the taxonomy does not list" in text
    assert '| "Lotus Notes" (not in the taxonomy) | NOT_SUPPORTED' in text
    assert r.structured_content["unresolved_target_total"] == 1


def test_superseded_fact_is_marked():
    marked = ms.format_fact(1, _fact(evidence_superseded=True))
    assert "from an earlier version of the page" in marked
    assert "(chunk k1;" in marked and "https://docs.example/rds" in marked
    assert "earlier version" not in ms.format_fact(1, _fact())


def test_superseded_fact_does_not_claim_a_newer_document_contradicts_it():
    # not current because its page was re-synced, not because another document disagrees
    text = ms.format_fact(1, _fact(evidence_superseded=True, is_current=False))
    assert "from an earlier version of the page" in text
    assert "a newer document says otherwise" not in text
    # a current-version row that a newer document outranks keeps its phrase
    assert "a newer document says otherwise" in ms.format_fact(1, _fact(is_current=False))


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
        assert "or topic" in r.content[0].text
        assert backend.calls == []
        r = await c.call_tool("query_capabilities", {"dimension": "x"})
        assert r.is_error and "RuntimeError: db down" in r.content[0].text


async def test_query_capabilities_takes_one_vendor_or_several():
    backend = FakeBackend(caps=CapResult(facts=[_fact()]))
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {"vendor": ["Examplevendor", " Other "],
                                                     "dimension": "immutability"})
        assert not r.is_error
        await c.call_tool("query_capabilities", {"vendor": "Examplevendor"})
        await c.call_tool("query_capabilities", {"vendor": [], "dimension": "immutability"})
    assert [call[1].vendor for call in backend.calls] == [
        ("Examplevendor", "Other"), ("Examplevendor",), None]


async def test_a_result_section_named_as_the_target_is_ignored_with_a_note():
    backend = FakeBackend(caps=CapResult(facts=[_fact()]))
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {"target": "product-wide",
                                                     "dimension": "immutability"})
        assert not r.is_error
        assert r.content[0].text.startswith("Note: 'product-wide' is a section of the results")
        assert backend.calls[-1][1].target is None
        assert "section of the results" in r.structured_content["note"]
        # with nothing else to filter on, it is the usual missing-filter error, explained
        r = await c.call_tool("query_capabilities", {"target": "Product wide"})
        assert r.is_error and "section of the results" in r.content[0].text
        assert "at least one filter" in r.content[0].text
    assert len(backend.calls) == 1


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
        assert "earlier version" not in r.content[0].text
        assert [x["superseded"] for x in r.structured_content["chunks"]] == [False] * 3
        r = await c.call_tool("read_section", {"chunk_id": "gone"})
        assert r.is_error and "search again" in r.content[0].text


async def test_read_section_of_an_earlier_version_says_so():
    async with Client(ms.build_server(FakeBackend())) as c:
        r = await c.call_tool("read_section", {"chunk_id": "old"})
    assert not r.is_error
    assert "source: https://docs.example/1 (from an earlier version of the page)" \
        in r.content[0].text
    assert "Before.\n\nThe chunk." in r.content[0].text
    assert [x["superseded"] for x in r.structured_content["chunks"]] == [True, True]


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


def _grid_answer(topic: load.Topic) -> dict:
    """An answer_question envelope for a grid of 30 vendors by the topic's aspects, one cell
    short of full (239 cells), its pages long URLs, two cells to a page."""
    vendors = [f"Vendor {n:02d}" for n in range(30)]
    cells = {}
    for n, (vendor, aspect) in enumerate((v, a) for v in vendors for a in topic.aspects):
        url = (f"https://help.example-vendor-documentation.com/en-us/{vendor.replace(' ', '-')}"
               f"/backup/security/topic-{n // 2}.html")
        hit = Hit(chunk_id=f"k{n}", doc_id=f"d{n}", text="t", section_path=[], vendor=vendor,
                  product="P", title=f"A rather long page title, number {n}", url=url,
                  rrf_rank=0, rerank_score=None)
        cells[(vendor, aspect.id)] = Cell("supported", hit, False)
    del cells[(vendors[-1], topic.aspects[-1].id)]
    grid = Grid(vendors=vendors, cells=cells, facts_as_of=None, no_fact=[])
    table, hits = render_grid(grid, topic)
    return {"answer": f"Thirty vendors have facts for every aspect.\n\n{table}",
            "shape": "enumeration", "topic": topic.id, "scope": {}, "uncited": False,
            "error": None, "citations": [citation(n, h) for n, h in enumerate(hits, 1)]}


async def test_a_grid_answer_lists_each_page_once_within_a_cap(ransomware):
    out = _grid_answer(ransomware)
    assert len(out["citations"]) == 239
    async with Client(ms.build_server(FakeBackend(answer=out))) as c:
        r = await c.call_tool("answer_question", {"question": "Which vendors protect backups?"})
    text = r.content[0].text
    assert len(text) <= 14_000
    answer, sources = text.split("\n\nSources:\n")
    assert answer == out["answer"]
    assert len("Sources:\n" + sources) <= ms.GRID_SOURCES_CHARS == 8000
    *lines, more = sources.split("\n")
    # one line per page, all its markers, no title
    by_url: dict[str, list[int]] = {}
    for c in out["citations"]:
        by_url.setdefault(c["url"], []).append(c["marker"])
    expected = [f"[{', '.join(map(str, m))}] {url}" for url, m in by_url.items()]
    assert lines == expected[:len(lines)] and lines[0].startswith("[1, 2] https://")
    assert not any("page title" in line for line in lines)
    assert more == (f"… {len(expected) - len(lines)} more sources: query_capabilities("
                    f"topic=\"ransomware_protection\") lists the cited facts.")
    assert len(expected) - len(lines) > 0
    # the structured content keeps every citation
    assert len(r.structured_content["citations"]) == 239


async def test_a_short_grid_answer_lists_every_page(ransomware):
    out = _grid_answer(ransomware)
    out["citations"] = out["citations"][:5]
    async with Client(ms.build_server(FakeBackend(answer=out))) as c:
        r = await c.call_tool("answer_question", {"question": "Which vendors protect backups?"})
    sources = r.content[0].text.split("\n\nSources:\n")[1].split("\n")
    # under the cap: every page, and no pointer line
    assert [line.rsplit(" ", 1)[0] for line in sources] == ["[1, 2]", "[3, 4]", "[5]"]


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


def _topic_call(backend: FakeBackend) -> tuple:
    """The one capabilities_many call of a topic query: (name, filters, limit)."""
    [call] = [c for c in backend.calls if c[0] == "capabilities_many"]
    assert not [c for c in backend.calls if c[0] == "capabilities"]
    return call


async def test_query_by_topic_groups_facts_by_aspect(ransomware):
    aspects = ransomware.aspects
    backend = FakeBackend(caps=CapResult(facts=[_fact()]), topics=[ransomware])
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {
            "topic": "ransomware protection", "vendor": "Examplevendor", "target": "RDS",
            "support": "supported", "offering": "enterprise", "limit": 20})
    assert not r.is_error
    _, filters, limit = _topic_call(backend)  # one call, every aspect's filter in order
    assert [f.dimension_ids for f in filters] == [a.dimensions for a in aspects]
    assert limit == max(1, 20 // len(aspects))
    for f in filters:
        assert (f.vendor, f.target, f.support, f.offering, f.dimension) == (
            ("Examplevendor",), "RDS", "supported", "enterprise", None)
    text = r.content[0].text
    assert [x for x in text.splitlines() if x.startswith("## ")] == [
        f"## {a.name}" for a in aspects]
    # one numbering across the aspects: every [N] points at one fact
    assert [f"[{n}] Examplevendor" in text for n in (1, len(aspects), len(aspects) + 1)] == [
        True, True, False]
    sc = r.structured_content
    assert sc["topic"] == ransomware.id
    assert [(a["id"], a["name"], tuple(a["dimensions"])) for a in sc["aspects"]] == [
        (a.id, a.name, a.dimensions) for a in aspects]
    assert all(a["facts_total"] == 1 and len(a["facts"]) == 1 for a in sc["aspects"])
    assert sc["aspects"][0]["facts"][0]["evidence_superseded"] is False


async def test_each_aspect_gets_at_least_one_fact(ransomware):
    backend = FakeBackend(topics=[ransomware])
    async with Client(ms.build_server(backend)) as c:
        await c.call_tool("query_capabilities", {"topic": "ransomware", "limit": 3})
    _, filters, limit = _topic_call(backend)
    assert (len(filters), limit) == (len(ransomware.aspects), 1)


async def test_an_aspect_without_facts_keeps_its_heading_and_claims_no_absence(ransomware):
    first = ransomware.aspects[0].dimensions

    def caps(f):
        if f.dimension_ids == first:
            return CapResult(facts=[_fact()])
        return CapResult(hint="No capability facts match. Search the document chunks")

    backend = FakeBackend(caps=caps, topics=[ransomware])
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {"topic": "ransomware",
                                                     "vendor": "Examplevendor"})
    assert not r.is_error
    text = r.content[0].text
    n = len(ransomware.aspects)
    assert [x for x in text.splitlines() if x.startswith("## ")] == [
        f"## {a.name}" for a in ransomware.aspects]
    assert text.count("No facts. No capability facts match. Search") == n - 1
    assert "not supported" not in text.lower() and "lacks" not in text
    assert [bool(a["facts"]) for a in r.structured_content["aspects"]] == [True] + [False] * (n - 1)


async def test_topic_resolves_by_id_name_trigger_and_separators(ransomware):
    backend = FakeBackend(topics=[ransomware])
    async with Client(ms.build_server(backend)) as c:
        for term in ("ransomware_protection", "Ransomware-Protection", "  RANSOMWARE  protection ",
                     "ransomware", "Cyber-Attack", "protection"):  # "protection": a whole word
            r = await c.call_tool("query_capabilities", {"topic": term})
            assert not r.is_error, term
            assert r.structured_content["topic"] == "ransomware_protection", term


async def test_topic_and_dimension_are_exclusive(ransomware):
    backend = FakeBackend(topics=[ransomware])
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {"topic": "ransomware",
                                                     "dimension": "immutability"})
    assert r.is_error
    assert "topic" in r.content[0].text and "dimension" in r.content[0].text
    assert backend.calls == []


async def test_unknown_topic_lists_the_known_ones(ransomware):
    backend = FakeBackend(topics=[ransomware])
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {"topic": "tape drives"})
    assert r.is_error
    text = r.content[0].text
    assert "'tape drives'" in text and "ransomware_protection (Ransomware protection)" in text
    assert [c[0] for c in backend.calls] == ["topics"]  # no capabilities query


async def test_a_topic_failure_is_reported(ransomware):
    backend = FakeBackend(topics=[ransomware], fail=RuntimeError("db down"))
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {"topic": "ransomware"})
    assert r.is_error and "RuntimeError: db down" in r.content[0].text


async def test_list_dimensions_shows_matching_topics(ransomware):
    backend = FakeBackend(topics=[ransomware])
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("list_dimensions", {"query": "ransomware"})
    assert not r.is_error
    text = r.content[0].text
    assert text.startswith("Topic ransomware_protection — Ransomware protection:\n- ")
    for a in ransomware.aspects:
        assert f"- {a.name}: {', '.join(a.dimensions)}\n" in text + "\n"
    # then the usual listing
    assert text.endswith("security.immutability — Immutability: Copies that cannot be changed."
                         " (aliases: immutable backups)")
    sc = r.structured_content
    assert [d["id"] for d in sc["dimensions"]] == ["security.immutability"]
    [t] = sc["topics"]
    assert (t["id"], t["name"]) == ("ransomware_protection", "Ransomware protection")
    assert [a["id"] for a in t["aspects"]] == [a.id for a in ransomware.aspects]
    assert ("dimensions", "ransomware", None, 25) in backend.calls


@pytest.mark.parametrize("args", [{}, {"query": "  "}, {"family": "security"},
                                  {"query": "retention"},
                                  {"query": "immutability"},  # an aspect's name, not the topic's
                                  {"query": "Air gap / isolated copies"}])
async def test_list_dimensions_without_a_topic_match_is_unchanged(ransomware, args):
    async with Client(ms.build_server(FakeBackend(topics=[ransomware]))) as c:
        r = await c.call_tool("list_dimensions", args)
    assert r.content[0].text == ("security.immutability — Immutability: Copies that cannot be"
                                 " changed. (aliases: immutable backups)")
    assert list(r.structured_content) == ["dimensions"]


async def test_an_aspect_name_does_not_resolve_a_topic(ransomware):
    backend = FakeBackend(topics=[ransomware])
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {"topic": "Immutability / WORM"})
    assert r.is_error and "unknown topic" in r.content[0].text
    assert [c[0] for c in backend.calls] == ["topics"]


async def test_unresolved_filters_are_reported_once_and_the_other_aspects_do_not_run(ransomware):
    hint = "No taxonomy entity matches target 'tape' (by id or alias). Search the documents"
    backend = FakeBackend(caps=CapResult(hint=hint, unresolved=["target 'tape'"]),
                          topics=[ransomware])
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {"topic": "ransomware", "target": "tape"})
    assert not r.is_error
    assert r.content[0].text == f"No facts. {hint}"  # once, with no aspect headings
    sc = r.structured_content
    assert (sc["topic"], sc["aspects"], sc["hint"]) == ("ransomware_protection", [], hint)
    # the backend was asked for every aspect and stopped at the first result
    [(_, filters, _)] = [c for c in backend.calls if c[0] == "capabilities_many"]
    assert len(filters) == len(ransomware.aspects)


async def test_a_topic_without_aspects_does_not_divide_by_zero():
    empty = load.Topic("empty_topic", "Empty topic", ("empty",), ())
    backend = FakeBackend(topics=[empty])
    async with Client(ms.build_server(backend)) as c:
        r = await c.call_tool("query_capabilities", {"topic": "empty", "limit": 7})
    assert not r.is_error
    assert _topic_call(backend)[1:] == ([], 7)
    assert r.structured_content["aspects"] == []


async def test_a_topic_fact_on_an_earlier_page_version_is_marked(ransomware):
    caps = CapResult(facts=[_fact(evidence_superseded=True, is_current=False)])
    async with Client(ms.build_server(FakeBackend(caps=caps, topics=[ransomware]))) as c:
        r = await c.call_tool("query_capabilities", {"topic": "ransomware"})
    assert "(from an earlier version of the page)" in r.content[0].text
    assert r.structured_content["aspects"][0]["facts"][0]["evidence_superseded"] is True


def test_the_instructions_name_the_topic_route():
    assert "query_capabilities(topic=...)" in ms.INSTRUCTIONS
    assert "list_dimensions with the subject shows the topic's aspects" in ms.INSTRUCTIONS
