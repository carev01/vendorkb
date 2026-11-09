import types
from pathlib import Path

import pytest

from vendorkb import answer as ans
from vendorkb import topics as topics_mod
from vendorkb.answer import REFUSAL, build_prompt, resolve_markers
from vendorkb.config import VendorkbSettings
from vendorkb.modes import ComparisonResult
from vendorkb.scope import ScopeIndex
from vendorkb.search import Hit, SearchResult
from vendorkb.taxonomy import load
from vendorkb.topics import Cell, Grid, TopicHits


def _hit(n, vendor="Veeam"):
    return Hit(chunk_id=f"c{n}", doc_id=f"d{n}", text=f"Title {n}\nA > B\n\nbody {n}",
               section_path=["A", "B"], vendor=vendor, product="VBR", title=f"Title {n}",
               url=f"https://docs.example/{n}", rrf_rank=n, rerank_score=0.5)


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, cost=0.001)
        msg = types.SimpleNamespace(content=r)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


def _settings():
    return VendorkbSettings(_env_file=None)


_LOAD_TOPICS = ans.load_topics  # the fixture below replaces it in every test


@pytest.fixture(autouse=True)
def _no_topics(monkeypatch):
    """No topic is loaded unless a test says so: these tests pass conn=None."""
    async def none(conn):
        return []

    monkeypatch.setattr(ans, "load_topics", none)


def _patch_search(monkeypatch, results):
    calls = []

    async def fake(conn, query, **kw):
        calls.append(kw)
        return results.pop(0)

    monkeypatch.setattr(ans, "search_docs", fake)
    return calls


INDEX = ScopeIndex(vendors=["Veeam"], products=[], aliases={})


async def _run(client, question="How does Veeam work?"):
    return await ans.answer(None, question, settings=_settings(), embedder=None,
                            reranker=None, scope_index=INDEX, client=client)


def test_markers_resolve_to_chunks_in_order_of_appearance():
    hits = [_hit(1), _hit(2), _hit(3)]
    text, cites, unknown = resolve_markers("A [3]. B [1, 3] C [1][2].", hits)
    assert [c["marker"] for c in cites] == [3, 1, 2]
    assert cites[0]["chunk_id"] == "c3" and cites[0]["url"] == "https://docs.example/3"
    assert unknown == 0
    assert "[3]" in text and "[2]" in text


def test_unknown_markers_are_dropped_and_counted():
    hits = [_hit(1)]
    text, cites, unknown = resolve_markers("Fact one [1] and two [7] done [2, 1].", hits)
    assert unknown == 2
    assert "[7]" not in text and "[2" not in text
    assert "  " not in text
    assert [c["marker"] for c in cites] == [1]


@pytest.mark.asyncio
async def test_uncited_non_refusal_is_flagged(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    r = await _run(FakeClient(["Plain answer without markers."]))
    assert r["uncited"] is True and r["citations"] == [] and r["answer"]


@pytest.mark.asyncio
async def test_refusal_is_not_flagged_uncited(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    r = await _run(FakeClient([REFUSAL]))
    assert r["answer"] == REFUSAL and r["uncited"] is False and r["citations"] == []


@pytest.mark.asyncio
async def test_filtered_empty_reruns_unfiltered_and_marks_relaxed(monkeypatch):
    calls = _patch_search(monkeypatch, [SearchResult([], True), SearchResult([_hit(1)], True)])
    r = await _run(FakeClient(["Yes [1]."]))
    assert calls[0]["vendor"] == ["Veeam"]
    assert not calls[1].get("vendor")
    assert r["scope"] == {"vendors": ["Veeam"], "products": [], "relaxed": True}
    assert r["retrieved"] == 1 and r["citations"][0]["marker"] == 1


@pytest.mark.asyncio
async def test_no_hits_refuses_without_calling_the_model(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([], True), SearchResult([], True)])
    client = FakeClient([])
    r = await _run(client)
    assert r["answer"] == REFUSAL and client.calls == [] and r["retrieved"] == 0
    assert r["uncited"] is False and r["usage"] is None


@pytest.mark.asyncio
async def test_empty_reply_is_an_error_not_an_answer(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    client = FakeClient([None, "  "])
    r = await _run(client)
    assert r["answer"] is None and r["error"] == "empty reply" and len(client.calls) == 2


@pytest.mark.asyncio
async def test_exception_retried_then_reported(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    client = FakeClient([RuntimeError("boom"), RuntimeError("boom")])
    r = await _run(client)
    assert r["answer"] is None and r["error"] == "RuntimeError: boom"


@pytest.mark.asyncio
async def test_retry_sums_usage(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    r = await _run(FakeClient([None, "ok [1]"]))
    assert r["usage"] == {"prompt_tokens": 20, "completion_tokens": 10, "cost": 0.002}


@pytest.mark.asyncio
async def test_model_call_passes_effort_and_provider(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    client = FakeClient(["ok [1]"])
    await _run(client)
    kw = client.calls[0]
    assert kw["temperature"] == 0 and kw["max_tokens"] == 4000
    assert kw["model"] == _settings().judge_model
    assert kw["extra_body"] == {
        "reasoning": {"effort": "low"},
        "provider": {"sort": "throughput", "allow_fallbacks": True},
        "usage": {"include": True},
    }
    assert "http" not in "".join(m["content"] for m in kw["messages"])


def test_prompt_contains_no_urls():
    p = build_prompt("What is X?", [_hit(1), _hit(2)])
    assert "http" not in p and "[1]" in p and "[2]" in p and "Title 2" in p
    assert "http" not in ans.SYSTEM_RULES and REFUSAL in ans.SYSTEM_RULES


def test_unknown_marker_removal_keeps_indentation():
    raw = "Steps:\n  - first [9]\n    - nested [1]\n  - done [8]."
    text, _, unknown = resolve_markers(raw, [_hit(1)])
    assert unknown == 2
    assert text == "Steps:\n  - first\n    - nested [1]\n  - done."


@pytest.mark.asyncio
async def test_empty_choices_is_retried_then_an_error(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    client = FakeClient([])

    async def create(**kw):
        client.calls.append(kw)
        return types.SimpleNamespace(choices=[], usage=None)

    client.chat.completions.create = create
    r = await _run(client)
    assert r["answer"] is None and r["error"] == "empty reply" and len(client.calls) == 2


@pytest.mark.asyncio
async def test_marker_only_reply_is_an_error_not_an_answer(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1), _hit(2)], True)])
    r = await _run(FakeClient(["  [9]  "]))
    assert r["answer"] is None and r["error"] == "empty reply" and r["unknown_markers"] == 1


@pytest.mark.asyncio
async def test_answer_is_stripped_after_marker_removal(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    r = await _run(FakeClient(["[9] Text [1]\n"]))
    assert r["answer"] == "Text [1]"


@pytest.mark.asyncio
async def test_message_none_is_retried_then_an_error(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    client = FakeClient([])

    async def create(**kw):
        client.calls.append(kw)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=None)], usage=None)

    client.chat.completions.create = create
    r = await _run(client)
    assert r["answer"] is None and r["error"] == "empty reply" and len(client.calls) == 2


INDEX_VP = ScopeIndex(vendors=["Veeam"], products=["Backup Exchange"], aliases={})


@pytest.mark.asyncio
async def test_scope_detected_with_hits_searches_once_with_both_filters(monkeypatch):
    calls = _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    r = await ans.answer(None, "Does Veeam Backup Exchange work?", settings=_settings(),
                         embedder=None, reranker=None, scope_index=INDEX_VP,
                         client=FakeClient(["Yes [1]."]))
    assert len(calls) == 1
    assert calls[0]["vendor"] == ["Veeam"] and calls[0]["product"] == ["Backup Exchange"]
    assert r["scope"]["relaxed"] is False and r["answer"] == "Yes [1]."


@pytest.mark.asyncio
async def test_both_filters_empty_relaxes_to_unfiltered(monkeypatch):
    calls = _patch_search(monkeypatch, [SearchResult([], True), SearchResult([_hit(1)], True)])
    r = await ans.answer(None, "Does Veeam Backup Exchange work?", settings=_settings(),
                         embedder=None, reranker=None, scope_index=INDEX_VP,
                         client=FakeClient(["Yes [1]."]))
    assert len(calls) == 2
    assert not calls[1].get("vendor") and not calls[1].get("product")
    assert r["scope"]["relaxed"] is True and r["retrieved"] == 1


@pytest.mark.asyncio
async def test_no_scope_empty_does_not_relax(monkeypatch):
    calls = _patch_search(monkeypatch, [SearchResult([], True)])
    client = FakeClient([])
    r = await ans.answer(None, "How do backups work?", settings=_settings(), embedder=None,
                         reranker=None, scope_index=INDEX, client=client)
    assert len(calls) == 1 and not calls[0].get("vendor")
    assert r["scope"] == {"vendors": [], "products": [], "relaxed": False}
    assert r["answer"] == REFUSAL and client.calls == []


def test_citations_carry_the_cited_chunk_text():
    hits = [_hit(1), _hit(2)]
    _text, cites, _unknown = resolve_markers("A [2].", hits)
    assert cites[0]["text"] == hits[1].text


# --- shape-aware answers -------------------------------------------------------------

def _patch_enum(monkeypatch, hits, kept, dropped=(), empty=()):
    from vendorkb.modes import EnumerationResult

    async def fake(conn, query, **kw):
        return EnumerationResult(hits, list(kept), list(dropped), list(empty), True)

    monkeypatch.setattr(ans, "enumerate_vendors", fake)


async def _run_enum(client, index, question="Which vendors support immutable backups?"):
    return await ans.answer(None, question, settings=_settings(), embedder=None,
                            reranker=None, scope_index=index, client=client)


@pytest.mark.asyncio
async def test_enumeration_drops_wrong_vendor_line(monkeypatch):
    idx = ScopeIndex(vendors=["Veeam", "Commvault"], products=[], aliases={})
    _patch_enum(monkeypatch, [_hit(1, "Commvault"), _hit(2, "Veeam")], ["Veeam", "Commvault"])
    reply = "Veeam (VBR): does X [1]\nCommvault (VBR): does Y [1]"
    r = await _run_enum(FakeClient([reply]), idx)
    assert r["shape"] == "enumeration"
    assert r["lines_dropped"] == 1
    assert r["vendors_listed"] == ["Commvault"]
    assert r["vendors_checked"] == ["Veeam", "Commvault"]
    assert "does X" not in r["answer"]


@pytest.mark.asyncio
async def test_enumeration_drops_alias_line(monkeypatch):
    idx = ScopeIndex(vendors=["Microsoft"], products=[], aliases={"Azure Backup": ["Microsoft"]})
    _patch_enum(monkeypatch, [_hit(1, "Microsoft"), _hit(2, "Microsoft")], ["Microsoft"])
    r = await _run_enum(FakeClient(["Azure Backup: does X [2]"]), idx)
    assert r["lines_dropped"] == 1 and r["vendors_listed"] == []
    assert r["answer"].startswith(REFUSAL)


@pytest.mark.asyncio
async def test_enumeration_closing_line_lists_unlisted_vendors(monkeypatch):
    idx = ScopeIndex(vendors=["A", "B", "C", "D"], products=[], aliases={})
    _patch_enum(monkeypatch, [_hit(1, "A")], ["A"], dropped=["C"], empty=["D"])
    r = await _run_enum(FakeClient(["A (VBR): does X [1]"]), idx)
    assert r["vendors_listed"] == ["A"]
    assert r["answer"].endswith("No supporting passage found for: B, C, D.")
    assert r["citations"][0]["chunk_id"] == "c1"


@pytest.mark.asyncio
async def test_enumeration_second_pass_asks_again_for_the_vendors_left_out(monkeypatch):
    idx = ScopeIndex(vendors=["A", "B", "C", "D"], products=[], aliases={})
    hits = [_hit(1, "A"), _hit(2, "B"), _hit(3, "C")]
    _patch_enum(monkeypatch, hits, ["A", "B", "C"], dropped=["D"])
    # the first answer leaves B and C out; the second sees only their passages, numbered 1-2
    client = FakeClient(["A (VBR): does X [1]", "B (VBR): does Y [1]\nC (VBR): nope [9]"])
    r = await _run_enum(client, idx)
    assert len(client.calls) == 2
    second = client.calls[1]["messages"][1]["content"]
    assert "Vendors: B (VBR); C (VBR)\n" in second and "body 1" not in second
    assert r["vendors_listed"] == ["A", "B"] and r["lines_dropped"] == 1
    assert "B (VBR): does Y" in r["answer"]
    assert [c["chunk_id"] for c in r["citations"]] == ["c1", "c2"]  # [1] of B renumbered to [2]
    assert r["answer"].endswith("Passages found but not judged to show it: C.\n"
                                "No supporting passage found for: D.")
    assert r["usage"]["prompt_tokens"] == 20  # both calls counted


@pytest.mark.asyncio
async def test_enumeration_all_lines_dropped_refuses(monkeypatch):
    idx = ScopeIndex(vendors=["A", "B"], products=[], aliases={})
    _patch_enum(monkeypatch, [_hit(1, "B")], ["A", "B"])
    # the second pass (over B, the kept vendor with passages) writes nothing usable either
    r = await _run_enum(FakeClient(["A (VBR): does X [1]\nZ: nope [1]", "Z: nope [1]"]), idx)
    assert r["lines_dropped"] == 3
    assert r["answer"] == (REFUSAL + "\nPassages found but not judged to show it: B."
                           "\nNo supporting passage found for: A.")


@pytest.mark.asyncio
async def test_enumeration_prompt_lists_kept_vendors(monkeypatch):
    idx = ScopeIndex(vendors=["A", "B"], products=[], aliases={})
    _patch_enum(monkeypatch, [_hit(1, "A")], ["A"])
    client = FakeClient(["A (VBR): x [1]"])
    await _run_enum(client, idx)
    msgs = client.calls[0]["messages"]
    assert ans.ENUMERATION_RULES in msgs[0]["content"]
    assert "Vendors: A (VBR)\n" in msgs[1]["content"]


@pytest.mark.asyncio
async def test_comparison_closing_line_for_empty_facet(monkeypatch):
    from vendorkb.modes import ComparisonResult
    idx = ScopeIndex(vendors=["Veeam", "AWS Backup"], products=[], aliases={})

    async def fake(conn, query, **kw):
        return ComparisonResult([_hit(1)], ["AWS Backup"], True)

    monkeypatch.setattr(ans, "compare_facets", fake)
    r = await ans.answer(None, "Compare Veeam vs AWS Backup", settings=_settings(),
                         embedder=None, reranker=None, scope_index=idx,
                         client=FakeClient(["Veeam does X [1]."]))
    assert r["shape"] == "comparison" and r["facets_empty"] == ["AWS Backup"]
    assert r["answer"].endswith("No supporting passage found for: AWS Backup.")


def test_closing_line_and_check_lines_units():
    assert ans.closing_line([]) == ""
    assert ans.closing_line(["b", "a"]) == "No supporting passage found for: a, b."
    assert ans.unconfirmed_line([]) == ""
    assert ans.unconfirmed_line(["b", "a"]) == "Passages found but not judged to show it: a, b."
    kept, listed, dropped = ans.check_enumeration_lines(
        "A: x [1]\n\nB (P): y [9]", [_hit(1, "A")], ["A", "B"])
    assert kept == "A: x [1]" and listed == ["A"] and dropped == 1


@pytest.mark.asyncio
async def test_single_shape_reports_shape(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    r = await _run(FakeClient(["Fact [1]."]))
    assert r["shape"] == "single" and r["answer"] == "Fact [1]."


# --- fix round 1: refusal prefix, parenthesised vendors, markdown, other-vendor mentions ----

_V = ["Veeam", "Commvault", "Hornetsecurity (Altaro)", "NinjaOne (Dropsuite)", "Dell", "Rubrik"]
_H = [_hit(1, "Veeam"), _hit(2, "Commvault"), _hit(3, "Hornetsecurity (Altaro)"),
      _hit(4, "NinjaOne (Dropsuite)"), _hit(5, "Dell"), _hit(6, "Rubrik")]


@pytest.mark.parametrize("line,expected,listed", [
    ("Hornetsecurity (Altaro) (VM Backup): X [3]",
     "Hornetsecurity (Altaro): X [3]", ["Hornetsecurity (Altaro)"]),
    ("Hornetsecurity (Altaro): X [3]", "Hornetsecurity (Altaro): X [3]",
     ["Hornetsecurity (Altaro)"]),
    ("NinjaOne (Dropsuite) (Backup): X [4]", "NinjaOne (Dropsuite): X [4]",
     ["NinjaOne (Dropsuite)"]),
    ("- Veeam (VBR): X [1]", "Veeam (VBR): X [1]", ["Veeam"]),
    ("* Veeam: X [1]", "Veeam: X [1]", ["Veeam"]),
    ("\u2022 Veeam: X [1]", "Veeam: X [1]", ["Veeam"]),
    ("**Veeam** (VBR): X [1]", "Veeam (VBR): X [1]", ["Veeam"]),
    ("**Veeam:** X [1]", "Veeam: X [1]", ["Veeam"]),
    ("__Veeam__: X [1]", "Veeam: X [1]", ["Veeam"]),
    ("1. Veeam: X [1]", "Veeam: X [1]", ["Veeam"]),
    ("2) Veeam (VBR, vbr): X [1]", "Veeam (VBR, vbr): X [1]", ["Veeam"]),
    ("   Veeam (VBR): X [1]", "Veeam (VBR): X [1]", ["Veeam"]),
    ("Veeam: X [2, 1]", "Veeam: X [1]", ["Veeam"]),
])
def test_check_lines_accepts_forms(line, expected, listed):
    kept, got, dropped = ans.check_enumeration_lines(line, _H, _V)
    assert kept == expected and got == listed and dropped == 0


@pytest.mark.parametrize("line", [
    "Dell EMC (PowerProtect): X [5]",
    "Veeam: X [2]",
    "Veeam (VBR): X",
    "Veeam: X [99]",
    "Veeam: X [0]",
    "Veeam: X [1-2]",
    "Veeam \u2014 X [1]",
    "Veeam and Commvault both support it [1][2].",
    "In summary, both work [1][2].",
])
def test_check_lines_drops(line):
    assert ans.check_enumeration_lines(line, _H, _V) == ("", [], 1)


def test_check_lines_strips_foreign_markers_and_skips_blank():
    kept, listed, dropped = ans.check_enumeration_lines(
        "Veeam: X [2][1]\n\nCommvault: Y [2]", _H, _V)
    assert kept == "Veeam: X [1]\nCommvault: Y [2]" and dropped == 0
    assert listed == ["Veeam", "Commvault"]


def test_lines_naming_other_vendors_counted_not_dropped():
    text = ("Veeam (VBR): X [1], unlike Rubrik which lacks it\n"
            "Commvault: Y [2]; veeam does the same\n"
            "Dell: Z [5]\nRubrik: backs up Microsoft 365 [6]")
    kept, listed, dropped = ans.check_enumeration_lines(text, _H, _V)
    assert dropped == 0 and len(listed) == 4
    assert ans.count_lines_naming_other_vendors(kept, _V) == 2


@pytest.mark.asyncio
async def test_enumeration_refusal_prefix_does_not_skip_check(monkeypatch):
    idx = ScopeIndex(vendors=_V, products=[], aliases={})
    _patch_enum(monkeypatch, _H, _V)
    reply = (REFUSAL + "\nHowever, Rubrik (Security Cloud): X [2]\nCommvault: Y")
    r = await _run_enum(FakeClient([reply]), idx)
    assert r["vendors_listed"] == [] and r["citations"] == []
    assert r["answer"].startswith(REFUSAL) and "Rubrik (Security" not in r["answer"]
    assert r["answer"].endswith("Rubrik, Veeam.")


@pytest.mark.asyncio
async def test_enumeration_bare_refusal_still_refuses(monkeypatch):
    idx = ScopeIndex(vendors=["A"], products=[], aliases={})
    _patch_enum(monkeypatch, [_hit(1, "A")], ["A"])
    r = await _run_enum(FakeClient([REFUSAL, REFUSAL]), idx)
    assert r["answer"] == REFUSAL + "\nPassages found but not judged to show it: A."


@pytest.mark.asyncio
async def test_enumeration_parenthesised_vendor_listed_end_to_end(monkeypatch):
    idx = ScopeIndex(vendors=_V, products=[], aliases={})
    _patch_enum(monkeypatch, _H, _V)
    r = await _run_enum(FakeClient(["Hornetsecurity (Altaro) (VM Backup): X [3]", REFUSAL]), idx)
    assert r["vendors_listed"] == ["Hornetsecurity (Altaro)"] and r["lines_dropped"] == 1
    assert "Hornetsecurity (Altaro)" not in r["answer"].rsplit("show it:", 1)[1]
    assert r["lines_naming_other_vendors"] == 0


@pytest.mark.asyncio
async def test_enumeration_reports_lines_naming_other_vendors(monkeypatch):
    idx = ScopeIndex(vendors=_V, products=[], aliases={})
    _patch_enum(monkeypatch, _H, _V)
    r = await _run_enum(FakeClient(["Veeam (VBR): X [1], unlike Rubrik which lacks it"]), idx)
    assert r["lines_naming_other_vendors"] == 1 and r["vendors_listed"] == ["Veeam"]


def test_enumeration_rules_forbid_other_vendors():
    assert "only its own vendor" in ans.ENUMERATION_RULES


def test_product_group_kept_only_when_every_product_is_cited_for_the_vendor():
    hits = [Hit("c1", "d1", "t", ["A"], "Veeam", "Veeam Backup & Replication", "T", "u", 1, 0.5),
            _hit(2, "Commvault")]
    v = ["Veeam", "Commvault"]
    kept, _, dropped = ans.check_enumeration_lines(
        "Veeam (Veeam Backup & Replication): X [1]", hits, v)
    assert kept == "Veeam (Veeam Backup & Replication): X [1]" and dropped == 0
    kept, _, _ = ans.check_enumeration_lines("Veeam (veeam backup & replication): X [1]", hits, v)
    assert kept.startswith("Veeam (veeam backup & replication):")
    kept, _, _ = ans.check_enumeration_lines(
        "Veeam (Veeam Backup & Replication, VBR): X [1]", hits, v)
    assert kept == "Veeam: X [1]"
    # a product that belongs to another vendor's hit is not this vendor's
    kept, _, _ = ans.check_enumeration_lines("Veeam (VBR): X [1]", hits, v)
    assert kept == "Veeam: X [1]"


@pytest.mark.asyncio
async def test_other_vendor_in_product_group_is_dropped_and_counted(monkeypatch):
    idx = ScopeIndex(vendors=_V, products=[], aliases={})
    _patch_enum(monkeypatch, _H, _V)
    r = await _run_enum(FakeClient(["Veeam (Rubrik, Commvault): X [1]"]), idx)
    assert r["answer"].startswith("Veeam: X [1]")
    assert r["lines_naming_other_vendors"] == 1 and r["vendors_listed"] == ["Veeam"]


def test_enumeration_prompt_lists_products_per_vendor():
    def h(n, v, p):
        return Hit(f"c{n}", f"d{n}", "t", ["A"], v, p, "T", "u", n, 0.5)
    hits = [h(1, "Veeam", "Veeam Backup & Replication"), h(2, "Veeam", "Veeam ONE"),
            h(3, "Veeam", "Veeam ONE"), h(4, "Rubrik", "Rubrik Security Cloud"),
            h(5, "Dell", None)]
    prompt = ans.build_enumeration_prompt("Q?", hits, ["Veeam", "Rubrik", "Dell"])
    assert prompt.endswith(
        "Vendors: Veeam (Veeam Backup & Replication, Veeam ONE); "
        "Rubrik (Rubrik Security Cloud); Dell\n")
    assert "only the product names listed for that vendor" in ans.ENUMERATION_RULES


# --- final fix wave ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_shape_with_vendors_wording_keeps_its_scope(monkeypatch):
    calls = _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    r = await _run(FakeClient(["Fact [1]."]),
                   question="Which storage vendors does Veeam integrate with?")
    assert r["shape"] == "single"
    assert calls[0]["vendor"] == ["Veeam"] and r["scope"]["vendors"] == ["Veeam"]


def test_line_naming_the_vendor_of_a_stripped_marker_is_dropped():
    # [6] is Rubrik's: the stripped marker shows the claim was about Rubrik
    line = "Veeam: supports AIX [1]. Rubrik also supports AIX [6]."
    assert ans.check_enumeration_lines(line, _H, _V) == ("", [], 1)
    # a workload named without a foreign marker is untouched
    kept, _, dropped = ans.check_enumeration_lines("Rubrik: backs up Veeam VMs [6]", _H, _V)
    assert kept == "Rubrik: backs up Veeam VMs [6]" and dropped == 0
    # a foreign marker whose vendor the text does not name is only stripped
    kept, _, dropped = ans.check_enumeration_lines("Veeam: X [1], Y [6].", _H, _V)
    assert kept == "Veeam: X [1], Y." and dropped == 0


@pytest.mark.asyncio
async def test_line_about_a_foreign_marker_vendor_counts_as_dropped(monkeypatch):
    idx = ScopeIndex(vendors=_V, products=[], aliases={})
    _patch_enum(monkeypatch, _H, _V)
    reply = "Veeam: supports AIX [1]. Rubrik also supports AIX [6].\nRubrik: AIX agent [6]"
    r = await _run_enum(FakeClient([reply]), idx)
    assert r["lines_dropped"] == 1 and r["vendors_listed"] == ["Rubrik"]
    assert r["answer"].startswith("Rubrik: AIX agent [6]")


def test_stripped_foreign_marker_leaves_no_space_before_punctuation():
    kept, _, _ = ans.check_enumeration_lines("Veeam: supports X [6], and Y [1] [6].", _H, _V)
    assert kept == "Veeam: supports X, and Y [1]."


def test_product_group_may_nest_one_level_of_parentheses():
    hits = [Hit("c1", "d1", "t", ["A"], "Microsoft", "Azure Backup (MABS)", "T", "u", 1, 0.5)]
    kept, listed, dropped = ans.check_enumeration_lines(
        "Microsoft (Azure Backup (MABS)): X [1]", hits, ["Microsoft"])
    assert kept == "Microsoft (Azure Backup (MABS)): X [1]"
    assert listed == ["Microsoft"] and dropped == 0


@pytest.mark.parametrize("line", [
    "Veeam: hardened repository backups cannot be deleted [1]",
    "Veeam: does not require an agent [1]",
    "Veeam: customers don't need to manage infrastructure [1]",
])
def test_capability_lines_with_negative_words_are_kept(line):
    assert ans.check_enumeration_lines(line, _H, _V) == (line, ["Veeam"], 0)


def test_enumeration_rules_ask_for_positive_lines():
    assert "never what it does not" in ans.ENUMERATION_RULES


def test_stripped_workload_vendor_marker_keeps_the_line():
    v = ["Veeam", "Microsoft", "Rubrik"]
    hits = [_hit(1, "Veeam"), _hit(2, "Rubrik"), _hit(3, "Microsoft")]
    kept, listed, dropped = ans.check_enumeration_lines(
        "Veeam: backs up Microsoft 365 [1][3]", hits, v)
    assert kept == "Veeam: backs up Microsoft 365 [1]" and dropped == 0
    assert listed == ["Veeam"]
    assert ans.check_enumeration_lines(
        "Veeam: supports AIX [1]. Rubrik also supports AIX [2].", hits, v) == ("", [], 1)


class FinishClient(FakeClient):
    def __init__(self, replies, finish):
        super().__init__(replies)
        self.finish = finish

    async def _create(self, **kw):
        resp = await super()._create(**kw)
        resp.choices[0].finish_reason = self.finish
        return resp


@pytest.mark.asyncio
async def test_enumeration_uses_a_larger_budget_and_flags_truncation(monkeypatch, caplog):
    idx = ScopeIndex(vendors=["A"], products=[], aliases={})
    _patch_enum(monkeypatch, [_hit(1, "A")], ["A"])
    client = FinishClient(["A: x [1]"], "length")
    r = await _run_enum(client, idx)
    assert client.calls[0]["max_tokens"] == ans.ENUMERATION_MAX_TOKENS == 16000
    assert r["truncated"] is True and "truncated" in caplog.text
    _patch_enum(monkeypatch, [_hit(1, "A")], ["A"])
    r = await _run_enum(FinishClient(["A: x [1]"], "stop"), idx)
    assert r["truncated"] is False


# --- topic answers ---------------------------------------------------------------------------

SHIPPED = Path(__file__).parent.parent.parent / "taxonomy" / "taxonomy.yaml"
TOPIC_Q = "How does Veeam protect backups against ransomware?"
INDEX_VC = ScopeIndex(vendors=["Veeam", "Commvault"], products=[], aliases={})


@pytest.fixture(scope="module")
def topic() -> load.Topic:
    """The shipped taxonomy's ransomware protection topic."""
    return load.validate(load.read_doc(SHIPPED)).topics[0]


def _patch_topics(monkeypatch, topic, hits, rerank_ok=True):
    """Load ``topic`` and answer every ``topic_hits`` with ``hits``; the returned namespace
    holds the ``load_topics`` calls (``loads``) and the ``topic_hits`` kwargs (``hits``)."""
    seen = types.SimpleNamespace(loads=[], hits=[])

    async def fake_load(conn):
        seen.loads.append(conn)
        return [topic]

    async def fake_hits(conn, question, topic, **kw):
        seen.hits.append({"question": question, "topic": topic, **kw})
        return TopicHits(list(hits), {}, rerank_ok, 40)

    monkeypatch.setattr(ans, "load_topics", fake_load)
    monkeypatch.setattr(ans, "topic_hits", fake_hits)
    return seen


def _never(monkeypatch, name):
    async def fail(*args, **kw):
        raise AssertionError(f"{name} must not be called")

    monkeypatch.setattr(ans, name, fail)


async def _run_topic(client, question, index=INDEX):
    return await ans.answer(None, question, settings=_settings(), embedder=None,
                            reranker=None, scope_index=index, client=client)


@pytest.mark.asyncio
async def test_a_topic_question_is_answered_per_aspect(monkeypatch, topic):
    hits = [_hit(1), _hit(2), _hit(3)]
    seen = _patch_topics(monkeypatch, topic, hits)
    _never(monkeypatch, "search_docs")
    client = FakeClient(["Immutable copies [2]. Malware scans [3][1]."])
    r = await _run_topic(client, TOPIC_Q)
    assert len(seen.hits) == 1
    assert seen.hits[0]["question"] == TOPIC_Q and seen.hits[0]["topic"] is topic
    assert seen.hits[0]["facets"] == [("vendor", "Veeam")]
    kw = client.calls[0]
    assert kw["messages"][0]["content"] == (
        ans.SYSTEM_RULES + ans.topic_rules(topic, products_named=True))
    assert kw["messages"][1]["content"] == build_prompt(TOPIC_Q, hits)  # no aspect labels
    assert kw["max_tokens"] == ans.ENUMERATION_MAX_TOKENS
    assert r["shape"] == "single" and r["topic"] == "ransomware_protection"
    assert r["aspects"] == [a.name for a in topic.aspects]
    assert [c["chunk_id"] for c in r["citations"]] == ["c2", "c3", "c1"]
    assert r["scope"] == {"vendors": ["Veeam"], "products": [], "relaxed": False}
    assert r["retrieved"] == 3 and r["rerank_ok"] is True and r["uncited"] is False
    assert "facets_empty" not in r and r["truncated"] is False
    assert r["reranked_docs"] == 40  # TopicHits.reranked_docs


@pytest.mark.asyncio
async def test_a_topic_answer_reports_the_retrieval_facts_of_topic_hits(monkeypatch, topic):
    _patch_topics(monkeypatch, topic, [_hit(1)], rerank_ok=False)
    r = await _run_topic(FakeClient(["Fact [1]."]), TOPIC_Q)
    assert r["rerank_ok"] is False and r["retrieved"] == 1


@pytest.mark.asyncio
async def test_a_topic_question_with_no_passages_refuses_without_the_model(monkeypatch, topic):
    _patch_topics(monkeypatch, topic, [])
    client = FakeClient([])
    r = await _run_topic(client, TOPIC_Q)
    assert r["answer"] == REFUSAL and client.calls == [] and r["topic"] == topic.id


@pytest.mark.asyncio
async def test_a_comparison_topic_question_uses_both_sides(monkeypatch, topic):
    seen = _patch_topics(monkeypatch, topic, [_hit(1, "Veeam"), _hit(2, "Commvault")])
    _never(monkeypatch, "compare_facets")
    client = FakeClient(["Veeam does X [1]. Commvault does Y [2]."])
    r = await _run_topic(client, "Compare Veeam and Commvault ransomware protection.", INDEX_VC)
    assert seen.hits[0]["facets"] == [("vendor", "Veeam"), ("vendor", "Commvault")]
    assert client.calls[0]["messages"][0]["content"] == (
        ans.SYSTEM_RULES + ans.topic_rules(topic, products_named=True))
    assert client.calls[0]["max_tokens"] == ans.ENUMERATION_MAX_TOKENS
    assert r["shape"] == "comparison" and r["topic"] == topic.id
    assert r["aspects"] == [a.name for a in topic.aspects]
    assert r["facets_empty"] == [] and r["scope"]["vendors"] == ["Veeam", "Commvault"]
    assert [c["chunk_id"] for c in r["citations"]] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_a_comparison_topic_question_names_a_side_with_no_passages(monkeypatch, topic):
    _patch_topics(monkeypatch, topic, [_hit(1, "Veeam")])
    r = await _run_topic(FakeClient(["Veeam does X [1]."]),
                         "Compare Veeam and Commvault ransomware protection.", INDEX_VC)
    assert r["facets_empty"] == ["Commvault"]
    assert r["answer"].endswith("No supporting passage found for: Commvault.")


@pytest.mark.asyncio
async def test_a_vendor_and_its_own_product_are_one_side(monkeypatch, topic):
    index = ScopeIndex(vendors=["Veeam"], products=["Backup Exchange"], aliases={},
                       product_vendors={"Backup Exchange": "Veeam"})
    seen = _patch_topics(monkeypatch, topic, [_hit(1)])
    r = await _run_topic(FakeClient(["Fact [1]."]),
                         "How does Veeam Backup Exchange protect against ransomware?", index)
    assert seen.hits[0]["facets"] == [("product", "Backup Exchange")]
    assert r["shape"] == "single" and r["topic"] == topic.id
    assert r["scope"]["vendors"] == [] and r["scope"]["products"] == ["Backup Exchange"]


@pytest.mark.asyncio
async def test_a_vendor_neutral_topic_question(monkeypatch, topic):
    seen = _patch_topics(monkeypatch, topic, [_hit(1), _hit(2, "Commvault")])
    _never(monkeypatch, "search_docs")
    client = FakeClient(["Veeam does X [1]. Commvault does Y [2]."])
    r = await _run_topic(client, "How can I protect my backups against ransomware?")
    assert seen.hits[0]["facets"] == []
    assert client.calls[0]["messages"][0]["content"] == (
        ans.SYSTEM_RULES + ans.topic_rules(topic, products_named=False))
    assert client.calls[0]["max_tokens"] == ans.ENUMERATION_MAX_TOKENS
    assert r["shape"] == "single" and r["topic"] == topic.id
    assert r["scope"] == {"vendors": [], "products": [], "relaxed": False}
    assert r["answer"] == "Veeam does X [1]. Commvault does Y [2]."


@pytest.mark.parametrize("question", [
    "Does Veeam support YARA rules for ransomware?",  # names an aspect
    "How does Veeam work?",  # no trigger
])
@pytest.mark.asyncio
async def test_an_aspect_question_is_not_expanded(monkeypatch, topic, question):
    seen = _patch_topics(monkeypatch, topic, [_hit(9)])
    calls = _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    client = FakeClient(["Yes [1]."])
    r = await _run_topic(client, question)
    assert seen.hits == [] and len(seen.loads) == 1 and len(calls) == 1
    assert r["topic"] is None and "aspects" not in r
    assert client.calls[0]["messages"][0]["content"] == ans.SYSTEM_RULES
    assert client.calls[0]["max_tokens"] == 4000
    assert r["answer"] == "Yes [1]."


# Product names the scope index holds: they never make a question fire (R25).
INDEX_NAMES = ScopeIndex(vendors=["Dell", "Zerto"],
                         products=["PowerProtect Cyber Recovery", "Cyber Resilience Vault"],
                         aliases={}, product_vendors={"PowerProtect Cyber Recovery": "Dell",
                                                      "Cyber Resilience Vault": "Zerto"})


@pytest.mark.parametrize("question", [
    "How do I configure a sync policy in PowerProtect Cyber Recovery?",
    "What are the system requirements for Dell PowerProtect Cyber Recovery 19.18?",
    "How do I set up the Zerto Cyber Resilience Vault?",
])
@pytest.mark.asyncio
async def test_a_product_name_never_triggers_the_topic(monkeypatch, topic, question):
    seen = _patch_topics(monkeypatch, topic, [_hit(9)])
    calls = _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    client = FakeClient(["Configure it so [1]."])
    r = await _run_topic(client, question, INDEX_NAMES)
    assert seen.hits == [] and len(seen.loads) == 1 and len(calls) == 1
    assert r["topic"] is None and "aspects" not in r
    assert client.calls[0]["messages"][0]["content"] == ans.SYSTEM_RULES
    assert client.calls[0]["max_tokens"] == 4000


@pytest.mark.asyncio
async def test_a_topic_question_beside_a_product_name_still_expands(monkeypatch, topic):
    seen = _patch_topics(monkeypatch, topic, [_hit(1)])
    r = await _run_topic(FakeClient(["Fact [1]."]),
                         "How does PowerProtect Cyber Recovery protect against ransomware?",
                         INDEX_NAMES)
    assert r["topic"] == topic.id
    assert seen.hits[0]["facets"] == [("product", "PowerProtect Cyber Recovery")]


def test_topic_rules_copy():
    aspects = (load.Aspect("a", "Alpha", "q", (), ()), load.Aspect("b", "Beta", "q", (), ()))
    t = load.Topic("t", "Ransomware protection", ("ransomware",), aspects)
    head = ("\n- The question asks about ransomware protection in general. Organise the answer by "
            "these aspects, in this order: Alpha; Beta.\n")
    assert ans.topic_rules(t, products_named=True) == head + (
        "- For each product in the question, state what its sources show for each aspect, with "
        "citations, and name the product each statement is about (a vendor may have several); "
        "for an aspect its sources do not cover, write \"the retrieved sources do not cover "
        "this\", never that the product lacks it.")
    assert ans.topic_rules(t, products_named=False) == head + (
        "- The question names no product: for each aspect, describe what the sources show, with "
        "citations, naming the vendor and product each statement comes from; do not present "
        "the vendors named as a complete list.")


@pytest.mark.asyncio
async def test_four_sides_still_take_the_topic_path(monkeypatch, topic):
    vendors = ["Veeam", "Commvault", "Rubrik", "Cohesity"]
    index = ScopeIndex(vendors=vendors, products=[], aliases={})
    seen = _patch_topics(monkeypatch, topic, [_hit(1)])
    r = await _run_topic(FakeClient(["Fact [1]."]),
                         "Compare Veeam, Commvault, Rubrik and Cohesity on ransomware protection.",
                         index)
    assert ans.TOPIC_MAX_SIDES == 4
    assert [n for _, n in seen.hits[0]["facets"]] == vendors and r["topic"] == topic.id


@pytest.mark.asyncio
async def test_a_five_side_comparison_keeps_todays_path(monkeypatch, topic):
    vendors = ["Veeam", "Commvault", "Rubrik", "Cohesity", "Druva"]
    index = ScopeIndex(vendors=vendors, products=[], aliases={})
    seen = _patch_topics(monkeypatch, topic, [_hit(9)])
    facets = []

    async def fake(conn, query, **kw):
        facets.append(kw["facets"])
        return ComparisonResult([_hit(1)], [], True)

    monkeypatch.setattr(ans, "compare_facets", fake)
    client = FakeClient(["Veeam does X [1]."])
    r = await _run_topic(
        client, "Compare Veeam, Commvault, Rubrik, Cohesity and Druva on ransomware protection.",
        index)
    assert seen.hits == [] and seen.loads == [] and len(facets[0]) == 5
    assert r["shape"] == "comparison" and r["topic"] is None and "aspects" not in r
    assert client.calls[0]["messages"][0]["content"] == ans.SYSTEM_RULES
    assert client.calls[0]["max_tokens"] == 4000


WHICH_VENDORS_Q = "Which vendors protect backups against ransomware?"
SUMMARY_SYSTEM = (
    "Summarise this coverage table of backup vendors in at most four sentences, using only the "
    "counts and names given; no citations, no claims beyond the table. A vendor not listed may "
    "still offer a capability: never state or imply that a vendor lacks one, and never write "
    "\"only\".")


def _patch_grid(monkeypatch, grid):
    """Answer every ``topic_grid`` with ``grid``; the returned list holds its kwargs."""
    calls = []

    async def fake(conn, topic, **kw):
        calls.append({"topic": topic, **kw})
        return grid

    monkeypatch.setattr(ans, "topic_grid", fake)
    _never(monkeypatch, "enumerate_vendors")
    _never(monkeypatch, "search_docs")
    return calls


def _grid_of(topic) -> Grid:
    """Veeam has three marks (the last of an earlier version of its page), Commvault one."""
    a = [x.id for x in topic.aspects]
    cells = {("Commvault", a[0]): Cell("supported", _hit(1, "Commvault"), False),
             ("Veeam", a[0]): Cell("partial", _hit(2), False),
             ("Veeam", a[1]): Cell("recent", _hit(3), False),
             ("Veeam", a[2]): Cell("supported", _hit(4), True)}
    return Grid(vendors=["Commvault", "Veeam"], cells=cells, facts_as_of=None,
                no_fact=["Rubrik"], rerank_ok=False, reranked_docs=123)


@pytest.mark.asyncio
async def test_which_vendors_topic_question_gets_the_grid(monkeypatch, topic):
    grid = _grid_of(topic)
    seen = _patch_topics(monkeypatch, topic, [_hit(9)])
    calls = _patch_grid(monkeypatch, grid)
    client = FakeClient(["Veeam covers most aspects."])
    r = await _run_topic(client, WHICH_VENDORS_Q, INDEX_VC)
    # every vendor with documents, and nothing else of the retrieval path
    assert calls == [{"topic": topic, "vendors": ["Veeam", "Commvault"], "embedder": None,
                      "reranker": None}]
    assert seen.hits == [] and len(seen.loads) == 1
    table, hits = topics_mod.render_grid(grid, topic)
    assert r["answer"] == f"Veeam covers most aspects.\n\n{table}"
    # [n] numbers the table row by row: Veeam's three cells, then Commvault's
    assert [h.chunk_id for h in hits] == ["c2", "c3", "c4", "c1"]
    assert r["citations"] == [ans.citation(n, h) for n, h in enumerate(hits, start=1)]
    assert [c["marker"] for c in r["citations"]] == [1, 2, 3, 4]
    assert (r["shape"], r["topic"]) == ("enumeration", topic.id)
    assert r["aspects"] == [a.name for a in topic.aspects]
    assert r["scope"] == {"vendors": [], "products": [], "relaxed": False}
    assert (r["retrieved"], r["rerank_ok"], r["reranked_docs"]) == (4, False, 123)
    assert (r["vendors_checked"], r["vendors_listed"]) == (
        ["Veeam", "Commvault"], ["Commvault", "Veeam"])  # the vendors with a row
    assert (r["lines_dropped"], r["second_pass_listed"]) == (0, 0)
    assert r["uncited"] is False and r["error"] is None and r["truncated"] is False
    assert r["summary_error"] is None and r["catch_up_ok"] is True
    assert r["unknown_markers"] == 0 and r["usage"]["completion_tokens"] == 5
    # the summary: the brief's rules, then a line per aspect with the vendors whose cell is a
    # fact (the page not yet analysed is not a fact), at most 2000 tokens
    (call,) = client.calls
    names = [a.name for a in topic.aspects]
    assert call["messages"][0]["content"] == SUMMARY_SYSTEM == ans.GRID_SUMMARY_RULES
    assert call["messages"][1]["content"] == "\n".join(
        [f"{names[0]}: facts found for 2 vendors: Commvault, Veeam",
         f"{names[1]}: no facts found", f"{names[2]}: facts found for 1 vendors: Veeam",
         *(f"{n}: no facts found" for n in names[3:])])
    assert call["max_tokens"] == 2000


@pytest.mark.asyncio
async def test_a_grid_summary_carries_no_markers_of_its_own(monkeypatch, topic):
    _patch_grid(monkeypatch, _grid_of(topic))
    _patch_topics(monkeypatch, topic, [])
    r = await _run_topic(FakeClient(["Veeam leads [2] and Commvault trails [1, 3]."]),
                         WHICH_VENDORS_Q, INDEX_VC)
    assert r["answer"].startswith("Veeam leads and Commvault trails.\n\n| Vendor |")
    assert r["unknown_markers"] == 3 and len(r["citations"]) == 4  # [2], [1, 3]: three numbers


@pytest.mark.asyncio
async def test_a_failed_summary_leaves_the_grid_alone(monkeypatch, topic):
    grid = _grid_of(topic)
    _patch_topics(monkeypatch, topic, [])
    _patch_grid(monkeypatch, grid)
    client = FakeClient([RuntimeError("boom"), RuntimeError("boom")])  # one retry, then none
    r = await _run_topic(client, WHICH_VENDORS_Q, INDEX_VC)
    table, hits = topics_mod.render_grid(grid, topic)
    assert len(client.calls) == 2 and r["answer"] == table
    assert len(r["citations"]) == len(hits) == 4 and r["truncated"] is False
    # the failure is reported, but `error` stays None: mcp_server turns `error` into a ToolError
    assert r["error"] is None and r["summary_error"] == "RuntimeError: boom"


@pytest.mark.parametrize("replies", [["", "  \n"], ["[1]", "[2, 3]"]])
@pytest.mark.asyncio
async def test_an_empty_summary_is_a_failed_summary(monkeypatch, topic, replies):
    grid = _grid_of(topic)
    _patch_topics(monkeypatch, topic, [])
    _patch_grid(monkeypatch, grid)
    r = await _run_topic(FakeClient(replies), WHICH_VENDORS_Q, INDEX_VC)
    assert r["answer"] == topics_mod.render_grid(grid, topic)[0]
    assert r["error"] is None and r["summary_error"] == "empty reply"


@pytest.mark.parametrize("reply", [
    "Only Veeam documents immutability.", "No vendor documents clean recovery.",
    "None of the vendors lists an air gap.", "Commvault lacks encryption.",
    "Veeam has facts, lacking a clean room.", "Rubrik does not offer a clean room.",
    "These vendors do not offer anomaly detection.", "Rubrik doesn't list it.",
    "Vendors don't list it.", "Anomaly detection is not supported by Commvault.",
    "Encryption is unsupported for Commvault.", "Veeam leads; Commvault lack the rest.",
    "veeam ONLY."])
@pytest.mark.asyncio
async def test_a_summary_that_states_absence_is_dropped(monkeypatch, topic, reply):
    grid = _grid_of(topic)
    _patch_topics(monkeypatch, topic, [])
    _patch_grid(monkeypatch, grid)
    r = await _run_topic(FakeClient([reply]), WHICH_VENDORS_Q, INDEX_VC)
    table, hits = topics_mod.render_grid(grid, topic)
    assert r["answer"] == table and len(r["citations"]) == len(hits)
    assert r["summary_error"] == "absence wording" and r["error"] is None
    assert r["usage"]["completion_tokens"] == 5  # the call was made and is accounted for


@pytest.mark.asyncio
async def test_a_summary_without_absence_wording_is_kept(monkeypatch, topic):
    _patch_topics(monkeypatch, topic, [])
    _patch_grid(monkeypatch, _grid_of(topic))
    reply = "Veeam commonly has facts for three aspects; Commvault for one, a supported one."
    r = await _run_topic(FakeClient([reply]), WHICH_VENDORS_Q, INDEX_VC)
    assert r["answer"].startswith(reply + "\n\n| Vendor |") and r["summary_error"] is None


@pytest.mark.asyncio
async def test_a_grid_without_its_catch_up_says_so(monkeypatch, topic):
    grid = _grid_of(topic)
    grid.catch_up_ok = False
    _patch_topics(monkeypatch, topic, [])
    _patch_grid(monkeypatch, grid)
    r = await _run_topic(FakeClient(["Veeam covers most aspects."]), WHICH_VENDORS_Q, INDEX_VC)
    assert r["catch_up_ok"] is False and r["answer"].startswith("Veeam covers")


@pytest.mark.parametrize("finish, flagged", [("length", True), ("stop", False)])
@pytest.mark.asyncio
async def test_a_truncated_grid_summary_is_flagged(monkeypatch, caplog, topic, finish, flagged):
    _patch_topics(monkeypatch, topic, [])
    _patch_grid(monkeypatch, _grid_of(topic))
    r = await _run_topic(FinishClient(["Veeam covers"], finish), WHICH_VENDORS_Q, INDEX_VC)
    assert r["truncated"] is flagged and ("truncated" in caplog.text) is flagged
    assert r["answer"].startswith("Veeam covers\n\n| Vendor |")


@pytest.mark.asyncio
async def test_a_grid_with_no_marks_answers_without_the_model(monkeypatch, topic):
    grid = Grid(vendors=[], cells={}, facts_as_of=None, no_fact=["Commvault", "Veeam"])
    _patch_topics(monkeypatch, topic, [])
    _patch_grid(monkeypatch, grid)
    client = FakeClient([])
    r = await _run_topic(client, WHICH_VENDORS_Q, INDEX_VC)
    table, hits = topics_mod.render_grid(grid, topic)
    assert client.calls == [] and hits == [] and r["answer"] == table
    assert r["answer"] == (
        "No fact about ransomware protection was found in the capability table for the vendors "
        "checked. This is not evidence that they lack it.\n\nFacts current to: no facts found.")
    assert r["citations"] == [] and r["retrieved"] == 0 and r["uncited"] is False
    assert r["topic"] == topic.id and r["truncated"] is False and r["usage"] is None
    assert r["summary_error"] is None and r["catch_up_ok"] is True


@pytest.mark.parametrize("question", [
    "Which vendors support immutable backups?",  # no trigger
    "Which vendors support YARA rules for ransomware?",  # names an aspect
])
@pytest.mark.asyncio
async def test_other_enumeration_questions_keep_todays_path(monkeypatch, topic, question):
    seen = _patch_topics(monkeypatch, topic, [_hit(9)])
    _never(monkeypatch, "topic_grid")
    _patch_enum(monkeypatch, [_hit(1)], ["Veeam"])
    client = FakeClient(["Veeam (VBR): does X [1]"])
    r = await _run_topic(client, question, INDEX_VC)
    assert r["shape"] == "enumeration" and r["topic"] is None and "aspects" not in r
    assert seen.hits == [] and len(seen.loads) == 1  # the topic query, nothing else
    assert client.calls[0]["messages"][0]["content"] == ans.SYSTEM_RULES + ans.ENUMERATION_RULES
    assert r["vendors_listed"] == ["Veeam"] and r["truncated"] is False


# Workload vendors (shape.WORKLOAD_VENDORS) beside the vendors that protect them.
INDEX_W = ScopeIndex(vendors=["Veeam", "Druva", "Commvault", "Microsoft", "Google",
                              "Salesforce"], products=["Azure Backup"],
                     aliases={"Azure": ["Microsoft"]},
                     product_vendors={"Azure Backup": "Microsoft"})


@pytest.mark.parametrize("question", [
    "Which vendors protect Microsoft 365 against ransomware?",
    "Which tools protect our Google Workspace against ransomware?",
    "Which backup vendors offer ransomware protection for Kubernetes?",
    "Which vendors offer a ransomware recovery warranty?",
    "Which vendors support instant recovery after a ransomware attack?",
])
@pytest.mark.asyncio
async def test_a_qualified_which_vendors_question_keeps_todays_path(monkeypatch, topic,
                                                                   question):
    seen = _patch_topics(monkeypatch, topic, [_hit(9)])
    _never(monkeypatch, "topic_grid")
    _patch_enum(monkeypatch, [_hit(1)], ["Veeam"])
    client = FakeClient(["Veeam (VBR): does X [1]"])
    r = await _run_topic(client, question, INDEX_W)
    assert r["shape"] == "enumeration" and r["topic"] is None and "aspects" not in r
    assert seen.hits == []
    assert client.calls[0]["messages"][0]["content"] == ans.SYSTEM_RULES + ans.ENUMERATION_RULES
    assert r["vendors_listed"] == ["Veeam"]


@pytest.mark.parametrize("question", [
    "Which vendors protect backups against ransomware?",  # tp-09
    "Which backup vendors offer ransomware protection?",  # tp-10
    "Which vendors offer the best ransomware protection?",
])
@pytest.mark.asyncio
async def test_a_broad_which_vendors_question_gets_the_grid(monkeypatch, topic, question):
    _patch_topics(monkeypatch, topic, [])
    calls = _patch_grid(monkeypatch, _grid_of(topic))
    r = await _run_topic(FakeClient(["Veeam covers most aspects."]), question, INDEX_W)
    assert len(calls) == 1 and (r["shape"], r["topic"]) == ("enumeration", topic.id)


@pytest.mark.parametrize("question", [
    "How does Veeam protect Microsoft 365 against ransomware?",
    "How does Druva protect Salesforce against ransomware?",
])
@pytest.mark.asyncio
async def test_a_workload_qualified_topic_question_keeps_todays_path(monkeypatch, topic,
                                                                    question):
    seen = _patch_topics(monkeypatch, topic, [_hit(9)])
    calls = _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    client = FakeClient(["It does [1]."])
    r = await _run_topic(client, question, INDEX_W)
    assert seen.hits == [] and len(calls) == 1
    assert r["topic"] is None and "aspects" not in r and "truncated" not in r
    assert client.calls[0]["messages"][0]["content"] == ans.SYSTEM_RULES
    assert client.calls[0]["max_tokens"] == 4000


@pytest.mark.asyncio
async def test_a_workload_vendor_alone_takes_the_topic_path(monkeypatch, topic):
    seen = _patch_topics(monkeypatch, topic, [_hit(1, "Microsoft")])
    r = await _run_topic(FakeClient(["Soft delete [1]."]),
                         "How does Microsoft protect Azure backups against ransomware?", INDEX_W)
    assert r["topic"] == topic.id and r["shape"] == "single"
    assert seen.hits[0]["facets"] == [("vendor", "Microsoft")]


def test_citation_is_what_resolve_markers_builds():
    hits = [_hit(1), _hit(2)]
    _, built, _ = resolve_markers("A [2].", hits)
    assert ans.citation(2, hits[1]) == built[0] == {
        "marker": 2, "chunk_id": "c2", "doc_id": "d2", "url": "https://docs.example/2",
        "title": "Title 2", "section_path": ["A", "B"], "text": hits[1].text}


@pytest.mark.asyncio
async def test_the_topic_functions_import_lazily_and_delegate_to_topics(monkeypatch, topic):
    # answer.py must not import the topics stack at the top: the CLI imports answer, and
    # `vendorkb sync` must not load the taxonomy stack (tests/unit/test_vendorkb_cli_imports.py)
    calls = []

    async def fake_load(conn):
        calls.append(("load", conn))
        return [topic]

    async def fake_hits(conn, question, topic, **kw):
        calls.append(("hits", conn, question, topic, kw))
        return "hits"

    async def fake_grid(conn, topic, **kw):
        calls.append(("grid", conn, topic, kw))
        return "grid"

    monkeypatch.setattr(ans, "load_topics", _LOAD_TOPICS)
    monkeypatch.setattr(topics_mod, "load_topics", fake_load)
    monkeypatch.setattr(topics_mod, "topic_hits", fake_hits)
    monkeypatch.setattr(topics_mod, "topic_grid", fake_grid)
    monkeypatch.setattr(topics_mod, "render_grid", lambda grid, topic: ("table", [grid, topic]))
    assert await ans.load_topics("conn") == [topic]
    assert ans.detect_topic(TOPIC_Q, [topic]) is topic
    facets = [("vendor", "Veeam")]
    assert await ans.topic_hits("conn", "q", topic, facets=facets, embedder="e",
                                reranker="r") == "hits"
    assert await ans.topic_hits("conn", "q", topic, facets=facets, embedder="e", reranker="r",
                                budget=60, per_aspect=1, own=2) == "hits"
    assert await ans.topic_grid("conn", topic, vendors=["V"], embedder="e",
                                reranker="r") == "grid"
    assert await ans.topic_grid("conn", topic, vendors=["V"], embedder="e", reranker="r",
                                catch_up=False) == "grid"
    assert ans.render_grid("g", topic) == ("table", ["g", topic])
    assert calls == [
        ("load", "conn"),
        # a tunable left out is not passed on: topics' own default applies
        ("hits", "conn", "q", topic, {"facets": facets, "embedder": "e", "reranker": "r"}),
        ("hits", "conn", "q", topic, {"facets": facets, "embedder": "e", "reranker": "r",
                                      "budget": 60, "per_aspect": 1, "own": 2}),
        ("grid", "conn", topic, {"vendors": ["V"], "embedder": "e", "reranker": "r"}),
        ("grid", "conn", topic, {"vendors": ["V"], "embedder": "e", "reranker": "r",
                                 "catch_up": False})]


@pytest.mark.parametrize("finish, flagged", [("length", True), ("stop", False)])
@pytest.mark.asyncio
async def test_a_truncated_topic_answer_is_flagged(monkeypatch, caplog, topic, finish, flagged):
    _patch_topics(monkeypatch, topic, [_hit(1)])
    r = await _run_topic(FinishClient(["Immutable copies [1]."], finish), TOPIC_Q)
    assert r["truncated"] is flagged
    assert ("topic answer truncated" in caplog.text) is flagged
    assert r["answer"] == "Immutable copies [1]."


@pytest.mark.asyncio
async def test_plain_single_and_comparison_answers_carry_no_truncated_flag(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    r = await _run(FinishClient(["Fact [1]."], "length"))
    assert r["shape"] == "single" and "truncated" not in r and "reranked_docs" not in r

    async def fake(conn, query, **kw):
        return ComparisonResult([_hit(1)], [], True)

    monkeypatch.setattr(ans, "compare_facets", fake)
    r = await _run_topic(FinishClient(["Fact [1]."], "length"), "Compare Veeam vs Commvault",
                         INDEX_VC)
    assert r["shape"] == "comparison" and "truncated" not in r


@pytest.mark.asyncio
async def test_a_single_question_with_several_sides_names_a_side_with_no_passages(
        monkeypatch, topic):
    seen = _patch_topics(monkeypatch, topic, [_hit(1, "Veeam")])
    r = await _run_topic(FakeClient(["Veeam does X [1]."]),
                         "How does Veeam protect Commvault backups against ransomware?", INDEX_VC)
    assert r["shape"] == "single" and r["topic"] == topic.id
    assert seen.hits[0]["facets"] == [("vendor", "Veeam"), ("vendor", "Commvault")]
    assert r["facets_empty"] == ["Commvault"]
    assert r["answer"].endswith("No supporting passage found for: Commvault.")
