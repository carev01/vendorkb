import types

import pytest

from vendorkb import answer as ans
from vendorkb.answer import REFUSAL, build_prompt, resolve_markers
from vendorkb.config import VendorkbSettings
from vendorkb.scope import ScopeIndex
from vendorkb.search import Hit, SearchResult


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
    assert r["answer"].endswith("Checked with nothing documented: B, C, D.")
    assert r["citations"][0]["chunk_id"] == "c1"


@pytest.mark.asyncio
async def test_enumeration_all_lines_dropped_refuses(monkeypatch):
    idx = ScopeIndex(vendors=["A", "B"], products=[], aliases={})
    _patch_enum(monkeypatch, [_hit(1, "B")], ["A", "B"])
    r = await _run_enum(FakeClient(["A (VBR): does X [1]\nZ: nope [1]"]), idx)
    assert r["lines_dropped"] == 2
    assert r["answer"] == REFUSAL + "\nChecked with nothing documented: A, B."


@pytest.mark.asyncio
async def test_enumeration_prompt_lists_kept_vendors(monkeypatch):
    idx = ScopeIndex(vendors=["A", "B"], products=[], aliases={})
    _patch_enum(monkeypatch, [_hit(1, "A")], ["A"])
    client = FakeClient(["A (VBR): x [1]"])
    await _run_enum(client, idx)
    msgs = client.calls[0]["messages"]
    assert ans.ENUMERATION_RULES in msgs[0]["content"]
    assert "Vendors: A" in msgs[1]["content"]


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
    assert r["answer"].endswith("Checked with nothing documented: AWS Backup.")


def test_closing_line_and_check_lines_units():
    assert ans.closing_line([]) == ""
    assert ans.closing_line(["b", "a"]) == "Checked with nothing documented: a, b."
    kept, listed, dropped = ans.check_enumeration_lines(
        "A: x [1]\n\nB (P): y [9]", [_hit(1, "A")], ["A", "B"])
    assert kept == "A: x [1]" and listed == ["A"] and dropped == 1


@pytest.mark.asyncio
async def test_single_shape_reports_shape(monkeypatch):
    _patch_search(monkeypatch, [SearchResult([_hit(1)], True)])
    r = await _run(FakeClient(["Fact [1]."]))
    assert r["shape"] == "single" and r["answer"] == "Fact [1]."
