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
    r = await _run_enum(FakeClient([REFUSAL]), idx)
    assert r["answer"] == REFUSAL + "\nChecked with nothing documented: A."


@pytest.mark.asyncio
async def test_enumeration_parenthesised_vendor_listed_end_to_end(monkeypatch):
    idx = ScopeIndex(vendors=_V, products=[], aliases={})
    _patch_enum(monkeypatch, _H, _V)
    r = await _run_enum(FakeClient(["Hornetsecurity (Altaro) (VM Backup): X [3]"]), idx)
    assert r["vendors_listed"] == ["Hornetsecurity (Altaro)"] and r["lines_dropped"] == 0
    assert "Hornetsecurity (Altaro)" not in r["answer"].rsplit("documented:", 1)[1]
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
    assert client.calls[0]["max_tokens"] == 8000
    assert r["truncated"] is True and "truncated" in caplog.text
    _patch_enum(monkeypatch, [_hit(1, "A")], ["A"])
    r = await _run_enum(FinishClient(["A: x [1]"], "stop"), idx)
    assert r["truncated"] is False
