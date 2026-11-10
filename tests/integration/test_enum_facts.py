"""Fact passages as candidates of a "which vendors" question (BACKLOG 43), on the fixture
taxonomy with claims inserted already mapped. Every claim and name here is invented."""
import uuid

import pytest
from pgvector import Vector

from tests.integration.test_capabilities_v2 import _claim, _fresh
from vendorkb import modes
from vendorkb.db import connect
from vendorkb.modes import (
    enumerate_vendors, fact_candidates, fixed_query, question_subject, question_taxonomy,
)
from vendorkb.taxonomy import capabilities

VEC = [0.1] * 8


class FakeEmbedder:
    async def embed(self, texts, *, input_type="document"):
        return [VEC for _ in texts]


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores  # of (query, document)
        self.calls = []
        self.queries = []

    async def rerank(self, query, documents, top_k):
        self.calls.append(list(documents))
        self.queries.append(query)
        order = sorted(range(len(documents)), key=lambda i: -self.scores(query, documents[i]))
        return [(i, self.scores(query, documents[i])) for i in order[:top_k]]


@pytest.fixture
async def kb(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        yield conn, await _fresh(conn)
    finally:
        await conn.close()


async def _fact(conn, version, vendor, dimension, target, text, **kw) -> str:
    """A mapped claim whose chunk says ``text``; the chunk id."""
    claim = await _claim(conn, dimension, target, vendor=vendor, mapped_version=version, **kw)
    cur = await conn.execute(
        "UPDATE chunks SET text = %s WHERE chunk_id = (SELECT chunk_id FROM raw_claims"
        " WHERE claim_id = %s) RETURNING chunk_id::text", (text, claim))
    return (await cur.fetchone())[0]


async def _page(conn, vendor, text) -> None:
    """A page of ``vendor`` with one chunk and no claim."""
    doc = uuid.uuid4()
    await conn.execute(
        "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
        " fetched_at) VALUES (%s, %s, 'P', 'S', 'T', 'http://u', %s, now())", (doc, vendor, str(doc)))
    await conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
        " embedding) VALUES (%s, %s, %s, 0, %s, 5, %s)",
        (uuid.uuid4(), doc, ["A"], text, Vector(VEC)))


@pytest.mark.parametrize("question, targets, dimensions", [
    ("Which vendors support Entra ID backup?", ["entra_id"], []),
    ("Which vendors back up Microsoft Entra ID (formerly Azure AD)?", ["entra_id"], []),
    ("Who protects azure ad?", ["entra_id"], []),  # "azure" alone would be another target
    ("Which products back up Amazon RDS databases?", ["aws.rds"], []),
    ("Which vendors document the longest retention?", [], ["retention.maximum_period"]),
    ("Which vendors have max retention for SFDC?", ["salesforce"], ["retention.maximum_period"]),
    ("Which vendors are fastest?", [], []),
    ("", [], []),
])
async def test_question_taxonomy_reads_exact_aliases(kb, question, targets, dimensions):
    conn, _ = kb
    assert await question_taxonomy(conn, question) == (targets, dimensions)


def test_phrases_split_on_brackets_and_offer_the_singular():
    got = modes._phrases("Immutable backups (Hyper-V), please?")
    assert got["immutable backups"] == (0, 2) and got["immutable backup"] == (0, 2)
    assert got["hyper-v"] == (2, 3) and "(hyper-v)," not in got
    assert max(len(p.split()) for p in modes._phrases("a b c d e f g h i")) == modes.MAX_ALIAS_WORDS


async def test_fact_candidates_are_the_vendors_fact_passages_for_what_the_question_names(kb):
    conn, v = kb
    a1 = await _fact(conn, v, "Alpha", "coverage.backup", "entra_id", "alpha entra backup")
    a2 = await _fact(conn, v, "Alpha", "coverage.restore", "entra_id", "alpha entra restore")
    await _fact(conn, v, "Alpha", "coverage.backup", "okta", "alpha okta backup")
    await _fact(conn, v, "Alpha", "coverage.backup", "entra_id", "alpha says no",
                support="not_supported")
    b1 = await _fact(conn, v, "Beta", "coverage.backup", "entra_id", "beta entra backup")
    await _fact(conn, v, "Beta", "coverage.backup", "entra_id", "beta old page", superseded=True)
    await capabilities.build(conn)
    # the target alone: every fact passage on it, a coverage dimension's first
    got = await fact_candidates(conn, "Which vendors support Entra ID?",
                                ["Alpha", "Beta", "Gamma"])
    assert {k: sorted(h.chunk_id for h in hs) for k, hs in got.items()} == {
        "Alpha": sorted([a1, a2]), "Beta": [b1]}
    assert got["Alpha"][0].vendor == "Alpha" and got["Alpha"][0].rerank_score is None
    one = await fact_candidates(conn, "Which vendors support Entra ID?", ["Alpha"], limit=1)
    assert [len(hs) for hs in one.values()] == [1]
    # a generic dimension narrows nothing ("backup scope" is an alias of coverage.backup): saying
    # "backup" or not must not change the passages
    again = await fact_candidates(conn, "What is each vendor's backup scope for Entra ID?",
                                  ["Alpha", "Beta", "Gamma"])
    assert {k: [h.chunk_id for h in hs] for k, hs in again.items()} == {
        k: [h.chunk_id for h in hs] for k, hs in got.items()}
    # nothing named, a generic word only, or switched off: no fact passage
    assert await fact_candidates(conn, "Which vendors are fastest?", ["Alpha"]) == {}
    assert await fact_candidates(conn, "Which vendors support Entra ID?", ["Alpha"], limit=0) == {}
    assert await fact_candidates(conn, "Which vendors support Entra ID?", []) == {}


async def test_a_named_dimension_narrows_a_targets_passages_and_stands_alone_without_one(kb):
    conn, v = kb
    await _fact(conn, v, "Alpha", "coverage.backup", "salesforce", "alpha sfdc backup")
    keep = await _fact(conn, v, "Alpha", "retention.maximum_period", "salesforce",
                       "alpha sfdc retention", support="partial")
    other = await _fact(conn, v, "Alpha", "retention.maximum_period", "entra_id",
                        "alpha entra retention")
    await capabilities.build(conn)
    got = await fact_candidates(conn, "Which vendors have max retention for SFDC?", ["Alpha"])
    assert [h.chunk_id for h in got["Alpha"]] == [keep]  # the dimension's fact on the target
    got = await fact_candidates(conn, "Which vendors document the longest retention?", ["Alpha"])
    assert sorted(h.chunk_id for h in got["Alpha"]) == sorted([keep, other])
    # a generic dimension alone names nothing ("backup scope" -> coverage.backup)
    assert await fact_candidates(conn, "What is the backup scope?", ["Alpha"]) == {}


async def test_a_fact_passage_keeps_a_vendor_whose_own_candidates_miss_it(kb):
    """Beta's page on the subject says "Azure AD": the hybrid search for "Entra ID" finds its
    other pages first, six of them, and the reranker never saw the one that counts."""
    conn, v = kb
    await _fact(conn, v, "Alpha", "coverage.backup", "entra_id", "alpha strong passage")
    for n in range(8):
        await _page(conn, "Beta", f"beta filler {n}")
    await _fact(conn, v, "Beta", "coverage.backup", "entra_id", "beta strong passage on azure ad")
    await _page(conn, "Gamma", "gamma filler")
    await capabilities.build(conn)
    # eight fillers and the fact chunk tie on distance; make the fact chunk the farthest
    far = [0.1] * 7 + [0.9]
    await conn.execute("UPDATE chunks SET embedding = %s WHERE text LIKE 'beta strong%%'",
                       (Vector(far),))
    rr = FakeReranker(lambda q, t: 0.9 if "strong" in t else 0.2)
    kw = dict(embedder=FakeEmbedder(), reranker=rr, vendors=["Alpha", "Beta", "Gamma"],
              per_vendor=2, per_vendor_multi=3, min_score=0.5)
    off = await enumerate_vendors(conn, "Which vendors support Entra ID backup?", fact_chunks=0,
                                  **kw)
    assert off.kept == ["Alpha"] and "Beta" in off.dropped and off.fact_vendors == []
    on = await enumerate_vendors(conn, "Which vendors support Entra ID backup?", **kw)
    assert sorted(on.kept) == ["Alpha", "Beta"] and on.dropped == ["Gamma"]
    assert on.fact_vendors == ["Alpha", "Beta"] and on.kept_by_facts == []
    beta = [h for h in on.hits if h.vendor == "Beta"]
    assert beta[0].text == "beta strong passage on azure ad" and beta[0].rerank_score == 0.9
    # a fact passage is judged like any other: a weak one keeps nobody
    weak = FakeReranker(lambda q, t: 0.2)
    res = await enumerate_vendors(conn, "Which vendors support Entra ID backup?",
                                  **{**kw, "reranker": weak})
    assert res.kept == [] and sorted(res.dropped) == ["Alpha", "Beta", "Gamma"]
    # and with no reranker it is no evidence at all
    res = await enumerate_vendors(conn, "Which vendors support Entra ID backup?",
                                  **{**kw, "reranker": None})
    assert res.fact_vendors == [] and all("strong" not in h.text or h.vendor == "Alpha"
                                          for h in res.hits)


async def test_a_vendor_known_only_by_a_fact_passage_is_judged_too(kb):
    conn, v = kb
    await _fact(conn, v, "Solo", "coverage.backup", "entra_id", "solo strong passage")
    await capabilities.build(conn)
    await conn.execute("UPDATE chunks SET superseded_at = NULL")
    rr = FakeReranker(lambda q, t: 0.9)
    res = await enumerate_vendors(conn, "Which vendors support Entra ID backup?",
                                  embedder=FakeEmbedder(), reranker=rr, vendors=["Solo", "None"],
                                  min_score=0.5)
    assert res.kept == ["Solo"] and res.empty == ["None"]


# --- a plain question and the fixed query -------------------------------------------------------

@pytest.mark.parametrize("question, plain", [
    ("Which vendors support Azure AD backup?", True),
    ("Which vendors can back up and restore Microsoft Entra ID?", True),
    ("Who protects Azure AD?", True),
    ("Which vendors back up Azure AD systems (formerly Entra)?", True),  # nothing added
    ("What is each vendor's backup scope for Entra ID?", False),  # "each", "vendor's", "scope"
    # anything more is a narrower question
    ("Which vendors back up Azure AD without agents?", False),
    ("Which vendors support Azure AD backup natively?", False),
    ("Which vendors back up Azure AD conditional access policies?", False),
    ("Which vendors support Entra ID backup?", False),  # "ID" is outside the fixture's alias,
    ("Which vendors support Azure AD backup? List all vendors, with conditions.", False),
    ("Which vendors have max retention for SFDC?", False),  # a specific dimension
    ("Which vendors document the longest retention?", False),  # no target
    ("Which vendors are fastest?", False),
])
async def test_a_question_is_plain_only_when_it_asks_who_backs_up_a_target(kb, question, plain):
    conn, _ = kb
    subject = await question_subject(conn, question)
    assert subject.plain is plain
    assert (await fixed_query(conn, subject) is not None) is plain


async def test_the_fixed_query_names_the_targets_and_nothing_of_the_wording(kb):
    conn, _ = kb
    a = await fixed_query(conn, await question_subject(conn, "Which vendors support Azure AD?"))
    b = await fixed_query(conn, await question_subject(
        conn, "Who can back up and restore Microsoft Entra ID?"))
    assert a == b == "Backup support for Microsoft Entra ID"


def _by_query(fixed_high: set[str]):
    """A fact passage naming one of ``fixed_high`` scores 0.8 against the fixed query; every
    passage scores 0.3 against the question as asked."""
    def score(query, text):
        if query.startswith("Backup support for"):
            return 0.8 if any(name in text for name in fixed_high) else 0.3
        return 0.3
    return score


async def test_a_plain_question_keeps_a_vendor_by_its_fact_passage_against_the_fixed_query(kb):
    conn, v = kb
    await _fact(conn, v, "Alpha", "coverage.backup", "entra_id", "alpha passage")
    await _fact(conn, v, "Beta", "coverage.backup", "entra_id", "beta passage")
    await _page(conn, "Gamma", "gamma passage")
    await capabilities.build(conn)
    rr = FakeReranker(_by_query({"alpha"}))
    kw = dict(embedder=FakeEmbedder(), vendors=["Alpha", "Beta", "Gamma"], per_vendor=2,
              per_vendor_multi=3, min_score=0.56)
    res = await enumerate_vendors(conn, "Which vendors support Azure AD backup?", reranker=rr,
                                  **kw)
    assert res.fixed_query == "Backup support for Microsoft Entra ID"
    assert rr.queries == ["Which vendors support Azure AD backup?", res.fixed_query]
    # asked as it was, nobody reaches 0.56; Alpha's fact passage reaches 0.65 on the fixed query
    assert res.kept == ["Alpha"] and res.kept_by_facts == ["Alpha"]
    assert sorted(res.dropped) == ["Beta", "Gamma"]
    assert res.best_scores["Alpha"] == 0.3 and res.fact_scores == {"Alpha": 0.8, "Beta": 0.3}
    assert [h.text for h in res.hits] == ["alpha passage"] and res.hits[0].rerank_score == 0.3
    # Gamma has no fact passage: only fact passages are scored against the fixed query
    assert len(rr.calls[1]) == 2 and "gamma passage" not in rr.calls[1]
    # the second floor is its own: 0.64 against the fixed query keeps nobody
    low = FakeReranker(lambda q, t: 0.64 if q.startswith("Backup support") else 0.3)
    res = await enumerate_vendors(conn, "Which vendors support Azure AD backup?", reranker=low,
                                  **kw)
    assert res.kept == [] and res.kept_by_facts == []


async def test_a_vendor_kept_as_asked_shows_its_best_passages_as_asked(kb):
    """The fixed query decides who is kept, not what a vendor that passed anyway shows."""
    conn, v = kb
    await _fact(conn, v, "Alpha", "coverage.backup", "entra_id", "alpha object table")
    await _page(conn, "Alpha", "alpha overview one")
    await _page(conn, "Alpha", "alpha overview two")
    await _fact(conn, v, "Beta", "coverage.backup", "entra_id", "beta object table")
    await _page(conn, "Beta", "beta other page")
    await capabilities.build(conn)

    def score(query, text):
        if query.startswith("Backup support for"):
            return 0.95  # every fact passage, far above anything as asked
        return {"alpha overview one": 0.8, "alpha overview two": 0.7}.get(text, 0.3)

    res = await enumerate_vendors(conn, "Which vendors support Azure AD backup?",
                                  embedder=FakeEmbedder(), reranker=FakeReranker(score),
                                  vendors=["Alpha", "Beta"], per_vendor=2, per_vendor_multi=3,
                                  min_score=0.56)
    assert res.kept == ["Alpha", "Beta"] and res.kept_by_facts == ["Beta"]
    by = {name: [h.text for h in res.hits if h.vendor == name] for name in ("Alpha", "Beta")}
    assert by["Alpha"] == ["alpha overview one", "alpha overview two"]  # not the object table
    assert by["Beta"][0] == "beta object table"  # what kept it comes first


async def test_a_narrower_question_is_judged_as_asked_only(kb):
    """"Without agents" is the point of the question: a passage on backing up the target in
    general must not keep a vendor (BACKLOG 43, user 2026-11-10)."""
    conn, v = kb
    await _fact(conn, v, "Alpha", "coverage.backup", "entra_id", "alpha passage")
    await capabilities.build(conn)
    for question in ("Which vendors back up Azure AD without agents?",
                     "Which vendors support Azure AD backup natively?"):
        rr = FakeReranker(_by_query({"alpha"}))
        res = await enumerate_vendors(conn, question, embedder=FakeEmbedder(), reranker=rr,
                                      vendors=["Alpha"], min_score=0.56)
        assert res.fixed_query is None and rr.queries == [question], question
        assert res.kept == [] and res.dropped == ["Alpha"] and res.fact_scores == {}
        assert res.fact_vendors == ["Alpha"]  # its fact passage was a candidate, as asked


async def test_a_failed_fixed_query_rerank_keeps_what_was_asked(kb):
    conn, v = kb
    await _fact(conn, v, "Alpha", "coverage.backup", "entra_id", "alpha strong passage")
    await _fact(conn, v, "Beta", "coverage.backup", "entra_id", "beta passage")
    await capabilities.build(conn)

    class SecondCallDown(FakeReranker):
        async def rerank(self, query, documents, top_k):
            if query.startswith("Backup support"):
                self.queries.append(query)
                return None
            return await super().rerank(query, documents, top_k)

    rr = SecondCallDown(lambda q, t: 0.9 if "strong" in t else 0.2)
    res = await enumerate_vendors(conn, "Which vendors support Azure AD backup?",
                                  embedder=FakeEmbedder(), reranker=rr,
                                  vendors=["Alpha", "Beta"], min_score=0.56)
    assert res.kept == ["Alpha"] and res.dropped == ["Beta"] and res.rerank_ok is True
    assert res.fixed_query is None and res.kept_by_facts == []
