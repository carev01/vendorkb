import asyncio
import json

from eval.answer_eval import load_done_ids, run_questions, summarize_grades

QS = [{"id": f"q{i}", "type": "single_vendor", "question": f"question {i}",
       "holdout": i == 3} for i in range(1, 5)]


class Tok:
    def __init__(self):
        self.tokens_used = 0


def _answer_factory(cost=1.0, rerank=100, embed=10, embedder=None, reranker=None):
    calls = []

    async def fake(question):
        calls.append(question)
        embedder.tokens_used += embed
        reranker.tokens_used += rerank
        return {"answer": "a [1]", "shape": "single", "rerank_ok": True,
                "citations": [{"marker": 1, "chunk_id": "c1", "doc_id": "d1",
                               "url": "http://u", "title": "t", "text": "chunk text"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": cost},
                "error": None, "scope": {}}
    return fake, calls


async def _vendors(doc_ids):
    return {d: "Acme" for d in doc_ids}


def _run(tmp_path, qs=QS, **kw):
    emb, rr = Tok(), Tok()
    cost = kw.pop("cost", 1.0)
    rerank = kw.pop("rerank", 100)
    fake, calls = _answer_factory(cost=cost, rerank=rerank, embedder=emb, reranker=rr)
    out = tmp_path / "a.jsonl"
    lines = []
    totals = asyncio.run(run_questions(qs, fake, out, embedder=emb, reranker=rr,
                                       vendor_lookup=_vendors, log=lines.append, **kw))
    return out, calls, totals, lines


def _rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def test_line_schema(tmp_path):
    out, _, _, _ = _run(tmp_path, qs=QS[:1])
    r = _rows(out)[0]
    for k in ("id", "type", "shape", "question", "answer", "citations", "vendors_listed",
              "vendors_checked", "facets_empty", "lines_dropped", "lines_naming_other_vendors",
              "rerank_ok", "scope", "usage", "error", "seconds", "rerank_tokens",
              "embed_tokens", "holdout"):
        assert k in r, k
    assert r["id"] == "q1" and r["question"] == "question 1"
    assert r["rerank_tokens"] == 100 and r["embed_tokens"] == 10
    c = r["citations"][0]
    assert c["vendor"] == "Acme" and c["url"] == "http://u" and c["text"] == "chunk text"


def test_exception_is_recorded(tmp_path):
    emb, rr = Tok(), Tok()

    async def boom(q):
        raise RuntimeError("x")
    out = tmp_path / "a.jsonl"
    asyncio.run(run_questions(QS[:1], boom, out, embedder=emb, reranker=rr,
                              vendor_lookup=_vendors, log=lambda s: None))
    assert "RuntimeError" in _rows(out)[0]["error"]


def test_resume_skips_done(tmp_path):
    out, calls, _, _ = _run(tmp_path, qs=QS[:2])
    assert calls == ["question 1", "question 2"]
    with out.open("a") as f:
        f.write('{"id": "q9"')  # torn line
    assert load_done_ids(out) == {"q1", "q2"}
    emb, rr = Tok(), Tok()
    fake, calls2 = _answer_factory(embedder=emb, reranker=rr)
    asyncio.run(run_questions(QS, fake, out, embedder=emb, reranker=rr,
                              vendor_lookup=_vendors, log=lambda s: None))
    assert calls2 == ["question 3", "question 4"]


def test_cost_cap_stops_before_next(tmp_path):
    out, calls, totals, _ = _run(tmp_path, cost=2.0, max_usd=4.0)
    assert len(calls) == 2 and totals["stopped"] == "max_usd"
    assert totals["cost"] == 4.0


def test_rerank_cap_stops(tmp_path):
    _, calls, totals, _ = _run(tmp_path, rerank=100, max_rerank_tokens=200)
    assert len(calls) == 2 and totals["stopped"] == "max_rerank_tokens"


def test_missing_cost_estimated_from_tokens(tmp_path):
    # usage has 5 prompt + 2 completion tokens: 5*3e-6 + 2*12e-6
    out, calls, totals, lines = _run(tmp_path, cost=None, qs=QS[:2])
    assert len(calls) == 2 and totals["unknown_cost"] == 0
    assert abs(totals["cost"] - 2 * 39e-6) < 1e-9
    assert abs(_rows(out)[0]["charge"] - 3.9e-05) < 1e-9


def test_no_cost_no_tokens_charges_flat_and_stops_after_three(tmp_path):
    emb, rr = Tok(), Tok()
    calls = []

    async def fake(q):
        calls.append(q)
        return {"answer": "a", "citations": [], "usage": {"cost": None}, "shape": "single"}
    qs = QS + [{"id": "q5", "type": "single_vendor", "question": "question 5"}]
    lines = []
    totals = asyncio.run(run_questions(qs, fake, tmp_path / "a.jsonl", embedder=emb,
                                       reranker=rr, vendor_lookup=_vendors, log=lines.append))
    assert len(calls) == 3 and totals["stopped"] == "unknown_cost"
    assert abs(totals["cost"] - 0.15) < 1e-9
    assert any("warning" in s.lower() for s in lines)


def test_exception_row_charged_flat_not_a_strike(tmp_path):
    emb, rr = Tok(), Tok()

    async def boom(q):
        raise RuntimeError("x")
    out = tmp_path / "a.jsonl"
    totals = asyncio.run(run_questions(QS, boom, out, embedder=emb, reranker=rr,
                                       vendor_lookup=_vendors, log=lambda s: None))
    assert totals["questions"] == 4 and totals["stopped"] is None
    assert abs(totals["cost"] - 0.20) < 1e-9
    assert _rows(out)[0]["charge"] == 0.05


def test_vendor_lookup_failure_writes_row(tmp_path):
    emb, rr = Tok(), Tok()
    fake, _ = _answer_factory(embedder=emb, reranker=rr)

    async def bad(ids):
        raise ConnectionError("db gone")
    out = tmp_path / "a.jsonl"
    asyncio.run(run_questions(QS[:1], fake, out, embedder=emb, reranker=rr,
                              vendor_lookup=bad, log=lambda s: None))
    r = _rows(out)[0]
    assert "ConnectionError" in r["error"] and r["charge"] == 1.0 and r["answer"] == "a [1]"


def test_caps_seed_from_existing_rows(tmp_path):
    out = tmp_path / "a.jsonl"
    out.write_text(json.dumps({"id": "q1", "usage": {"cost": 6.0}, "rerank_tokens": 150,
                               "error": None}) + "\n"
                   + json.dumps({"id": "q2", "usage": None, "rerank_tokens": 0,
                                 "error": "RuntimeError: x"}) + "\n")
    _, calls, totals, _ = _run(tmp_path, cost=2.0, max_usd=8.0)
    # prior 6.0 + 0.05 (errored row); one new question reaches 8.05
    assert calls == ["question 3"] and totals["stopped"] == "max_usd"
    out.write_text(json.dumps({"id": "q1", "usage": {"cost": 0.1}, "rerank_tokens": 150,
                               "error": None}) + "\n")
    _, calls, totals, _ = _run(tmp_path, max_rerank_tokens=250, rerank=100)
    assert len(calls) == 1 and totals["stopped"] == "max_rerank_tokens"


def test_torn_line_then_append_stays_valid(tmp_path):
    out = tmp_path / "a.jsonl"
    out.write_text('{"id": "q9"')
    _run(tmp_path, qs=QS[:1])
    assert _rows_ok(out) == ["q1"]


def _rows_ok(p):
    ids = []
    for x in p.read_text().splitlines():
        try:
            ids.append(json.loads(x)["id"])
        except json.JSONDecodeError:
            pass
    return ids


def test_running_totals_printed(tmp_path):
    _, _, _, lines = _run(tmp_path, qs=QS[:2])
    assert any("cost" in s and "rerank" in s for s in lines)


def test_ids_and_limit(tmp_path):
    _, calls, _, _ = _run(tmp_path, ids={"q2", "q3", "q4"}, limit=2)
    assert calls == ["question 2", "question 3"]


def _g(id, shape, **kw):
    base = {"id": id, "shape": shape, "facts_total": 4, "facts_covered": 2,
            "unsupported_claims": 1, "refusal_correct": None, "vendor_precision": None,
            "vendor_recall": None, "closing_line_wrong": None, "facets_answered": None}
    base.update(kw)
    return base


def test_summarize(tmp_path):
    ans = tmp_path / "a.jsonl"
    ans.write_text("".join(json.dumps(r) + "\n" for r in
                           [{"id": "e1", "holdout": False}, {"id": "e2", "holdout": True},
                            {"id": "c1", "holdout": True}, {"id": "u1", "holdout": False}]))
    grades = tmp_path / "g.jsonl"
    grades.write_text("".join(json.dumps(r) + "\n" for r in [
        _g("e1", "enumeration", vendor_precision=1.0, vendor_recall=0.5, closing_line_wrong=0),
        _g("e2", "enumeration", facts_covered=4, vendor_precision=0.5, vendor_recall=1.0,
           closing_line_wrong=2),
        _g("c1", "comparison", facets_answered=True),
        _g("u1", "single", facts_total=0, facts_covered=0, unsupported_claims=0,
           refusal_correct=True)]))
    s = summarize_grades(ans, grades)
    en = s["by_shape"]["enumeration"]
    assert en["n"] == 2 and en["vendor_precision"] == 0.75 and en["vendor_recall"] == 0.75
    assert en["fact_coverage"] == 0.75 and en["closing_line_wrong"] == 2
    assert s["by_shape"]["comparison"]["facets_answered"] == 1.0
    assert s["by_shape"]["single"]["refusal_correct"] == 1.0
    assert s["by_holdout"]["holdout"]["n"] == 2
    assert s["by_holdout"]["calibration"]["n"] == 2
    assert s["overall"]["n"] == 4
