import json

import pytest
import yaml

from eval import agent_eval
from eval.agent_eval import (check_call, gate, load_questions, parse_reply, score_answer,
                             score_run)

QUESTIONS = [
    {"id": "xv-1", "type": "cross_vendor", "question": "Which vendors  back up\nwidgets?",
     "expected_vendors": [{"vendor": "Acme"}, {"vendor": "Globex"}, {"vendor": "Initech"}],
     "unclear_vendors": ["Initech"]},
    {"id": "xv-2", "type": "cross_vendor", "question": "Which vendors restore gadgets?",
     "expected_vendors": [{"vendor": "Acme"}, {"vendor": "Umbrella"}]},
    {"id": "xv-3", "type": "cross_vendor", "question": "Unlabelled.", "expected_vendors": []},
    {"id": "sv-1", "type": "single_vendor", "question": "Single.",
     "expected_vendors": [{"vendor": "Acme"}]},
]


@pytest.fixture
def qfile(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump(QUESTIONS))
    return path


def test_only_labelled_cross_vendor_questions_are_used(qfile):
    assert [q["id"] for q in load_questions(qfile)] == ["xv-1", "xv-2"]
    assert [q["id"] for q in load_questions(qfile, ["xv-2"])] == ["xv-2"]
    with pytest.raises(SystemExit, match="xv-3"):
        load_questions(qfile, ["xv-3"])


def test_score_answer_excludes_unclear_vendors_and_normalizes_names():
    answer = {"supported": [{"vendor": " acme "}, {"vendor": "Initech"}, {"vendor": "Hooli"}]}
    r = score_answer(QUESTIONS[0], answer)
    # Initech is unclear: neither required nor counted; Hooli is a false vendor
    assert (r["recall"], r["precision"]) == (0.5, 0.5)
    assert (r["listed"], r["false"], r["missed"]) == (3, ["hooli"], ["globex"])
    assert score_answer(QUESTIONS[0], {"supported": [{"vendor": "Initech"}]})["precision"] is None


def _run(recalls, precision=1.0):
    return {"questions": {q: {"recall": r, "precision": precision} for q, r in recalls.items()}}


def test_gate_fails_on_a_mean_drop_and_warns_on_a_question_drop():
    baseline = _run({"a": 1.0, "b": 0.8, "c": 0.9})
    assert gate(_run({"a": 1.0, "b": 0.8, "c": 0.9}), baseline, 0.05, 0.25) == ([], [])
    failures, warnings = gate(_run({"a": 0.5, "b": 0.8, "c": 0.9}), baseline, 0.05, 0.25)
    assert len(failures) == 1 and failures[0].startswith("mean recall")
    assert warnings == ["a: recall 50% (baseline 100%)"]
    # only the questions both runs scored are compared
    assert gate(_run({"a": 1.0, "z": 0.0}), baseline, 0.05, 0.25) == ([], [])
    assert gate(_run({"z": 1.0}), baseline, 0.05, 0.25)[0] == [
        "no question in common with the baseline"]
    failures, _ = gate(_run({"a": 1.0, "b": 0.8, "c": 0.9}, precision=0.8),
                       _run({"a": 1.0, "b": 0.8, "c": 0.9}), 0.05, 0.25)
    assert failures and failures[0].startswith("mean precision")


def test_check_call_enforces_the_arm_and_the_search_budget(tmp_path):
    log = tmp_path / "calls.jsonl"
    assert check_call("query_capabilities", "full", log, 2) is None
    assert "not available" in check_call("query_capabilities", "search", log, 2)
    log.write_text("".join(json.dumps({"tool": t, "arguments": {}}) + "\n"
                           for t in ("search_docs", "list_vendors", "search_docs")))
    assert "budget used up" in check_call("search_docs", "full", log, 2)
    assert check_call("read_section", "full", log, 2) is None


def test_check_call_caps_gather_evidence_retrievals_not_parts(tmp_path):
    log = tmp_path / "calls.jsonl"
    assert check_call("gather_evidence", "full", log, 10, {"question": "q"}) is None
    assert "not available" in check_call("gather_evidence", "search", log, 10)
    calls = [{"question": "q1"}, {"question": "q1", "part": 2}, {"question": "q2", "part": 1}]
    log.write_text("".join(json.dumps({"tool": "gather_evidence", "arguments": a}) + "\n"
                           for a in calls))
    # two retrievals made: a third question is refused, a later part of one is not
    assert "evidence budget used up" in check_call("gather_evidence", "full", log, 10,
                                                   {"question": "q3"})
    assert check_call("gather_evidence", "full", log, 10, {"question": "q1", "part": 3}) is None


def test_prepare_then_score_round_trip(qfile, tmp_path, capsys):
    run = tmp_path / "run"
    assert agent_eval.main(["prepare", "--out", str(run), "--questions", str(qfile),
                            "--max-searches", "4", "--label", "test"]) == 0
    brief = (run / "briefs" / "full-xv-1.md").read_text()
    assert "MAX_SEARCHES=4" in brief and "query_capabilities" in brief
    assert "Which vendors back up widgets?" in brief  # whitespace collapsed
    assert str(run / "results" / "full-xv-1.json") in brief
    assert "expected_vendors" not in brief and "Globex" not in brief  # no labels leak
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest == {"arm": "full", "max_searches": 4, "questions": ["xv-1", "xv-2"],
                        "label": "test"}

    (run / "results" / "full-xv-1.json").write_text(json.dumps(
        {"supported": [{"vendor": "Acme"}, {"vendor": "Globex"}]}))
    with pytest.raises(ValueError, match="full-xv-2"):
        score_run(run, qfile)
    (run / "results" / "full-xv-2.json").write_text(json.dumps(
        {"supported": [{"vendor": "Acme"}]}))
    (run / "logs" / "full-xv-2.jsonl").write_text(
        json.dumps({"tool": "search_docs", "arguments": {"query": "x"}}) + "\n")
    result = score_run(run, qfile)
    assert result["summary"] == {"questions": 2, "recall": 0.75, "precision": 1.0}
    assert result["questions"]["xv-2"]["tools"] == {"search_docs": 1}

    baseline = tmp_path / "baseline.json"
    assert agent_eval.main(["score", str(run), "--questions", str(qfile),
                            "--write-baseline", str(baseline)]) == 0
    assert agent_eval.main(["score", str(run), "--questions", str(qfile),
                            "--baseline", str(baseline)]) == 0
    # a worse run fails the gate; a run with another search budget is not comparable
    (run / "results" / "full-xv-2.json").write_text(json.dumps({"supported": []}))
    assert agent_eval.main(["score", str(run), "--questions", str(qfile),
                            "--baseline", str(baseline)]) == 1
    other = json.loads(baseline.read_text()) | {"max_searches": 10}
    baseline.write_text(json.dumps(other))
    assert agent_eval.main(["score", str(run), "--questions", str(qfile),
                            "--baseline", str(baseline)]) == 2
    assert "FAIL mean recall" in capsys.readouterr().out


def test_parse_reply_reads_sse_and_reports_unusable_replies():
    ok = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "hi"}]}}
    assert parse_reply(200, "event: message\ndata: " + json.dumps(ok) + "\n") == (0, "hi")
    assert parse_reply(200, json.dumps(ok)) == (0, "hi")
    err = {"jsonrpc": "2.0", "id": 1, "result": {"isError": True,
                                                 "content": [{"type": "text", "text": "bad"}]}}
    assert parse_reply(200, json.dumps(err)) == (0, "TOOL ERROR: bad")
    code, text = parse_reply(200, json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"code": 1}}))
    assert code == 1 and text.startswith("error:")
    # a restarting pod: an empty or non-JSON reply says so instead of a traceback
    code, text = parse_reply(502, "")
    assert code == 1 and "HTTP 502" in text and "try the call again" in text
    assert parse_reply(200, "<html>oops</html>")[0] == 1


def test_mcp_call_retries_gateway_errors_and_dropped_connections(monkeypatch):
    import httpx

    replies = []

    class Reply:
        def __init__(self, status, body):
            self.status_code, self.text = status, body

    def post(*a, **kw):
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    ok = json.dumps({"jsonrpc": "2.0", "id": 1,
                     "result": {"content": [{"type": "text", "text": "Facts: 3"}]}})
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(agent_eval.time, "sleep", lambda s: None)
    monkeypatch.setenv("VENDORKB_MCP_TOKEN", "t")
    # a dropped connection, then a 502, then the reply: the agent sees only the reply
    replies[:] = [httpx.ConnectError("reset"), Reply(502, "Bad Gateway"), Reply(200, ok)]
    assert agent_eval.mcp_call("list_vendors", {}) == (0, "Facts: 3")
    assert replies == []
    # three failures: the error, after three attempts
    replies[:] = [httpx.ReadError("x"), httpx.ReadError("x"), httpx.ReadError("x")]
    code, text = agent_eval.mcp_call("list_vendors", {})
    assert code == 1 and "after 3 attempts" in text
    # a client error is not retried
    replies[:] = [Reply(401, json.dumps({"jsonrpc": "2.0", "id": 1, "error": "no"}))]
    assert agent_eval.mcp_call("list_vendors", {})[0] == 1 and replies == []
