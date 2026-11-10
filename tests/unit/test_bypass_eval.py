import json

import pytest

from eval import bypass_eval as be

C1 = "11111111-1111-4111-8111-111111111111"
C2 = "22222222-2222-4222-8222-222222222222"
C3 = "33333333-3333-4333-8333-333333333333"
ANSWER = (f"Alpha (Backup): locks backups [1][2].\nBeta: has MFA [3].\n"
          "No supporting passage found for: Gamma.\n\nSources:\n"
          f"[1] Page one — https://a.example/1 (chunk {C1})\n"
          f"[2] Page two — https://a.example/2 (chunk {C2})\n"
          f"[3] Page three — https://b.example/3 (chunk {C3})")
EVIDENCE = (f"Evidence for: q\nRead as: a question. Part 1 of 1.\n\n## Alpha\n\n"
            f"[1] Alpha · Backup — Page one › Guide\n    source: https://a.example/1 (chunk {C1})\n"
            "Alpha locks backups.\n\n"
            f"[2] Alpha · Backup — Page two\n    source: https://a.example/2 (chunk {C2})\nMore.")


def _log(tool, result, question="Which vendors lock backups?"):
    return [{"tool": tool, "arguments": {"question": question}, "code": 0, "seconds": 2.5,
             "result": result}]


def test_the_pilot_set_loads_with_labels_only_for_which_vendors_questions():
    qs = be.load_set(be.PILOT)
    assert [q["id"] for q in qs] == be.PILOT and len(qs) == 12
    kinds = {q["id"]: q["kind"] for q in qs}
    assert sum(k == "vendors" for k in kinds.values()) == 6
    assert kinds["u-01"] == kinds["tp-01"] == kinds["cv-12"] == "other"
    assert next(q for q in qs if q["id"] == "u-01")["question"] == (
        "Compare ransomware protection capabilities between Veeam and Commvault")
    assert "expected_vendors" in qs[0] and "expected_vendors" not in qs[-1]
    with pytest.raises(SystemExit, match="nope"):
        be.load_set(["nope"])


def test_prepare_writes_a_brief_per_arm_with_its_tools_and_no_labels(tmp_path):
    run = tmp_path / "run"
    assert be.main(["prepare", "--out", str(run), "--ids", "cv-01", "u-01"]) == 0
    names = sorted(p.name for p in (run / "briefs").iterdir())
    assert names == [f"{arm}-{q}.md" for arm in ("evidence", "raw", "relay")
                     for q in ("cv-01", "u-01")]
    relay = (run / "briefs" / "relay-cv-01.md").read_text()
    assert "ARM=relay " in relay and "`answer_question`" in relay
    assert "gather_evidence" not in relay and "search_docs" not in relay
    assert "word for word" in relay and "Relay its answer and its Sources list unchanged" in relay
    evidence = (run / "briefs" / "evidence-u-01.md").read_text()
    assert "ARM=evidence " in evidence and "`gather_evidence`, `read_section`" in evidence
    assert "Compare ransomware protection capabilities between Veeam and Commvault" in evidence
    raw = (run / "briefs" / "raw-cv-01.md").read_text()
    assert "ARM=full " in raw and "MAX_SEARCHES=6" in raw and "query_capabilities" in raw
    assert "answer_question" not in raw
    for text in (relay, evidence, raw):
        assert "expected_vendors" not in text and str(run / "results") in text
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["arms"] == ["relay", "evidence", "raw"]
    assert [q["kind"] for q in manifest["questions"]] == ["vendors", "other"]
    assert "expected_vendors" not in manifest["questions"][0]


def test_split_reply_reads_the_replys_own_sources():
    body, own = be.split_reply("Alpha locks [1].\n\n**Sources:**\n- [1] Page — "
                               "https://a.example/1.\n[2] [Two](https://a.example/2)")
    assert body == "Alpha locks [1].\n" and own == {1: "https://a.example/1",
                                                    2: "https://a.example/2"}
    assert be.split_reply("No list here [1].") == ("No list here [1].", {})


def test_sentences_skip_headings_and_short_lines():
    body = ("## Immutability\n**Veeam**\n- Veeam locks backups for a set time [1]. It also "
            "supports object lock on S3 [2, 3].\n| --- | --- |\nNot found.\n")
    assert be.sentences(body) == ["Veeam locks backups for a set time [1].",
                                  "It also supports object lock on S3 [2, 3]."]
    assert be._numbers(body) == [1, 2, 3]


def test_stride_is_even_and_keeps_the_first():
    assert be.stride(list(range(5)), 8) == [0, 1, 2, 3, 4]
    assert be.stride(list(range(20)), 4) == [0, 5, 10, 15]


def test_tool_sources_reads_both_numbering_formats():
    numbered, by_url = be.tool_sources(_log("answer_question", ANSWER), "relay")
    assert numbered == {1: ("https://a.example/1", C1), 2: ("https://a.example/2", C2),
                        3: ("https://b.example/3", C3)}
    assert by_url["https://a.example/1"] == [C1]
    numbered, by_url = be.tool_sources(_log("gather_evidence", EVIDENCE), "evidence")
    assert numbered == {1: ("https://a.example/1", C1), 2: ("https://a.example/2", C2)}
    # the raw arm's tools number each result from 1: only the URLs count
    numbered, by_url = be.tool_sources(_log("search_docs", EVIDENCE), "raw")
    assert numbered == {} and by_url == {"https://a.example/1": [C1], "https://a.example/2": [C2]}


def test_resolve_prefers_the_replys_sources_and_falls_back_to_the_tools_numbers():
    numbered, by_url = be.tool_sources(_log("answer_question", ANSWER), "relay")
    assert be.resolve(1, {}, numbered, by_url) == ("https://a.example/1", [C1])
    # the reply renumbered: its [1] is the tool's third source
    assert be.resolve(1, {1: "https://b.example/3"}, numbered, by_url) == (
        "https://b.example/3", [C3])
    assert be.resolve(9, {}, numbered, by_url) == (None, [])
    assert be.resolve(1, {1: "https://invented.example/x"}, numbered, by_url) == (
        "https://invented.example/x", [])


def test_reply_facts_of_a_relay_that_dropped_markers():
    q = "Which vendors lock backups?"
    kept = be.reply_facts(ANSWER, _log("answer_question", ANSWER), "relay", q)
    assert kept["markers_given"] == 3 and kept["markers_kept"] == 3
    assert kept["markers"] == 3 and kept["markers_resolved"] == 3 and kept["verbatim"] is True
    assert kept["sentences"] == 3 and kept["sentences_cited"] == 2  # the closing line has none
    assert kept["tool_calls"] == 1 and kept["tool_seconds"] == 2.5 and kept["has_sources_list"]
    bare = be.reply_facts("Alpha locks backups and Beta has MFA.",
                          _log("answer_question", ANSWER, q + " List all vendors."), "relay", q)
    assert bare["markers_kept"] == 0 and bare["markers"] == 0 and bare["sentences_cited"] == 0
    assert bare["verbatim"] is False and bare["asked"] == [q + " List all vendors."]


def test_build_item_gives_each_sampled_statement_its_sources():
    texts = {C1: "Alpha locks backups.", C2: None}
    item = be.build_item(ANSWER, _log("answer_question", ANSWER), "relay", "q", 8, texts.get)
    assert [c["statement"] for c in item["checks"]] == [
        "Alpha (Backup): locks backups [1][2].", "Beta: has MFA [3]."]
    assert item["checks"][0]["sources"] == [
        {"marker": 1, "url": "https://a.example/1", "texts": ["Alpha locks backups."]},
        {"marker": 2, "url": "https://a.example/2", "unresolvable": True}]
    assert item["checks"][1]["sources"][0]["unresolvable"] is True
    assert set(item) == {"question", "reply", "checks"}  # no arm


def test_score_run_and_grades(tmp_path, monkeypatch):
    run = tmp_path / "run"
    assert be.main(["prepare", "--out", str(run), "--ids", "cv-01", "--arms", "relay",
                    "evidence"]) == 0
    expected = [e["vendor"] for e in be.load_set(["cv-01"])[0]["expected_vendors"]]
    for arm, listed in (("relay", expected), ("evidence", expected[:len(expected) // 2])):
        (run / "results" / f"{arm}-cv-01.md").write_text(ANSWER)
        (run / "results" / f"{arm}-cv-01.json").write_text(json.dumps({"supported": listed}))
        (run / "logs" / f"{arm}-cv-01.jsonl").write_text(json.dumps(_log(
            "answer_question" if arm == "relay" else "gather_evidence", ANSWER,
            be.load_set(["cv-01"])[0]["question"])[0]) + "\n")
    monkeypatch.setattr(be, "mcp_call", lambda tool, args: (0, "head\nsource\n\nThe chunk text."))
    assert be.main(["items", str(run)]) == 0
    batch = json.loads((run / "grading" / "batch-01.json").read_text())
    key = json.loads((run / "grading" / "key.json").read_text())
    assert len(batch) == 2 and {k["arm"] for k in key.values()} == {"relay", "evidence"}
    assert all("arm" not in item and "relay" not in json.dumps(item) for item in batch)
    assert batch[0]["checks"][0]["sources"][0]["texts"] == ["The chunk text."]
    grades = [{"item": item["item"], "claims_total": 4, "uncited_claims": ["x"],
               "checks": [{"n": 1, "verdict": "supported", "why": ""},
                          {"n": 2, "verdict": "unsupported", "why": ""}]} for item in batch]
    (run / "grading" / "grades-opus-01.json").write_text(json.dumps(grades))
    assert be.main(["score", str(run)]) == 0
    score = json.loads((run / "score.json").read_text())
    assert score["arms"]["relay"]["summary"]["recall"] == 1.0
    assert score["arms"]["evidence"]["summary"]["recall"] < 1.0
    assert score["arms"]["relay"]["summary"]["markers_kept"] == 1.0
    assert score["arms"]["evidence"]["summary"]["verbatim"] == 1.0
    cell = score["grades"]["graders"]["opus"]["relay"]
    assert cell["checks"] == 2 and cell["unsupported_rate"] == 0.5 and cell["uncited_rate"] == 0.25
    grades[0]["checks"].pop()
    (run / "grading" / "grades-opus-01.json").write_text(json.dumps(grades))
    with pytest.raises(SystemExit, match="needs 2 verdicts"):
        be.main(["score", str(run)])
