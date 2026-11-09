import json
from pathlib import Path

import pytest
import yaml

from eval import topic_eval as te

REPO = Path(__file__).resolve().parent.parent.parent
SCRATCH = Path("/tmp/claude-1000/-home-openclaw-Documents-Shared-projects-graph-rag/"
               "964252d0-58b1-4131-93c7-53185320cfde/scratchpad/topic-eval/topic_questions.yaml")

IDS = ["a1", "a2", "a3", "a4"]
ASPECTS = {"topic_x": [{"id": i, "name": f"Aspect {i}", "description": f"about {i}"}
                       for i in IDS]}
CODES = {"s": "covered_supported", "u": "covered_unsupported", "n": "not_covered",
         "d": "stated_not_covered"}


def _q(qid, question, sides, grading):
    return {"id": qid, "type": "cross_vendor", "question": question, "topic": "topic_x",
            "sides": sides, "grading": grading}


QS = [_q("q1", "How does Acme protect?", ["Acme"], "sides"),
      _q("q2", "Compare Acme and Globex.", ["Acme", "Globex"], "sides"),
      _q("q3", "How do I protect?", [], "by_aspect"),
      _q("q4", "Which vendors protect?", [], "enumeration")]


def _row(qid, answer="Acme does a [1].", error=None, question=None):
    question = question or next((q["question"] for q in QS if q["id"] == qid), "?")
    return {"id": qid, "type": "cross_vendor", "holdout": True, "shape": "single",
            "question": question, "answer": answer, "error": error, "seconds": 1.5,
            "usage": {"cost": 0.01}, "scope": {"vendors": ["Acme"]}, "charge": 0.01,
            "citations": [{"marker": 2, "chunk_id": "c2", "doc_id": "d2", "vendor": "Globex",
                           "url": "http://u/2", "title": "T2", "text": "second"},
                          {"marker": 1, "chunk_id": "c1", "doc_id": "d1", "vendor": "Acme",
                           "url": "http://u/1", "title": "T1", "text": "first"}]}


def _runs():
    return {"zebra-before": [_row(q["id"]) for q in QS],
            "quokka-after": [_row(q["id"]) for q in QS]}


def _slots(codes):
    return {i: CODES[c] for i, c in zip(IDS, codes)}


def _sides_grade(item, per_side, claims, unsupported=0):
    return {"item": item, "aspects": {side: _slots(c) for side, c in per_side.items()},
            "claims_total": claims, "unsupported_claims": [f"claim {n}" for n in range(unsupported)]}


def _enum_grade(item, pairs_total, supported, unsupported=0, claims=0):
    pairs = [{"vendor": "Acme", "aspect_id": "a1", "supported": n < supported}
             for n in range(4)]
    return {"item": item, "pairs_total": pairs_total, "pairs": pairs, "claims_total": claims,
            "unsupported_claims": [f"claim {n}" for n in range(unsupported)]}


KEY = {"i1": {"run": "before", "question_id": "q1"},
       "i2": {"run": "before", "question_id": "q2"},
       "i3": {"run": "before", "question_id": "q3"},
       "i4": {"run": "before", "question_id": "q4"},
       "j1": {"run": "after", "question_id": "q1"},
       "j2": {"run": "after", "question_id": "q2"},
       "j3": {"run": "after", "question_id": "q3"},
       "j4": {"run": "after", "question_id": "q4"}}


def _before_grades():
    return [_sides_grade("i1", {"Acme": "snnn"}, 5, 1),
            _sides_grade("i2", {"Acme": "ssnn", "Globex": "sndd"}, 10, 2),
            _sides_grade("i3", {"*": "unnn"}, 4, 1),
            _enum_grade("i4", 40, 3, unsupported=2, claims=20)]


def _after_grades(unsupported=(1, 2, 1)):
    return [_sides_grade("j1", {"Acme": "ssss"}, 5, unsupported[0]),
            _sides_grade("j2", {"Acme": "sssn", "Globex": "sssu"}, 10, unsupported[1]),
            _sides_grade("j3", {"*": "ssnn"}, 4, unsupported[2]),
            _enum_grade("j4", 60, 4, claims=30)]


def _score(grades, corpus=None):
    graded = {g["item"] for g in grades}
    key = {item: who for item, who in KEY.items() if item in graded}
    return te.score_records(te.parse_grades(grades, key, QS, ASPECTS), corpus or {})


# ---- the questions file -------------------------------------------------------------------

def test_questions_file_loads_with_a_grading_kind_each():
    qs = te.load_questions(REPO / "eval" / "topic_questions.yaml")
    assert [q["id"] for q in qs] == [f"tp-{n:02d}" for n in range(1, 11)]
    assert {q["id"]: q["grading"] for q in qs} == {
        **{f"tp-0{n}": "sides" for n in range(1, 8)}, "tp-08": "by_aspect",
        "tp-09": "enumeration", "tp-10": "enumeration"}
    assert {q["id"]: q["sides"] for q in qs}["tp-05"] == ["Veeam Backup & Replication",
                                                           "Azure Backup"]
    assert all(q["topic"] == "ransomware_protection" and q["holdout"] is True for q in qs)


@pytest.mark.skipif(not SCRATCH.exists(), reason="the scratchpad questions file of the "
                    "'before' run exists only on the machine that ran it")
def test_questions_match_the_before_run_character_for_character():
    ours = yaml.safe_load((REPO / "eval" / "topic_questions.yaml").read_text())
    theirs = yaml.safe_load(SCRATCH.read_text())
    assert [(q["id"], q["type"], q["question"]) for q in ours] == [
        (q["id"], q["type"], q["question"]) for q in theirs]
    for ours_q, theirs_q in zip(ours, theirs):
        assert all(ours_q[k] == v for k, v in theirs_q.items())


def test_load_questions_rejects_a_question_with_no_sides_and_no_grading(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump([{"id": "x", "question": "Q?", "topic": "topic_x"}]))
    with pytest.raises(SystemExit, match="x: .*grading"):
        te.load_questions(path)
    path.write_text(yaml.safe_dump([{"id": "x", "question": "Q?", "topic": "topic_x",
                                     "sides": ["A"], "grading": "enumeration"}]))
    with pytest.raises(SystemExit, match="x: .*sides"):
        te.load_questions(path)


def test_aspects_come_from_the_taxonomy_file():
    aspects = te.load_aspects(["ransomware_protection"])["ransomware_protection"]
    assert len(aspects) == 8
    assert all(a["id"] and a["name"] and a["description"] for a in aspects)
    with pytest.raises(SystemExit, match="no_such_topic"):
        te.load_aspects(["no_such_topic"])


# ---- build --------------------------------------------------------------------------------

def test_build_is_blind_and_the_key_maps_back():
    items, key = te.build_items(_runs(), QS, ASPECTS, seed=7)
    assert len(items) == 8
    assert {i["item"] for i in items} == set(key)
    assert {(v["run"], v["question_id"]) for v in key.values()} == {
        (run, q["id"]) for run in ("zebra-before", "quokka-after") for q in QS}
    text = json.dumps(items).lower()
    for secret in ("zebra", "quokka", "before", "after", ".jsonl", "charge", "usage", "scope",
                   "seconds", "holdout", "shape"):
        assert secret not in text
    by_q = {q["id"]: q for q in QS}
    for item in items:
        assert set(item) == {"item", "kind", "question", "sides", "aspects", "answer",
                             "sources"}
        question = by_q[key[item["item"]]["question_id"]]
        assert item["question"] == question["question"]
        assert (item["kind"], item["sides"]) == (question["grading"], question["sides"])
        assert [a["id"] for a in item["aspects"]] == IDS
        assert item["aspects"][0] == {"id": "a1", "name": "Aspect a1", "description": "about a1"}
        assert item["item"] not in ("zebra-before", "quokka-after", question["id"])
        assert [s["marker"] for s in item["sources"]] == [1, 2]
        assert set(item["sources"][0]) == {"marker", "title", "vendor", "url", "text"}


def test_build_shuffles_with_the_seed_and_ids_depend_on_it():
    a, key_a = te.build_items(_runs(), QS, ASPECTS, seed=7)
    b, key_b = te.build_items(_runs(), QS, ASPECTS, seed=7)
    c, _ = te.build_items(_runs(), QS, ASPECTS, seed=8)
    assert (a, key_a) == (b, key_b)
    assert {i["item"] for i in a}.isdisjoint({i["item"] for i in c})
    unshuffled = [(run, q["id"]) for run in ("zebra-before", "quokka-after") for q in QS]
    shuffled = [(key_a[i["item"]]["run"], key_a[i["item"]]["question_id"]) for i in a]
    assert sorted(shuffled) == sorted(unshuffled)


def test_build_refuses_an_answer_it_cannot_grade():
    runs = _runs()
    runs["quokka-after"][1] = _row("q2", answer=None, error="boom")
    with pytest.raises(SystemExit, match="quokka-after: q2: no answer"):
        te.build_items(runs, QS, ASPECTS, seed=1)
    runs = _runs()
    runs["quokka-after"][1] = _row("zz")
    with pytest.raises(SystemExit, match="zz: not in the questions file"):
        te.build_items(runs, QS, ASPECTS, seed=1)
    runs = _runs()
    runs["quokka-after"][1] = _row("q2", question="A reworded question?")
    with pytest.raises(SystemExit, match="q2: .*differs from the questions file"):
        te.build_items(runs, QS, ASPECTS, seed=1)


def test_runs_are_labelled_by_file_stem_and_stems_must_differ(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    for name in ("one/topic-before.jsonl", "two/topic-after.jsonl"):
        (tmp_path / name).write_text(json.dumps(_row("q1")) + "\n\n")
    runs = te.runs_from_files([tmp_path / "one/topic-before.jsonl",
                               tmp_path / "two/topic-after.jsonl"])
    assert list(runs) == ["topic-before", "topic-after"]
    assert [r["id"] for r in runs["topic-after"]] == ["q1"]
    (tmp_path / "two/topic-before.jsonl").write_text("")
    with pytest.raises(SystemExit, match="topic-before"):
        te.runs_from_files([tmp_path / "one/topic-before.jsonl",
                            tmp_path / "two/topic-before.jsonl"])
    (tmp_path / "dup.jsonl").write_text((json.dumps(_row("q1")) + "\n") * 2)
    with pytest.raises(SystemExit, match="q1 twice"):
        te.runs_from_files([tmp_path / "dup.jsonl"])


# ---- score --------------------------------------------------------------------------------

def test_coverage_and_unsupported_rate_per_run():
    before = _score(_before_grades())["before"]
    # 4 + 8 + 4 slots; 1 + (2 + 1) + 0 of them covered with support
    assert before["coverage"] == {"slots": 16, "excluded": 0, "denominator": 16,
                                  "covered_supported": 4, "coverage": 0.25}
    assert before["unsupported"] == {"claims_total": 19, "unsupported_claims": 4,
                                     "rate": 4 / 19}
    assert before["labels"] == {"covered_supported": 4, "covered_unsupported": 1,
                                "not_covered": 9, "stated_not_covered": 2}
    assert before["by_question"]["q2"]["covered_supported"] == 3
    assert before["by_question"]["q2"]["coverage"] == 3 / 8
    after = _score(_after_grades())["after"]
    assert after["coverage"]["covered_supported"] == 4 + 3 + 3 + 2
    assert after["coverage"]["coverage"] == 12 / 16


def test_stated_not_covered_leaves_the_denominator_only_where_the_corpus_lacks_it():
    grades = _before_grades()
    # the only stated_not_covered slots are Globex's a3 and a4 in i2
    assert _score(grades)["before"]["coverage"]["coverage"] == 4 / 16
    documented = {"Globex": {"a3": True, "a4": True}}
    assert _score(grades, documented)["before"]["coverage"]["coverage"] == 4 / 16
    lacks_one = {"Globex": {"a3": False, "a4": True}}
    cov = _score(grades, lacks_one)["before"]["coverage"]
    assert (cov["excluded"], cov["denominator"], cov["coverage"]) == (1, 15, 4 / 15)
    lacks_both = {"Globex": {"a3": False, "a4": False}}
    cov = _score(grades, lacks_both)["before"]["coverage"]
    assert (cov["excluded"], cov["denominator"], cov["coverage"]) == (2, 14, 4 / 14)
    # side names match without regard to case and spacing
    cov = _score(grades, {" globex": {"a3": False, "a4": False}})["before"]["coverage"]
    assert cov["excluded"] == 2


def test_only_a_stated_not_covered_slot_is_excluded():
    grades = _before_grades()
    # Acme's a3 is graded not_covered (n) in i1: the corpus lacking it does not excuse the answer
    cov = _score(grades, {"Acme": {"a3": False, "a2": False}})["before"]["coverage"]
    assert (cov["excluded"], cov["denominator"]) == (0, 16)
    # a vendor-neutral item has no corpus entry, so nothing is excluded for it
    grades = [_sides_grade("i3", {"*": "dddd"}, 0)]
    key = {"i3": KEY["i3"]}
    records = te.parse_grades(grades, key, QS, ASPECTS)
    assert te.score_records(records, {})["before"]["coverage"]["denominator"] == 4
    assert te.score_records(records, {"*": dict.fromkeys(IDS, False)}
                            )["before"]["coverage"]["denominator"] == 0


def test_enumeration_estimate_is_pairs_total_times_the_supported_share():
    enum = _score(_before_grades())["before"]["enumeration"]
    assert (enum["items"], enum["pairs_total"], enum["sampled"], enum["supported"]) == (
        1, 40, 4, 3)
    assert enum["share"] == 0.75
    assert enum["estimate"] == 30.0
    assert (enum["claims_total"], enum["unsupported_claims"]) == (20, 2)
    after = _score(_after_grades())["after"]["enumeration"]
    assert (after["share"], after["estimate"]) == (1.0, 60.0)
    # the enumeration's claims stay out of the unsupported rate
    assert _score(_before_grades())["before"]["unsupported"]["claims_total"] == 19


def test_enumeration_with_no_sampled_pair_has_no_estimate():
    grade = {"item": "i4", "pairs_total": 12, "pairs": [], "claims_total": 0,
             "unsupported_claims": []}
    enum = _score([grade])["before"]["enumeration"]
    assert (enum["share"], enum["estimate"]) == (None, None)
    grade["pairs_total"] = 0
    assert _score([grade])["before"]["enumeration"]["estimate"] == 0.0


def test_agreement_is_the_share_of_identical_labels_between_the_first_two_files():
    first = te.parse_grades(_before_grades() + _after_grades(), KEY, QS, ASPECTS)
    changed = _before_grades() + _after_grades()
    changed[0]["aspects"]["Acme"]["a1"] = "not_covered"
    changed[5]["aspects"]["Globex"]["a4"] = "covered_supported"
    second = te.parse_grades(changed, KEY, QS, ASPECTS)
    result = te.agreement(first, second)
    assert (result["slots"], result["identical"]) == (32, 30)
    assert result["share"] == 30 / 32
    assert result["disagreements"] == [
        {"item": "i1", "side": "Acme", "aspect": "a1",
         "labels": ["covered_supported", "not_covered"]},
        {"item": "j2", "side": "Globex", "aspect": "a4",
         "labels": ["covered_unsupported", "covered_supported"]}]
    assert te.agreement(first, first)["share"] == 1.0


def _metrics(coverage, rate):
    return {"coverage": {"coverage": coverage}, "unsupported": {"rate": rate}}


def test_pass_check_needs_ten_points_of_coverage_and_at_most_two_of_rate():
    ok = te.pass_check(_metrics(0.30, 0.05), _metrics(0.40, 0.07))
    assert ok["pass"] is True
    assert (ok["coverage_rise_points"], ok["unsupported_rise_points"]) == (10.0, 2.0)
    assert ok["coverage_ok"] and ok["unsupported_ok"]
    assert "10" in ok["bar"] and "2" in ok["bar"]
    assert te.pass_check(_metrics(0.30, 0.05), _metrics(0.399, 0.05))["pass"] is False
    assert te.pass_check(_metrics(0.30, 0.05), _metrics(0.60, 0.0701))["pass"] is False
    assert te.pass_check(_metrics(0.30, 0.05), _metrics(0.60, 0.01))["pass"] is True
    unknown = te.pass_check(_metrics(0.30, None), _metrics(0.60, 0.01))
    assert unknown["pass"] is False and unknown["unsupported_ok"] is None


def test_report_has_a_pass_check_only_for_exactly_two_runs_with_labels():
    records = te.parse_grades(_before_grades() + _after_grades(), KEY, QS, ASPECTS)
    one = te.report_for(records, {}, before="before", after="after")
    assert set(one["runs"]) == {"before", "after"}
    assert one["pass_check"]["pass"] is True
    assert one["pass_check"]["coverage_rise_points"] == 50.0
    assert "skipped" in te.report_for(records, {}, before=None, after=None)["pass_check"]
    only_before = te.parse_grades(_before_grades(), {k: v for k, v in KEY.items()
                                                    if v["run"] == "before"}, QS, ASPECTS)
    assert "exactly two" in te.report_for(only_before, {}, before="before",
                                          after="after")["pass_check"]["skipped"]
    with pytest.raises(SystemExit, match="nope"):
        te.report_for(records, {}, before="before", after="nope")


def test_unsupported_rise_too_large_fails_the_check():
    records = te.parse_grades(_before_grades() + _after_grades(unsupported=(3, 5, 3)),
                              KEY, QS, ASPECTS)
    check = te.report_for(records, {}, before="before", after="after")["pass_check"]
    assert check["unsupported_rise_points"] == pytest.approx(
        (11 / 19 - 4 / 19) * 100, abs=0.05)
    assert check["unsupported_ok"] is False
    assert check["pass"] is False


# ---- grade validation -----------------------------------------------------------------------

def _problems(grades, key=KEY):
    with pytest.raises(te.GradeError) as err:
        te.parse_grades(grades, key, QS, ASPECTS)
    return "\n".join(err.value.problems)


def test_a_slot_the_grader_left_out_is_an_error():
    grades = _before_grades()
    del grades[0]["aspects"]["Acme"]["a3"]
    key = {k: v for k, v in KEY.items() if v["run"] == "before"}
    assert "i1: Acme: aspect a3 not graded" in _problems(grades, key)
    grades = _before_grades()
    del grades[1]["aspects"]["Globex"]
    assert "i2: side Globex not graded" in _problems(grades, key)


def test_bad_labels_sides_items_and_counts_are_all_reported():
    key = {k: v for k, v in KEY.items() if v["run"] == "before"}
    grades = _before_grades()
    grades[0]["aspects"]["Acme"]["a1"] = "corpus_lacks"
    grades[1]["aspects"]["Initech"] = _slots("ssss")
    grades[2]["aspects"]["*"]["zz"] = "not_covered"
    grades[2]["unsupported_claims"] = ["x"] * 9
    grades[3]["pairs_total"] = 2
    grades.append({"item": "nope", "aspects": {}, "claims_total": 0, "unsupported_claims": []})
    grades.append(dict(grades[0]))
    grades[1]["aspects"]["acme"] = _slots("ssss")
    del grades[3]["claims_total"]
    text = _problems(grades, key)
    for expected in ("i1: Acme: a1: unknown label 'corpus_lacks'",
                     "i2: unknown side 'Initech'", "i2: side Acme graded twice", "i3: *: unknown aspect 'zz'",
                     "i3: more unsupported claims than claims_total",
                     "i4: pairs_total 2 is below the 4 pairs listed",
                     "i4: claims_total must be a non-negative integer",
                     "nope: not in the key", "i1: graded twice"):
        assert expected in text


def test_an_item_missing_from_a_grades_file_is_an_error():
    assert "j1: not graded" in _problems(_before_grades())


def test_side_names_match_without_regard_to_case_or_spacing():
    grades = _before_grades()
    grades[1]["aspects"] = {" ACME": grades[1]["aspects"]["Acme"],
                            "globex": grades[1]["aspects"]["Globex"]}
    key = {k: v for k, v in KEY.items() if v["run"] == "before"}
    records = te.parse_grades(grades, key, QS, ASPECTS)
    assert {r["item"]: r for r in records}["i2"]["slots"][("Acme", "a1")] == "covered_supported"


# ---- the command line -------------------------------------------------------------------------

def test_build_then_score_end_to_end(tmp_path, capsys):
    qs_path = tmp_path / "q.yaml"
    qs_path.write_text(yaml.safe_dump(
        [{"id": "q1", "type": "single_vendor", "question": "How does Acme protect?",
          "topic": "ransomware_protection", "sides": ["Acme"]},
         {"id": "q2", "type": "cross_vendor", "question": "Which vendors protect?",
          "topic": "ransomware_protection", "sides": [], "grading": "enumeration"}]))
    for name in ("before", "after"):
        (tmp_path / f"topic-{name}.jsonl").write_text(
            "".join(json.dumps(_row(q, question=text)) + "\n" for q, text in (
                ("q1", "How does Acme protect?"), ("q2", "Which vendors protect?"))))
    items_path, key_path = tmp_path / "items.json", tmp_path / "key.json"
    argv = ["build", str(tmp_path / "topic-before.jsonl"), str(tmp_path / "topic-after.jsonl"),
            "--questions", str(qs_path), "--out", str(items_path), "--key", str(key_path),
            "--seed", "3"]
    assert te.main(argv) == 0
    items = json.loads(items_path.read_text())
    key = json.loads(key_path.read_text())
    assert len(items) == 4 and len(items[0]["aspects"]) == 8
    assert "topic-" not in items_path.read_text()
    paths = []
    for label in ("opus", "fable"):
        grades = []
        for item in items:
            run = key[item["item"]]["run"]
            if item["kind"] == "sides":
                code = "covered_supported" if run == "topic-after" else "not_covered"
                grades.append({"item": item["item"], "claims_total": 4,
                               "unsupported_claims": [],
                               "aspects": {"Acme": {a["id"]: code for a in item["aspects"]}}})
            else:
                grades.append({"item": item["item"], "pairs_total": 10, "claims_total": 0,
                               "unsupported_claims": [],
                               "pairs": [{"vendor": "Acme",
                                          "aspect_id": item["aspects"][0]["id"],
                                          "supported": True}]})
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(grades))
        paths.append(str(path))
    capsys.readouterr()
    argv = ["score", str(key_path), *paths, "--questions", str(qs_path),
            "--before", "topic-before", "--after", "topic-after"]
    assert te.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report["grades"]) == set(paths)
    first = report["grades"][paths[0]]
    assert first["runs"]["topic-before"]["coverage"]["coverage"] == 0.0
    assert first["runs"]["topic-after"]["coverage"]["coverage"] == 1.0
    assert first["runs"]["topic-after"]["enumeration"]["estimate"] == 10.0
    assert first["pass_check"]["pass"] is True
    assert report["agreement"]["share"] == 1.0
    # a grades file with a hole exits 2 and says where
    broken = json.loads(Path(paths[0]).read_text())[1:]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(broken))
    assert te.main(["score", str(key_path), str(bad), "--questions", str(qs_path)]) == 2
    assert "not graded" in capsys.readouterr().err


def test_the_grading_brief_names_every_label_kind_and_field():
    brief = (REPO / "eval" / "TOPIC_GRADING_BRIEF.md").read_text()
    for word in (*te.LABELS, *te.KINDS, "claims_total", "unsupported_claims", "pairs_total",
                 "aspect_id", "supported", '"*"'):
        assert word in brief
