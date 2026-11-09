import hashlib
import json
from pathlib import Path

import pytest
import yaml

from eval import topic_eval as te
from vendorkb.answer import REFUSAL, closing_line, unconfirmed_line
from vendorkb.search import Hit
from vendorkb.taxonomy import load
from vendorkb.topics import BLANK_NOTE, LEGEND, Cell, Grid, render_grid

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


def _enum_grade(item, pairs_total, supported):
    """The even-stride sample of ``pairs_total`` pairs (ceil(total / 30) apart), of which the
    first ``supported`` are supported."""
    stride = max(1, -(-pairs_total // 30))
    listed = -(-pairs_total // stride)
    pairs = [{"vendor": f"V{n}", "aspect_id": IDS[n % 4], "supported": n < supported}
             for n in range(listed)]
    return {"item": item, "pairs_total": pairs_total, "pairs": pairs}


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
            _enum_grade("i4", 40, 16)]


def _after_grades(unsupported=(1, 2, 1), enum_supported=30):
    return [_sides_grade("j1", {"Acme": "ssss"}, 5, unsupported[0]),
            _sides_grade("j2", {"Acme": "sssn", "Globex": "sssu"}, 10, unsupported[1]),
            _sides_grade("j3", {"*": "ssnn"}, 4, unsupported[2]),
            _enum_grade("j4", 60, enum_supported)]


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


def test_the_shuffle_mixes_two_runs_of_ten_questions():
    questions = [_q(f"q{n}", f"Question {n}?", ["Acme"], "sides") for n in range(10)]
    runs = {run: [_row(q["id"], question=q["question"]) for q in questions]
            for run in ("zebra-before", "quokka-after")}
    items, key = te.build_items(runs, questions, ASPECTS, seed=1)
    order = [(key[i["item"]]["run"], key[i["item"]]["question_id"]) for i in items]
    unshuffled = [(run, q["id"]) for run in runs for q in questions]
    assert len(order) == 20 and sorted(order) == sorted(unshuffled)
    assert order != unshuffled
    # the runs interleave: both fill the first half, and the run changes often down the list
    assert {run for run, _ in order[:10]} == set(runs)
    assert sum(x[0] != y[0] for x, y in zip(order, order[1:])) >= 8
    # and the questions themselves leave their file order
    assert [q for _, q in order] != [q for _, q in unshuffled]


def test_item_ids_are_opaque_not_a_hash_of_run_and_question():
    seeds = (7, 8)
    for seed in seeds:
        items, key = te.build_items(_runs(), QS, ASPECTS, seed=seed)
        ids = {i["item"] for i in items}
        assert len(ids) == 8 and all(len(i) == 8 and int(i, 16) >= 0 for i in ids)
        for run in ("zebra-before", "quokka-after"):
            for q in QS:
                for salt in (f"{s}|{run}|{q['id']}" for s in (*seeds, "", 0)):
                    assert hashlib.sha256(salt.encode()).hexdigest()[:8] not in ids
                for text in (run, q["id"], f"{run}|{q['id']}"):
                    assert hashlib.sha256(text.encode()).hexdigest()[:8] not in ids


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


GRID_TOPIC = load.Topic("topic_x", "Topic X", ("x",), tuple(
    load.Aspect(i, f"Aspect {i}", "q", (), ("d.d",)) for i in IDS))


def _grid_row(qid="q4", vendors=15, legend=True):
    """An answer row whose answer is a vendorkb grid (a summary, then the table) of ``vendors``
    vendors by the four aspects, about two cells in three filled; each cited text names its
    vendor and aspect. ``legend`` False: only the row's keys tell it is a grid."""
    cells = {}
    for n in range(vendors):
        for k, aspect in enumerate(IDS):
            if (n + k) % 3:
                vendor = f"Vendor {n:02d}"
                cells[(vendor, aspect)] = Cell("supported", Hit(
                    chunk_id=f"c{n}{aspect}", doc_id=f"d{n}{aspect}",
                    text=f"{vendor} does {aspect}", section_path=[], vendor=vendor, product="P",
                    title=f"T {n}", url=f"http://u/{n}/{aspect}", rrf_rank=0,
                    rerank_score=None), False)
    grid = Grid(vendors=sorted({v for v, _ in cells}), cells=cells, facts_as_of=None,
                no_fact=["Zeta"])
    table, hits = render_grid(grid, GRID_TOPIC)
    if not legend:
        table = table.replace(LEGEND, "")
    row = _row(qid, answer=f"Vendors have facts.\n\n{table}")
    row.update(shape="enumeration", topic="topic_x")
    row["citations"] = [{"marker": n, "chunk_id": h.chunk_id, "doc_id": h.doc_id,
                         "vendor": h.vendor, "url": h.url, "title": h.title, "text": h.text}
                        for n, h in enumerate(hits, 1)]
    return row, hits


def _build_grid(row, seed=5):
    items, key = te.build_items({"after": [row]}, QS, ASPECTS, seed=seed)
    return items[0], key[items[0]["item"]]


def test_a_grid_answer_is_graded_on_a_random_sample_of_its_cells():
    row, hits = _grid_row()
    assert len(hits) == 40
    item, key = _build_grid(row)
    assert item["kind"] == "enumeration" and item["pairs_total"] == 40
    cells = item["cells"]
    assert len(cells) == 30 == len({(c["vendor"], c["aspect_id"]) for c in cells})
    # each cell: its row's vendor and its column's aspect, as the cited text says; row-major
    for c in cells:
        assert hits[c["marker"] - 1].text == f"{c['vendor']} does {c['aspect_id']}"
    assert [c["marker"] for c in cells] == sorted(c["marker"] for c in cells)
    # only the sampled cells' sources, each with its text
    assert [s["marker"] for s in item["sources"]] == [c["marker"] for c in cells]
    assert all(s["text"] for s in item["sources"])
    # not the first 30 cells: a seeded random sample, the key holds it for the scorer
    assert [c["marker"] for c in cells] != list(range(1, 31))
    assert key["cells"] == [[c["vendor"], c["aspect_id"]] for c in cells]
    assert key["pairs_total"] == 40 and key["question_id"] == "q4"
    again, _ = _build_grid(row)
    other, _ = _build_grid(row, seed=6)
    assert again["cells"] == cells and other["cells"] != cells


def test_a_small_grid_gives_every_cell_and_a_keyed_grid_needs_no_legend():
    row, hits = _grid_row(vendors=6)
    assert len(hits) <= 30
    item, _ = _build_grid(row)
    assert [c["marker"] for c in item["cells"]] == list(range(1, len(hits) + 1))
    row, hits = _grid_row(legend=False)
    item, _ = _build_grid(row)
    assert item["pairs_total"] == 40 and len(item["cells"]) == 30
    # vendorkb's answer when no cell is filled: a grid of nothing to grade
    empty = _row("q4", answer=render_grid(Grid([], {}, None, ["Acme"]), GRID_TOPIC)[0])
    empty.update(shape="enumeration", topic="topic_x", citations=[])
    item, key = _build_grid(empty)
    assert (item["pairs_total"], item["cells"], item["sources"]) == (0, [], [])
    assert key["cells"] == []


def test_a_prose_enumeration_answer_is_not_a_grid():
    items, key = te.build_items({"after": [_row("q4")]}, QS, ASPECTS, seed=1)
    assert "cells" not in items[0] and "pairs_total" not in items[0]
    assert set(key[items[0]["item"]]) == {"run", "question_id"}


def _grid_key(row):
    item, key = _build_grid(row)
    return item, {item["item"]: key}


def test_a_grid_item_is_graded_on_exactly_its_cells():
    item, key = _grid_key(_grid_row()[0])
    pairs = [{"vendor": c["vendor"], "aspect_id": c["aspect_id"], "supported": n % 2 == 0}
             for n, c in enumerate(item["cells"])]
    grade = {"item": item["item"], "pairs_total": 40, "pairs": pairs}
    [record] = te.parse_grades([grade], key, QS, ASPECTS)
    enum = te.score_records([record], {})["after"]["enumeration"]
    assert (enum["pairs_total"], enum["sampled"], enum["supported"]) == (40, 30, 15)
    assert enum["estimate"] == pytest.approx(20.0)
    first = item["cells"][0]
    missing = {**grade, "pairs": pairs[1:]}
    text = "\n".join(_problems_for([missing], key))
    assert f"cell {first['vendor']} / {first['aspect_id']} not graded" in text
    extra = {**grade, "pairs": [*pairs[1:], {"vendor": "Vendor 99", "aspect_id": "a1",
                                              "supported": True}]}
    assert "Vendor 99 / a1 is not one of the item's cells" in "\n".join(
        _problems_for([extra], key))
    wrong_total = {**grade, "pairs_total": 30}
    assert "pairs_total must be 40, the grid's cell count" in "\n".join(
        _problems_for([wrong_total], key))


def _problems_for(grades, key):
    with pytest.raises(te.GradeError) as err:
        te.parse_grades(grades, key, QS, ASPECTS)
    return err.value.problems


def test_build_refuses_a_run_expected_to_take_the_topic_path(tmp_path, capsys):
    grid = {**_row("q4"), "shape": "enumeration", "topic": "ransomware_protection"}
    rows = {"before": [_row("q1"), _row("q4")],
            "after": [{**_row("q1"), "topic": None}, {**grid, "truncated": True}]}
    paths = []
    for label, data in rows.items():
        path = tmp_path / f"{label}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in data))
        paths.append(str(path))
    qs_path = tmp_path / "q.yaml"
    qs_path.write_text(yaml.safe_dump([{**q, "topic": "ransomware_protection"} for q in QS]))
    base = ["build", *paths, "--questions", str(qs_path), "--out", str(tmp_path / "i.json"),
            "--key", str(tmp_path / "k.json")]
    with pytest.raises(SystemExit) as err:
        te.main([*base, "--expect-topic", "after"])
    assert str(err.value) == ("--expect-topic: these rows have no topic or a truncated answer:"
                              "\nafter: q1: topic is null\nafter: q4: truncated")
    with pytest.raises(SystemExit, match="no such run: nope"):
        te.main([*base, "--expect-topic", "nope"])
    # the "before" run (no topic key: the code before topics) is not checked unless named
    capsys.readouterr()
    rows["after"] = [{**_row("q1"), "topic": "ransomware_protection"}, grid]
    Path(paths[1]).write_text("".join(json.dumps(r) + "\n" for r in rows["after"]))
    assert te.main([*base, "--expect-topic", "after"]) == 0


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
    # pooled over every item: 4 of 19 claims, and 40 pairs of which 24 of 30 sampled hold
    assert before["unsupported"] == {
        "claims_total": 19, "unsupported_claims": 4, "pairs_total": 40,
        "unsupported_pairs_estimate": pytest.approx(8.0), "rate": pytest.approx(12 / 59),
        "prose_rate": pytest.approx(4 / 19), "pair_rate": pytest.approx(8 / 40)}
    assert before["labels"] == {"covered_supported": 4, "covered_unsupported": 1,
                                "not_covered": 9, "stated_not_covered": 2}
    assert before["by_question"]["q2"]["covered_supported"] == 3
    assert before["by_question"]["q2"]["coverage"] == 3 / 8
    after = _score(_after_grades())["after"]
    assert after["coverage"]["covered_supported"] == 4 + 3 + 3 + 2
    assert after["coverage"]["coverage"] == 12 / 16
    assert after["unsupported"]["rate"] == 4 / 79


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
        1, 40, 20, 16)
    assert enum["share"] == 0.8
    assert enum["estimate"] == pytest.approx(32.0)
    assert enum["unsupported_estimate"] == pytest.approx(8.0)
    assert "claims_total" not in enum
    after = _score(_after_grades())["after"]["enumeration"]
    assert (after["share"], after["estimate"]) == (1.0, 60.0)
    assert after["unsupported_estimate"] == 0.0


def test_a_pair_total_under_the_cap_lists_every_pair():
    grade = _enum_grade("i4", 7, 5)
    assert len(grade["pairs"]) == 7
    enum = _score([grade])["before"]["enumeration"]
    assert (enum["sampled"], enum["share"], enum["estimate"]) == (7, 5 / 7, pytest.approx(5.0))


def test_enumeration_with_no_sampled_pair_has_no_estimate():
    records = te.parse_grades([_enum_grade("i4", 12, 5)], {"i4": KEY["i4"]}, QS, ASPECTS)
    records[0]["pairs"] = []  # not reachable through the validator, which wants a full sample
    run = te.score_records(records, {})["before"]
    assert run["enumeration"]["share"] is None
    assert run["enumeration"]["estimate"] is None
    assert run["unsupported"]["rate"] is None
    empty = _score([_enum_grade("i4", 0, 0)])["before"]
    assert empty["enumeration"]["estimate"] == 0.0
    assert empty["unsupported"]["rate"] is None  # no claims and no pairs at all


def test_a_run_without_enumeration_items_has_a_null_estimate():
    run = _score(_before_grades()[:3])["before"]
    assert run["enumeration"]["estimate"] is None
    assert run["enumeration"]["items"] == 0
    # the pooled rate is then just claims over claims
    assert run["unsupported"]["rate"] == 4 / 19


def test_the_pooled_rate_weights_pairs_like_claims():
    grades = [_sides_grade("i1", {"Acme": "snnn"}, 10, 1), _enum_grade("i4", 30, 15)]
    rate = _score(grades)["before"]["unsupported"]
    # 1 unsupported of 10 claims, and half of 30 pairs: (1 + 15) / (10 + 30)
    assert rate["rate"] == pytest.approx(16 / 40)


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


def test_agreement_shows_counts_and_pairs_side_by_side():
    first = te.parse_grades(_before_grades(), {k: v for k, v in KEY.items()
                                               if v["run"] == "before"}, QS, ASPECTS)
    changed = _before_grades()
    changed[0]["claims_total"] = 7
    changed[0]["unsupported_claims"] = ["a", "b"]
    changed[3]["pairs"][0]["supported"] = False
    changed[3]["pairs"][1]["vendor"] = "W1"
    second = te.parse_grades(changed, {k: v for k, v in KEY.items()
                                       if v["run"] == "before"}, QS, ASPECTS)
    items = {i["item"]: i for i in te.agreement(first, second)["items"]}
    assert list(items) == ["i1", "i2", "i3", "i4"]
    assert items["i1"] == {"item": "i1", "kind": "sides", "claims_total": [5, 7],
                           "unsupported_claims": [1, 2]}
    assert items["i3"]["claims_total"] == [4, 4] and "pairs" not in items["i3"]
    enum = items["i4"]
    assert enum["kind"] == "enumeration" and enum["pairs_total"] == [40, 40]
    assert "claims_total" not in enum
    assert enum["pairs"][0] == {"vendor": "V0", "aspect_id": "a1", "supported": [True, False]}
    # a pair only one file lists: None for the other
    assert {"vendor": "V1", "aspect_id": "a2", "supported": [True, None]} in enum["pairs"]
    assert {"vendor": "W1", "aspect_id": "a2", "supported": [None, True]} in enum["pairs"]
    assert len(enum["pairs"]) == 21


def _metrics(coverage, rate, prose="same"):
    return {"coverage": {"coverage": coverage},
            "unsupported": {"rate": rate, "prose_rate": rate if prose == "same" else prose}}


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


def test_pass_check_bounds_the_prose_rate_too():
    # supported grid cells pooled in cannot hide a rise in the prose answers' unsupported claims
    check = te.pass_check(_metrics(0.30, 0.20, prose=0.05), _metrics(0.60, 0.02, prose=0.08))
    assert check["unsupported_ok"] is True and check["prose_ok"] is False
    assert check["prose_rise_points"] == 3.0 and check["pass"] is False
    assert check["prose_rate"] == {"before": 0.05, "after": 0.08}
    assert "prose" in check["bar"]
    ok = te.pass_check(_metrics(0.30, 0.05, prose=0.05), _metrics(0.60, 0.06, prose=0.07))
    assert ok["prose_ok"] is True and ok["pass"] is True
    unknown = te.pass_check(_metrics(0.30, 0.05, prose=None), _metrics(0.60, 0.05))
    assert unknown["prose_ok"] is None and unknown["pass"] is False


def test_a_large_supported_grid_cannot_swamp_a_prose_regression():
    after = _after_grades(unsupported=(3, 3, 2))
    after[3] = _enum_grade("j4", 600, 30)  # 600 cells, every sampled one supported
    records = te.parse_grades(_before_grades() + after, KEY, QS, ASPECTS)
    report = te.report_for(records, {}, before="before", after="after")
    check = report["pass_check"]
    assert check["unsupported_ok"] is True  # pooled: 12 / 59 before, 8 / 619 after
    assert check["prose_rise_points"] == pytest.approx((8 / 19 - 4 / 19) * 100, abs=1e-4)
    assert check["prose_ok"] is False and check["pass"] is False
    # the pair rate is reported apart
    assert report["runs"]["after"]["unsupported"]["pair_rate"] == 0.0


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
    after = _after_grades(unsupported=(3, 5, 3), enum_supported=15)
    records = te.parse_grades(_before_grades() + after, KEY, QS, ASPECTS)
    check = te.report_for(records, {}, before="before", after="after")["pass_check"]
    # after: (11 + 60 * 0.5) / (19 + 60); before: (4 + 8) / (19 + 40)
    assert check["unsupported_rise_points"] == pytest.approx(
        (41 / 79 - 12 / 59) * 100, abs=0.01)
    assert check["unsupported_rate"]["before"] == pytest.approx(12 / 59)
    assert check["unsupported_ok"] is False
    assert check["pass"] is False


def test_a_question_missing_from_one_run_is_left_out_of_the_pass_check():
    before = [g for g in _before_grades()]
    after = [g for g in _after_grades() if g["item"] != "j2"]
    key = {k: v for k, v in KEY.items() if k != "j2"}
    records = te.parse_grades(before + after, key, QS, ASPECTS)
    report = te.report_for(records, {}, before="before", after="after")
    check = report["pass_check"]
    assert check["questions_compared"] == ["q1", "q3", "q4"]
    assert check["questions_dropped"] == ["q2"]
    assert report["warnings"] == [
        "pass check: q2 has no graded answer in run after; left out of the comparison"]
    # compared over q1 and q3 only (8 slots each run): before 1 + 0 covered, after 4 + 2
    assert check["coverage"]["before"] == 1 / 8
    assert check["coverage"]["after"] == (4 + 2) / 8
    # the per-run figures still cover everything each run has
    assert report["runs"]["before"]["coverage"]["slots"] == 16
    assert report["runs"]["after"]["coverage"]["slots"] == 8
    # the other way round names the run that lacks it
    key = {k: v for k, v in KEY.items() if k != "i1"}
    records = te.parse_grades([g for g in _before_grades() if g["item"] != "i1"]
                              + _after_grades(), key, QS, ASPECTS)
    report = te.report_for(records, {}, before="before", after="after")
    assert report["warnings"] == [
        "pass check: q1 has no graded answer in run before; left out of the comparison"]
    assert report["pass_check"]["questions_dropped"] == ["q1"]


def test_runs_with_no_question_in_common_skip_the_pass_check():
    key = {"i1": KEY["i1"], "j2": KEY["j2"]}
    grades = [_before_grades()[0], _after_grades()[1]]
    report = te.report_for(te.parse_grades(grades, key, QS, ASPECTS), {}, "before", "after")
    assert "no question in common" in report["pass_check"]["skipped"]
    assert len(report["warnings"]) == 2


def test_no_warning_when_both_runs_have_every_question():
    records = te.parse_grades(_before_grades() + _after_grades(), KEY, QS, ASPECTS)
    assert te.report_for(records, {}, "before", "after")["warnings"] == []


def test_stated_not_covered_without_a_corpus_file_warns():
    records = te.parse_grades(_before_grades(), {k: v for k, v in KEY.items()
                                                 if v["run"] == "before"}, QS, ASPECTS)
    report = te.report_for(records, None, None, None)
    assert len(report["warnings"]) == 1
    assert "2 stated_not_covered slots" in report["warnings"][0]
    assert "--corpus" in report["warnings"][0]
    # a corpus file (even an empty one) silences it, and so does having no such slot
    assert te.report_for(records, {}, None, None)["warnings"] == []
    clean = te.parse_grades([_sides_grade("i1", {"Acme": "snnn"}, 5, 1)],
                            {"i1": KEY["i1"]}, QS, ASPECTS)
    assert te.report_for(clean, None, None, None)["warnings"] == []


def test_the_corpus_must_name_known_sides_and_aspects():
    te.check_corpus({"Acme": {"a1": True, "a4": False}, " globex": {"a2": False},
                     "*": {"a3": True}}, QS, ASPECTS)
    with pytest.raises(te.GradeError) as err:
        te.check_corpus({"Initech": {"a1": True}, "Acme": {"zz": False}, "*": {"a9": True}},
                        QS, ASPECTS)
    assert err.value.problems == ["corpus: unknown side 'Initech'",
                                  "corpus: Acme: unknown aspect 'zz'",
                                  "corpus: *: unknown aspect 'a9'"]
    # a side only an enumeration question has is not a side to excuse
    only_enum = [q for q in QS if q["grading"] == "enumeration"]
    with pytest.raises(te.GradeError, match="unknown side 'Acme'"):
        te.check_corpus({"Acme": {"a1": True}}, only_enum, ASPECTS)


def test_the_real_corpus_sides_and_aspects_are_checked_by_the_command(tmp_path, capsys):
    questions = te.load_questions(REPO / "eval" / "topic_questions.yaml")
    aspects = te.load_aspects(["ransomware_protection"])
    ids = [a["id"] for a in aspects["ransomware_protection"]]
    te.check_corpus({"Veeam": dict.fromkeys(ids, True), "Azure Backup": {ids[0]: False}},
                    questions, aspects)
    with pytest.raises(te.GradeError, match="unknown side 'Veem'"):
        te.check_corpus({"Veem": {ids[0]: True}}, questions, aspects)
    key = tmp_path / "key.json"
    key.write_text("{}")
    bad = tmp_path / "corpus.json"
    bad.write_text(json.dumps({"Veeam": {"not_an_aspect": False}}))
    assert te.main(["score", str(key), str(tmp_path / "g.json"), "--corpus", str(bad)]) == 2
    assert "corpus: Veeam: unknown aspect 'not_an_aspect'" in capsys.readouterr().err


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
    grades[1]["aspects"]["acme"] = _slots("ssss")
    grades[2]["aspects"]["*"]["zz"] = "not_covered"
    grades[2]["unsupported_claims"] = ["x"] * 9
    grades.append({"item": "nope", "aspects": {}, "claims_total": 0, "unsupported_claims": []})
    grades.append(dict(grades[0]))
    text = _problems(grades, key)
    for expected in ("i1: Acme: a1: unknown label 'corpus_lacks'",
                     "i2: unknown side 'Initech'", "i2: side Acme graded twice",
                     "i3: *: unknown aspect 'zz'",
                     "i3: more unsupported claims than claims_total",
                     "nope: not in the key", "i1: graded twice"):
        assert expected in text
    grades = _before_grades()
    del grades[0]["claims_total"]
    assert "i1: claims_total must be a non-negative integer" in _problems(grades, key)


def test_a_prose_enumeration_sample_is_an_even_stride_of_distinct_pairs():
    key = {"i4": KEY["i4"]}
    # 40 pairs: every 2nd, so 20 (listing 30, the cap, is too many)
    many = _enum_grade("i4", 40, 16)
    many["pairs"] += [{"vendor": f"W{n}", "aspect_id": "a1", "supported": True}
                      for n in range(10)]
    assert ("i4: 30 pairs listed, expected 20 (stride 2 over pairs_total 40, "
            "starting with the first pair)") in _problems([many], key)
    short = _enum_grade("i4", 7, 5)
    del short["pairs"][-1]
    assert "i4: 6 pairs listed, expected 7 (all of pairs_total 7)" in _problems([short], key)
    under = _enum_grade("i4", 7, 5)
    under["pairs_total"] = 5
    assert "i4: 7 pairs listed, expected 5 (all of pairs_total 5)" in _problems([under], key)
    for total, listed in ((0, 0), (1, 1), (30, 30), (31, 16), (40, 20), (60, 30), (61, 21),
                          (100, 25), (300, 30)):
        grade = _enum_grade("i4", total, 0)
        assert len(grade["pairs"]) == listed
        assert te.parse_grades([grade], key, QS, ASPECTS)
        assert te.sample_size(total) == listed
    assert [te.pair_stride(n) for n in (1, 30, 31, 60, 61, 100)] == [1, 1, 2, 2, 3, 4]


def test_a_repeated_vendor_aspect_pair_is_an_error():
    key = {"i4": KEY["i4"]}
    grade = _enum_grade("i4", 3, 3)
    grade["pairs"][1] = {"vendor": " v0", "aspect_id": "a1", "supported": False}
    assert "i4: pair 1 repeats  v0 / a1" in _problems([grade], key)
    # the same vendor under another aspect, and another vendor under the same one, are fine
    grade["pairs"][1] = {"vendor": "V0", "aspect_id": "a2", "supported": False}
    assert te.parse_grades([grade], key, QS, ASPECTS)


def test_an_enumeration_item_needs_no_claims_and_extra_claims_are_ignored():
    key = {"i4": KEY["i4"]}
    grade = _enum_grade("i4", 3, 2)
    assert "claims_total" not in grade
    plain = te.parse_grades([grade], key, QS, ASPECTS)[0]
    grade["claims_total"] = 99
    grade["unsupported_claims"] = ["x"] * 99
    assert te.parse_grades([grade], key, QS, ASPECTS)[0] == plain
    assert (plain["claims_total"], plain["unsupported"]) == (0, [])


def test_an_unhashable_item_is_a_listed_problem_not_a_crash():
    key = {k: v for k, v in KEY.items() if v["run"] == "before"}
    grades = [*_before_grades(), {"item": ["i1"]}, {"item": {"a": 1}}, "text", {"item": None}]
    text = _problems(grades, key)
    assert "['i1']: not in the key" in text
    assert "{'a': 1}: not in the key" in text
    assert "entry 6: not an object" in text
    assert "None: not in the key" in text


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
                grades.append({"item": item["item"], "pairs_total": 1,
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
    assert first["runs"]["topic-after"]["enumeration"]["estimate"] == 1.0
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
                 "aspect_id", "supported", '"*"', "ceil(pairs_total / 30)", "stride"):
        assert word in brief
    # enumeration items no longer carry claims: the brief says so
    assert "`enumeration` items have no claims" in " ".join(brief.split())


def test_the_grading_brief_describes_the_lines_vendorkb_emits_and_only_those():
    brief = " ".join((REPO / "eval" / "TOPIC_GRADING_BRIEF.md").read_text().split())
    ransomware = load.validate(load.read_doc(REPO / "taxonomy" / "taxonomy.yaml")).topics[0]
    empty_grid = render_grid(Grid([], {}, None, ["Acme"]), ransomware)[0].split("\n")
    assert empty_grid[1:] == ["", "Facts current to: no facts found."]
    emitted = (REFUSAL, BLANK_NOTE, LEGEND, "the retrieved sources do not cover this",
               "No fact found for:", "Facts current to", empty_grid[0], empty_grid[2],
               closing_line(["X"]).removesuffix("X."), unconfirmed_line(["X"]).removesuffix("X."))
    for line in emitted:
        assert " ".join(line.split()) in brief, line
    # the refusal can be followed by the closing lines, and the brief says so
    assert "followed by the closing lines" in brief
    # grid items: the cells are given, graded exactly
    for word in ("`cells`", "exactly the given cells", "copy `pairs_total`"):
        assert word in brief, word
    # lines vendorkb never writes stay out of the brief
    for never in ("needs validation", "Checked", "Documented for", "Vendor ·", "(Vendor"):
        assert never not in brief, never
