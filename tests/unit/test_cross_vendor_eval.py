from types import SimpleNamespace

import json

from eval.cross_vendor import (facet_recall, gold_by_facet, load_done, percentile,
                               summarize, vendor_noise, vendor_recall)


def _hit(doc_id):
    return SimpleNamespace(doc_id=doc_id)


def test_vendor_recall_basic():
    assert vendor_recall({"A", "B", "C", "D"}, set(), {"A", "B", "X"}) == 0.5


def test_vendor_recall_unclear_excluded_from_both_sides():
    # D is unclear: neither required nor counted as a hit
    assert vendor_recall({"A", "B", "D"}, {"D"}, {"A"}) == 0.5
    assert vendor_recall({"A", "D"}, {"D"}, {"A", "D"}) == 1.0


def test_vendor_recall_empty_expected_is_none():
    assert vendor_recall(set(), set(), {"A"}) is None
    assert vendor_recall({"D"}, {"D"}, {"D"}) is None


def test_vendor_noise_excludes_expected_and_unclear():
    assert vendor_noise({"A"}, {"U"}, {"A", "U", "Z", "Y"}) == ["Y", "Z"]
    assert vendor_noise({"A"}, set(), {"A"}) == []


def test_facet_recall_missing_facet_scores_zero():
    gold = {"Veeam": {"d1"}, "AWS": {"d2", "d3"}}
    assert facet_recall(gold, [_hit("d1"), _hit("d9")]) == 0.5
    assert facet_recall(gold, [_hit("d1"), _hit("d3")]) == 1.0
    assert facet_recall(gold, []) == 0.0


def test_facet_recall_no_facets_is_none():
    assert facet_recall({}, [_hit("d1")]) is None
    assert facet_recall({"Veeam": set()}, [_hit("d1")]) is None


def test_gold_by_facet_maps_docs_by_vendor_or_product():
    facets = [("vendor", "AWS"), ("product", "Azure Backup")]
    docs = {"d1": ("AWS", "AWS Backup"), "d2": ("Microsoft", "Azure Backup"),
            "d3": ("Veeam", "VBR")}
    assert gold_by_facet(facets, {"d1", "d2", "d3"}, docs) == {
        "AWS": {"d1"}, "Azure Backup": {"d2"}}


def test_percentile_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([5.0], 90) == 5.0
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == 9


def test_summarize_splits_by_holdout():
    results = [
        {"id": "e1", "type": "cross_vendor", "shape": "enumeration", "holdout": False,
         "vendor_recall": 1.0, "vendor_noise": ["X"], "latency_s": 1.0, "rerank_tokens": 10},
        {"id": "e2", "type": "cross_vendor", "shape": "enumeration", "holdout": True,
         "vendor_recall": 0.5, "vendor_noise": [], "latency_s": 2.0, "rerank_tokens": 30},
        {"id": "e3", "type": "cross_vendor", "shape": "enumeration", "holdout": True,
         "vendor_recall": None, "vendor_noise": [], "latency_s": 3.0, "rerank_tokens": 20},
        {"id": "c1", "type": "cross_vendor", "shape": "comparison", "holdout": False,
         "facet_recall": 0.5, "latency_s": 1.0, "rerank_tokens": 0},
        {"id": "s1", "type": "single_vendor", "shape": "single", "holdout": False,
         "pool_rank": 3, "latency_s": 1.0, "rerank_tokens": 0},
        {"id": "s2", "type": "single_vendor", "shape": "single", "holdout": False,
         "pool_rank": None, "latency_s": 1.0, "rerank_tokens": 0},
    ]
    s = summarize(results, rerank=True)
    assert s["calibration"]["enumeration"] == {
        "n": 1, "n_scored": 1, "n_unlabelled": 0, "vendor_recall": 1.0,
        "noise_per_question": 1.0}
    assert s["holdout"]["enumeration"]["n"] == 2
    assert s["holdout"]["enumeration"]["n_scored"] == 1
    assert s["holdout"]["enumeration"]["n_unlabelled"] == 1
    assert s["holdout"]["enumeration"]["vendor_recall"] == 0.5
    assert s["calibration"]["comparison"]["facet_recall"] == 0.5
    assert s["calibration"]["single_vendor"]["pool_recall"] == 0.5
    assert s["calibration"]["single_vendor"]["recall@10"] == 0.5
    assert s["holdout"]["rerank_tokens_per_question"] == 25.0
    assert s["holdout"]["latency_s"]["enumeration"] == {"p50": 2.0, "p90": 3.0}
    assert "comparison" not in s["holdout"]


def test_summarize_noise_ignores_unlabelled_questions():
    results = [
        {"id": "a", "type": "cross_vendor", "shape": "enumeration", "holdout": True,
         "vendor_recall": 1.0, "vendor_noise": ["X"], "latency_s": 1.0},
        # unlabelled: recall None, every hit vendor listed as noise; must not count
        {"id": "b", "type": "cross_vendor", "shape": "enumeration", "holdout": True,
         "vendor_recall": None, "vendor_noise": ["P", "Q", "R", "S"], "latency_s": 1.0},
    ]
    e = summarize(results, rerank=False)["holdout"]["enumeration"]
    assert e["noise_per_question"] == 1.0 and e["n_unlabelled"] == 1


def test_summarize_counts_errors_and_rerank_failures():
    results = [
        {"id": "a", "type": "single_vendor", "shape": "single", "holdout": False,
         "pool_rank": 1, "latency_s": 2.0, "rerank_ok": True},
        {"id": "b", "type": "single_vendor", "shape": "single", "holdout": False,
         "pool_rank": 2, "latency_s": 4.0, "rerank_ok": False},
        {"id": "c", "type": "single_vendor", "shape": "single", "holdout": False,
         "error": "boom", "latency_s": 0.01},
    ]
    s = summarize(results, rerank=True)["calibration"]
    assert s["n_errors"] == 1 and s["rerank_failures"] == 1
    assert s["single_vendor"]["n"] == 2
    assert s["latency_s"]["single"]["p50"] == 2.0  # the fast failure is not in it


def test_load_done_skips_errored_failed_and_torn_lines(tmp_path):
    f = tmp_path / "r.jsonl"
    rows = [{"id": "ok"}, {"id": "err", "error": "x"}, {"id": "nr", "rerank_ok": False},
            {"id": "ok2", "rerank_ok": True}]
    f.write_text("\n".join(json.dumps(r) for r in rows) + '\n{"id": "torn', encoding="utf-8")
    assert [r["id"] for r in load_done(f)] == ["ok", "ok2"]
    assert load_done(tmp_path / "missing.jsonl") == []


async def test_score_questions_stops_at_the_rerank_token_cap(tmp_path):
    from eval.cross_vendor import score_questions

    rr = SimpleNamespace(tokens_used=0)
    ran = []

    async def run(q):
        ran.append(q["id"])
        rr.tokens_used += 60
        return {"id": q["id"], "rerank_tokens": 60}

    qs = [{"id": i} for i in ("a", "b", "c", "d")]
    jsonl = tmp_path / "r.jsonl"
    results = []
    stopped = await score_questions(qs, run, jsonl, results, reranker=rr,
                                    max_rerank_tokens=100, log=lambda _m: None)
    assert stopped is True and ran == ["a", "b"]  # checked before each question
    assert [json.loads(x)["id"] for x in jsonl.read_text().splitlines()] == ["a", "b"]
    # resumed rows count against the cap too; done ids are skipped
    rr.tokens_used = 0
    ran.clear()
    prior = [{"id": "a", "rerank_tokens": 90}]
    stopped = await score_questions(qs, run, tmp_path / "s.jsonl", prior, reranker=rr,
                                    max_rerank_tokens=100, log=lambda _m: None)
    assert stopped is True and ran == ["b"]
    # no reranker (free mode): never stops
    ran.clear()
    assert await score_questions(qs, run, tmp_path / "t.jsonl", [], reranker=None,
                                 max_rerank_tokens=0, log=lambda _m: None) is False
    assert ran == ["a", "b", "c", "d"]


def test_score_parser_has_a_rerank_token_cap():
    from eval.cross_vendor import build_parser

    args = build_parser().parse_args(["score", "--out", "x.json"])
    assert args.max_rerank_tokens == 3_000_000
    args = build_parser().parse_args(["score", "--out", "x.json", "--max-rerank-tokens", "5"])
    assert args.max_rerank_tokens == 5


def test_pool_depth_names_the_pool_each_shape_sees():
    from eval.cross_vendor import _pool_depth

    assert _pool_depth("enumeration", legacy=False, rerank=True) == "enum-up-to-60-reranked"
    assert _pool_depth("comparison", legacy=False, rerank=False) == "5-per-facet-rrf"
    assert _pool_depth("single", legacy=False, rerank=True) == "top10-reranked"
    assert _pool_depth("enumeration", legacy=True, rerank=False) == "top40-rrf"


def test_calibrate_floor_prints_both_rules_and_the_ruling(tmp_path, capsys):
    from eval import calibrate_floor

    qs = [{"id": "a", "expected_vendors": [{"vendor": v} for v in "ABCDE"],
           "unclear_vendors": []},
          {"id": "h", "holdout": True, "expected_vendors": [{"vendor": "A"}],
           "unclear_vendors": []}]
    run = {"per_question": [
        {"id": "a", "shape": "enumeration", "vendor_best_scores": {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.3, "Z": 0.4}},
        {"id": "h", "shape": "enumeration", "vendor_best_scores": {"A": 0.7}}]}
    (tmp_path / "q.yaml").write_text(json.dumps(qs), encoding="utf-8")
    (tmp_path / "r.json").write_text(json.dumps(run), encoding="utf-8")
    floors = calibrate_floor.main([str(tmp_path / "r.json"), str(tmp_path / "q.yaml")])
    # calibration recall 1.0 up to 0.30, 0.8 up to 0.60, lower above
    assert floors == {"recall_0.8": 0.6, "recall_0.95": 0.3}
    out = capsys.readouterr().out
    assert "recall >= 0.8" in out and "recall >= 0.95" in out
    assert "0.46" in out and "ruling" in out
