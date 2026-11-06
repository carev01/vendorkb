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
