from eval.map_replay import bm25_ranker, score, words


def test_words_drop_stopwords_short_words_and_plurals():
    assert words("The backups of SQL databases, by a DB") == ["backup", "sql", "database"]


def test_bm25_ranks_by_shared_words_within_allowed_minus_excluded():
    rank = bm25_ranker({"a": "immutable backup copies", "b": "backup schedule frequency",
                        "c": "immutable object lock retention", "d": "nothing shared"})
    # b and c share one word each with the query; the shorter document ranks first
    assert rank("immutable backups", {"a", "b", "c", "d"}, set(), 5) == ["a", "b", "c"]
    assert rank("immutable backups", {"a", "b", "c", "d"}, {"a"}, 5) == ["b", "c"]
    assert rank("immutable backups", {"b", "d"}, set(), 5) == ["b"]  # zero scores left out


def test_score_precision_recall_and_unlabelled_answers():
    items = [
        {"item_id": "1", "acceptable": ["x"], "assigned_2026_11_08": "x"},
        {"item_id": "2", "acceptable": [None], "assigned_2026_11_08": None},
        {"item_id": "3", "acceptable": ["y", None], "assigned_2026_11_08": "z"},
        {"item_id": "4", "acceptable": ["w"], "assigned_2026_11_08": None},
    ]
    s = score({"1": "x", "2": "q", "3": None, "4": None}, items)
    # answered 1 and 2: 1 is right, 2 should be none (and is a dimension the labels never saw)
    assert (s["answered"], s["precision"], s["unlabelled"]) == (2, 0.5, 1)
    # items with an acceptable dimension: 1, 3 (none also fine there) and 4
    assert (s["recall"], s["right"]) == (2 / 3, 2)
