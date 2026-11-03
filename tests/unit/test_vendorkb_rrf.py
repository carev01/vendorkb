from vendorkb.search import rrf


def test_rrf_scores_by_reciprocal_rank():
    assert rrf([["a", "b"], ["b", "c"]]) == ["b", "a", "c"]


def test_rrf_tie_breaks_by_first_appearance():
    assert rrf([["a"], ["b"]]) == ["a", "b"]
    assert rrf([["x", "y"], ["y", "x"]]) == ["x", "y"]
