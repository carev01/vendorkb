import math

import pytest

from vendorkb.taxonomy import cluster, propose


def _unit(*xs: float) -> list[float]:
    n = math.sqrt(sum(x * x for x in xs))
    return [x / n for x in xs]


def test_near_duplicates_cluster_together():
    strings = ["immutable backups", "immutable backup", "instant vm recovery",
               "instant recovery of vms", "tape support"]
    vectors = [_unit(1, 0.02, 0), _unit(1, 0, 0.03), _unit(0, 1, 0.01),
               _unit(0.03, 1, 0), _unit(0, 0, 1)]
    groups = cluster.cluster_strings(strings, vectors, distance_threshold=0.2)
    assert sorted(sorted(g) for g in groups) == [[0, 1], [2, 3], [4]]


def test_threshold_controls_merging():
    vectors = [_unit(1, 0), _unit(1, 1), _unit(0, 1)]
    assert len(cluster.cluster_strings(["a", "b", "c"], vectors,
                                       distance_threshold=0.01)) == 3
    assert len(cluster.cluster_strings(["a", "b", "c"], vectors,
                                       distance_threshold=1.5)) == 1


def test_degenerate_inputs():
    assert cluster.cluster_strings([], [], distance_threshold=0.2) == []
    assert cluster.cluster_strings(["x"], [[1.0, 0.0]], distance_threshold=0.2) == [[0]]
    with pytest.raises(ValueError):
        cluster.cluster_strings(["x", "y"], [[1.0, 0.0]], distance_threshold=0.2)


def test_normalize():
    assert cluster.normalize("  Immutable\tBackups \n") == "immutable backups"
    assert cluster.normalize("S3   Object  LOCK") == "s3 object lock"


@pytest.mark.parametrize(("raw", "expected"), [
    ("Immutable-Backups", "immutable_backups"),
    ("instant vm recovery", "instant_vm_recovery"),
    ("  Role ", "role"),
])
def test_snake_ids(raw, expected):
    assert propose._snake(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("AWS.RDS", "aws.rds"),
    ("aws rds.postgres-sql", "aws_rds.postgres_sql"),
    (".aws..s3.", "aws.s3"),
])
def test_dotted_ids(raw, expected):
    assert propose._dotted(raw) == expected


def test_snake_refuses_an_empty_id():
    with pytest.raises(ValueError):
        propose._snake("--")


@pytest.mark.parametrize(("text", "quoted"), [("yes", True), ("1:20", True), ("null", True),
                                              ("365", True), ("admin roles", False),
                                              ("dd boost", False)])
def test_scalar_quotes_what_yaml_1_1_would_misread(text, quoted):
    out = propose._scalar(text)
    assert out == text and (type(out) is not str) == quoted


def test_long_tail_path():
    assert str(propose.long_tail_path("taxonomy/taxonomy.yaml")) == "taxonomy/long_tail.yaml"
    assert str(propose.long_tail_path("x/t.yaml")) == "x/t.long_tail.yaml"


def test_cluster_accepts_float32_arrays():
    import numpy as np
    vectors = np.asarray([_unit(1, 0.01), _unit(1, 0), _unit(0, 1)], dtype=np.float32)
    assert cluster.cluster_strings(["a", "b", "c"], vectors, distance_threshold=0.2) == [
        [0, 1], [2]]
