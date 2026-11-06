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


@pytest.mark.parametrize(("kind", "raw", "expected"), [
    ("feature", "Immutable-Backups", "immutable_backups"),
    ("feature", "instant vm recovery", "instant_vm_recovery"),
    ("feature", "3-2-1 rule", "f_3_2_1_rule"),
    ("target", "AWS.RDS", "aws.rds"),
    ("target", "aws rds.postgres-sql", "aws_rds.postgres_sql"),
    ("target", ".aws..s3.", "aws.s3"),
])
def test_slug_ids(kind, raw, expected):
    assert propose.slug_id(raw, kind) == expected


def test_parent_of():
    assert propose.parent_of("aws.rds.postgres") == "aws.rds"
    assert propose.parent_of("aws") is None
