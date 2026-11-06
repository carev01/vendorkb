"""String normalization and agglomerative clustering of embedded strings."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def normalize(text: str) -> str:
    """Lowercase, whitespace runs collapsed to one space, trimmed."""
    return " ".join(text.split()).lower()


def cluster_strings(strings: list[str], vectors: Sequence[Sequence[float]] | np.ndarray, *,
                    distance_threshold: float) -> list[list[int]]:
    """Index groups of ``strings`` whose embeddings lie within ``distance_threshold`` (cosine
    distance, average linkage). Every index is in exactly one group; groups are ordered by
    their first index. Memory is O(n^2) in the number of strings: scipy builds a condensed
float64 distance matrix and a copy, about 8 * n^2 bytes in all (1.8 GB at n = 15,000)."""
    if len(strings) != len(vectors):
        raise ValueError(f"{len(strings)} strings but {len(vectors)} vectors")
    if not strings:
        return []
    if len(strings) == 1:
        return [[0]]
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=distance_threshold,
    ).fit_predict(np.asarray(vectors, dtype=np.float32))
    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(int(label), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])
