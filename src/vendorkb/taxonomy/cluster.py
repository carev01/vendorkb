"""String normalization and agglomerative clustering of embedded strings."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def normalize(text: str) -> str:
    """Lowercase, whitespace runs collapsed to one space, trimmed."""
    return " ".join(text.split()).lower()


def cluster_strings(strings: list[str], vectors: list[list[float]], *,
                    distance_threshold: float) -> list[list[int]]:
    """Index groups of ``strings`` whose embeddings lie within ``distance_threshold`` (cosine
    distance, average linkage). Every index is in exactly one group; groups are ordered by
    their first index. Memory is O(n^2) in the number of strings."""
    if len(strings) != len(vectors):
        raise ValueError(f"{len(strings)} strings but {len(vectors)} vectors")
    if not strings:
        return []
    if len(strings) == 1:
        return [[0]]
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=distance_threshold,
    ).fit_predict(np.asarray(vectors, dtype=np.float64))
    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(int(label), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])
