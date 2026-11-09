"""Text helpers with no heavy imports: the answer path and the taxonomy loader use them, and
must not pull in the clustering stack (numpy, scikit-learn) that ``taxonomy.cluster`` needs."""
from __future__ import annotations


def normalize(text: str) -> str:
    """Lowercase, whitespace runs collapsed to one space, trimmed."""
    return " ".join(text.split()).lower()
