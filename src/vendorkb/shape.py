"""Question shape: one facet (single), two or more compared (comparison), or a cross-vendor sweep (enumeration)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from vendorkb.scope import ScopeIndex, find_facets, is_cross_vendor

Facet = tuple[Literal["vendor", "product"], str]

_COMPARISON_WORDING = re.compile(
    r"\bcompar\w*|\bvs\.?(?!\w)|\bversus\b|\bdiffer\w*|\beach\b|\bboth\b|\bbetter\b|\bworse\b"
    r"|\bwhich one\b", re.IGNORECASE)
# product-style cross-vendor wording; only an enumeration when the question names nothing
_PRODUCT_SWEEP = re.compile(
    r"\b(which|what|any) products\b|\bwhich (of (these|the documented) )?(backup )?(vendors|products)\b",
    re.IGNORECASE)


@dataclass(frozen=True)
class Shape:
    kind: Literal["single", "comparison", "enumeration"]
    facets: list[Facet]


def _surfaces(vendor: str, index: ScopeIndex) -> list[str]:
    return [vendor, *(a for a, targets in index.aliases.items() if vendor in targets)]


def _starts_with(text: str, prefix: str) -> bool:
    return re.match(rf"{re.escape(prefix)}(?!\w)", text, re.IGNORECASE) is not None


def _vendor_of(product: str, index: ScopeIndex) -> set[str]:
    owners = {v for v in index.vendors
              if any(_starts_with(product, s) for s in _surfaces(v, index))}
    for source, products in index.source_products.items():
        if product in products and source in index.source_vendors:
            owners.add(index.source_vendors[source])
    return owners


def _has_or_pair(question: str, facets: list[Facet]) -> bool:
    names = [re.escape(n) for _, n in facets]
    return any(re.search(rf"(?<!\w){a} or {b}(?!\w)", question, re.IGNORECASE)
               for a in names for b in names if a != b)


def detect_shape(question: str, index: ScopeIndex) -> Shape:
    vendors, products = find_facets(question, index)
    # a product is more specific than its vendor: one facet, not two
    owners = {v for p in products for v in _vendor_of(p, index)}
    facets: list[Facet] = [("vendor", v) for v in vendors if v not in owners]
    facets += [("product", p) for p in products]
    if len(facets) >= 2:
        if (len(products) >= 2 or _COMPARISON_WORDING.search(question)
                or _has_or_pair(question, facets)):
            return Shape("comparison", facets)
        return Shape("single", facets)
    if (is_cross_vendor(question) and len(facets) <= 1) or (
            not facets and _PRODUCT_SWEEP.search(question)):
        return Shape("enumeration", facets)
    return Shape("single", facets)
