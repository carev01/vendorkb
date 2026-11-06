"""Question shape: one facet (single), two or more compared (comparison), or a cross-vendor sweep (enumeration)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from vendorkb.scope import ScopeIndex, find_facets, is_cross_vendor

Facet = tuple[Literal["vendor", "product"], str]

_COMPARISON_WORDING = re.compile(
    r"\bcompar\w*|\bvs\.?(?!\w)|\bversus\b|\bdifference between\b|\beach\b|\bboth\b|\bbetter\b|\bworse\b"
    r"|\bwhich one\b", re.IGNORECASE)
# "which tools/products/..." wording: an enumeration unless the comparison rules hold
_SWEEP = re.compile(
    r"\b(which|what)\s+(of (these|the documented)\s+)?(backup\s+)?"
    r"(vendors?|products?|tools?|solutions?|consoles?|platforms?|software|offerings?)\b"
    r"|\bany products\b", re.IGNORECASE)


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


def _product_vendor(product: str, index: ScopeIndex) -> str:
    if product in index.product_vendors:
        return index.product_vendors[product]
    owners = _vendor_of(product, index)
    return next(iter(owners)) if len(owners) == 1 else f"?{product}"  # unknown: its own vendor


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
        cross = len({_product_vendor(p, index) for p in products}) >= 2
        if (cross or _COMPARISON_WORDING.search(question)
                or _has_or_pair(question, facets)):
            return Shape("comparison", facets)
        if is_cross_vendor(question) or _SWEEP.search(question):
            return Shape("enumeration", facets)
        return Shape("single", facets)
    if is_cross_vendor(question) or _SWEEP.search(question):
        return Shape("enumeration", facets)
    return Shape("single", facets)
