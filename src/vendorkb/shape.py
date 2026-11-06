"""Question shape: one facet (single), two or more (comparison), or a cross-vendor sweep (enumeration)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from vendorkb.scope import ScopeIndex, find_facets, is_cross_vendor

Facet = tuple[Literal["vendor", "product"], str]


@dataclass(frozen=True)
class Shape:
    kind: Literal["single", "comparison", "enumeration"]
    facets: list[Facet]


def _surfaces(vendor: str, index: ScopeIndex) -> list[str]:
    return [vendor, *(a for a, targets in index.aliases.items() if vendor in targets)]


def _vendor_of(product: str, index: ScopeIndex) -> set[str]:
    owners = {v for v in index.vendors if product.lower().startswith(v.lower())}
    owners |= {v for v in index.vendors
               if any(product.lower().startswith(s.lower()) for s in _surfaces(v, index))}
    for source, products in index.source_products.items():
        if product in products and source in index.source_vendors:
            owners.add(index.source_vendors[source])
    return owners


def _is_qualifier(vendor: str, products: list[str], question: str, index: ScopeIndex) -> bool:
    """A vendor word that only qualifies something else: "Cohesity FortKnox", "Explorer for Microsoft SQL"."""
    for surface in _surfaces(vendor, index):
        v = re.escape(surface)
        if re.search(rf"(?<!\w)(for|from) {v}(?!\w)", question, re.IGNORECASE):
            return True
        if any(re.search(rf"(?<!\w){v} {re.escape(p)}(?!\w)", question, re.IGNORECASE)
               for p in products):
            return True
    return False


def detect_shape(question: str, index: ScopeIndex) -> Shape:
    vendors, products = find_facets(question, index)
    # a product is more specific than its vendor: one facet, not two
    named_by_product = {v for p in products for v in _vendor_of(p, index)}
    facets: list[Facet] = [("vendor", v) for v in vendors if v not in named_by_product
                           and not _is_qualifier(v, products, question, index)]
    facets += [("product", p) for p in products]
    if len(facets) >= 2:
        return Shape("comparison", facets)
    if is_cross_vendor(question):
        return Shape("enumeration", facets)
    return Shape("single", facets)
