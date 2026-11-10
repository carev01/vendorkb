"""Question shape: one facet (single), two or more compared (comparison), or a cross-vendor sweep (enumeration)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from vendorkb.scope import ScopeIndex, find_facets, is_cross_vendor

Facet = tuple[Literal["vendor", "product"], str]

_COMPARISON_WORDING = re.compile(
    r"\bcompar\w*|\bvs\.?(?!\w)|\bversus\b|\bdifference between\b|\bbetter\b|\bworse\b"
    r"|\bwhich one\b", re.IGNORECASE)
# the wording that compares on its own; "better"/"worse" and "X or Y" can sit in a workload phrase
_STRONG_COMPARISON = re.compile(
    r"\bcompar\w*|\bvs\.?(?!\w)|\bversus\b|\bdifference between\b|\bwhich one\b",
    re.IGNORECASE)
# "which tools/products/..." wording: an enumeration unless the comparison rules hold
_SWEEP = re.compile(
    r"\b(which|what)\s+(of (these|the documented)\s+)?(backup\s+)?"
    r"(vendors?|products?|tools?|solutions?|consoles?|platforms?|software|offerings?)\b"
    r"|\bany products\b", re.IGNORECASE)
# with "vendors" wording and one facet: the facet is excluded from, or rivals, the answer
_EXCLUDING = re.compile(
    r"\b(other|besides|except|apart from|alternatives?|compet(e|es|ing|itors?))\b", re.IGNORECASE)
# a vendor this often names a workload ("our Google Workspace", "Salesforce metadata")
WORKLOAD_VENDORS = frozenset({"Microsoft", "Google", "AWS", "IBM", "Salesforce", "Red Hat"})


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
    if product in index.product_vendors:
        owners.add(index.product_vendors[product])
    for source, products in index.source_products.items():
        if product in products and source in index.source_vendors:
            owners.add(index.source_vendors[source])
    return owners


def _product_vendor(product: str, index: ScopeIndex) -> str:
    if product in index.product_vendors:
        return index.product_vendors[product]
    owners = _vendor_of(product, index)
    return next(iter(owners)) if len(owners) == 1 else f"?{product}"  # unknown: its own vendor


def _follows(words: str, vendor: str, question: str, index: ScopeIndex) -> bool:
    return any(re.search(rf"\b({words})\s+{re.escape(surface)}(?!\w)", question, re.IGNORECASE)
               for surface in _surfaces(vendor, index))


def _is_subject(vendor: str, question: str, index: ScopeIndex) -> bool:
    """The vendor is what is asked about ("does Microsoft offer", "from Veeam")."""
    return _follows("does|do|did|from|by|of|in|at", vendor, question, index)


def _is_protected(vendor: str, question: str, index: ScopeIndex) -> bool:
    """The vendor is what gets protected ("protect Cohesity", "back up Salesforce")."""
    return _follows("protect|back up|backup of", vendor, question, index)


def _has_or_pair(question: str, facets: list[Facet]) -> bool:
    names = [re.escape(n) for _, n in facets]
    return any(re.search(rf"(?<!\w){a} or {b}(?!\w)", question, re.IGNORECASE)
               for a in names for b in names if a != b)


def named_facets(question: str, index: ScopeIndex) -> list[Facet]:
    """The facets the question names: its vendors and products, a vendor of a named product
    counted once, as the product (a product is more specific than its vendor)."""
    vendors, products = find_facets(question, index)
    owners = {v for p in products for v in _vendor_of(p, index)}
    return [("vendor", v) for v in vendors if v not in owners] + [("product", p) for p in products]


def detect_shape(question: str, index: ScopeIndex) -> Shape:
    facets = named_facets(question, index)
    products = [name for kind, name in facets if kind == "product"]
    if len(facets) >= 2:  # named facets: compared or asked about together, never a sweep
        cross = len({_product_vendor(p, index) for p in products}) >= 2
        if (cross or _COMPARISON_WORDING.search(question)
                or _has_or_pair(question, facets)):
            if not cross and not _STRONG_COMPARISON.search(question):
                # "back up to AWS or Azure": a workload vendor is no side on a weak signal
                sides = [f for f in facets
                         if not (f[0] == "vendor" and f[1] in WORKLOAD_VENDORS)]
                if not sides:
                    # every facet is a workload vendor: no subject to fall back on, so keep
                    # them all ("Is Microsoft or Google better for email archiving?")
                    if _COMPARISON_WORDING.search(question):
                        return Shape("comparison", facets)
                    return Shape("single", facets)
                if len(sides) < 2:
                    return Shape("single", sides)
                if len(sides) < len(facets):
                    facets = sides
                    if not (_COMPARISON_WORDING.search(question) or _has_or_pair(question, sides)):
                        return Shape("single", facets)
            return Shape("comparison", facets)
        return Shape("single", facets)
    if not facets:
        if is_cross_vendor(question) or _SWEEP.search(question):
            return Shape("enumeration", facets)
        return Shape("single", facets)
    kind, name = facets[0]
    if is_cross_vendor(question):
        # "other vendors besides Veeam" sweeps; "storage vendors Veeam integrates with" does not
        if _EXCLUDING.search(question):
            return Shape("enumeration", facets)
        # a workload vendor, or its product ("which vendors support Entra ID backup?": the
        # phrase is also the name of Microsoft's product), is what gets protected
        owner = name if kind == "vendor" else _product_vendor(name, index)
        if owner in WORKLOAD_VENDORS and not _is_subject(name, question, index):
            return Shape("enumeration", facets)
        return Shape("single", facets)
    if _SWEEP.search(question) and kind == "vendor" and not _is_subject(name, question, index) and (
            name in WORKLOAD_VENDORS or _is_protected(name, question, index)):
        # the vendor is a workload ("back up our Google Workspace"), not what is asked about
        return Shape("enumeration", facets)
    return Shape("single", facets)
