"""DocExtractor delta-feed line types and parsing."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapStart:
    next_since: str


@dataclass(frozen=True)
class CursorLine:
    next_since: str
    count: int


@dataclass(frozen=True)
class Content:
    doc_id: uuid.UUID
    change_type: str
    source_id: uuid.UUID | None
    vendor: str
    product: str | None
    title: str | None
    url: str | None
    content_hash: str | None
    markdown: str | None


@dataclass(frozen=True)
class Tombstone:
    doc_id: uuid.UUID


FeedItem = BootstrapStart | CursorLine | Content | Tombstone


def _uuid(value: object) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid uuid in feed line: {value!r}") from exc


def parse_line(line: str) -> FeedItem | None:
    """Parse one NDJSON line; None for control/change types we do not act on."""
    try:
        rec = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"feed line is not JSON: {exc}") from exc
    if not isinstance(rec, dict):
        raise ValueError("feed line is not a JSON object")
    control = rec.get("control")
    if control in ("bootstrap_start", "cursor"):
        since = rec.get("next_since")
        if not isinstance(since, str) or not since:
            raise ValueError(f"{control} line needs a non-empty string next_since")
        if control == "bootstrap_start":
            return BootstrapStart(since)
        count = rec.get("count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ValueError("cursor line needs an integer count")
        return CursorLine(since, count)
    if control is not None:
        return None
    change = rec.get("change_type")
    if change not in ("added", "updated", "removed"):
        return None
    if rec.get("id") is None:
        raise ValueError("feed record has no id")
    doc_id = _uuid(rec["id"])
    if change == "removed":
        return Tombstone(doc_id)
    if not rec.get("vendor"):
        raise ValueError("feed content record has no vendor")
    sid = rec.get("source_id")
    return Content(
        doc_id=doc_id, change_type=change,
        source_id=_uuid(sid) if sid else None,
        vendor=rec["vendor"], product=rec.get("product"), title=rec.get("title"),
        url=rec.get("source_url"), content_hash=rec.get("content_hash"),
        markdown=rec.get("content_markdown"))
