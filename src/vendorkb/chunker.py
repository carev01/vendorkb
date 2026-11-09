"""Heading-aware section chunker for DocExtractor markdown.

Chunks are plain-text containers (markdown is not rendered). Every chunk text starts with
``"{title}\\n{path}\\n\\n"`` where ``path`` is the heading trail below the title.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_SETEXT_H1 = re.compile(r"^ {0,3}={3,}[ \t]*$")
_SETEXT_H2 = re.compile(r"^ {0,3}-{3,}[ \t]*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")


@dataclass(frozen=True)
class Chunk:
    section_path: tuple[str, ...]
    ordinal: int
    text: str
    token_count: int


def approx_tokens(text: str) -> int:
    return len(text) // 4


_MONTHS = ("jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
           "|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
_WHEN = (r"(?:\d{4}\\?-\d{1,2}\\?-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?Z?)?"  # ISO-like
         rf"|(?:{_MONTHS})\.? \d{{1,2}}, \d{{4}}"
         r"|today|yesterday|this week|last week"
         r"|\d+ (?:minutes?|hours?|days?|weeks?|months?|years?) ago"
         r"|a (?:day|week|month|year) ago)")
# A line that only says when the page was updated (run on text whose spaces are collapsed).
# Never a bare "Updated ..." line: that is a changelog entry.
_DATE_LINE = re.compile(rf"^ ?(?:last )?updated(?: ?:| on)? ?{_WHEN} ?$",
                        re.IGNORECASE | re.MULTILINE)
_SPACES = re.compile(r"[^\S\n]+")  # whitespace within a line
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def reuse_key(text: str) -> str:
    """The text with what re-syncs change without changing a fact removed: date lines
    ("Last Updated ..."), markdown link and image targets, backslash escapes and whitespace.
    Two chunk texts with equal keys say the same thing."""
    text = _DATE_LINE.sub("", _SPACES.sub(" ", text))
    while (reduced := _LINK.sub(r"\1", text)) != text:  # a linked image takes two passes
        text = reduced
    return " ".join(text.replace("\\", "").split())


@dataclass
class _Block:
    text: str
    is_table: bool


@dataclass
class _Section:
    path: tuple[str, ...]
    blocks: list[_Block]


def _prefix(title: str, path: tuple[str, ...]) -> str:
    trail = " > ".join(path[1:])
    return f"{title}\n{trail}\n\n" if trail else f"{title}\n\n"


def _same_title(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def _parse(title: str, markdown: str) -> list[_Section]:
    lines = markdown.splitlines()
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    cur = _Section((title,), [])
    buf: list[str] = []
    in_table = False
    fence: str | None = None
    seen_heading = False

    def flush() -> None:
        nonlocal in_table
        text = "\n".join(buf).strip("\n")
        if text.strip():
            cur.blocks.append(_Block(text, in_table))
        buf.clear()
        in_table = False

    def heading(level: int, text: str) -> None:
        nonlocal cur, seen_heading
        first = not seen_heading
        seen_heading = True
        if first and level == 1 and _same_title(text, title):
            return
        sections.append(cur)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        cur = _Section((title, *[t for _, t in stack]), [])

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if fence is not None:
            buf.append(line)
            if stripped.startswith(fence * 3):
                fence = None
            i += 1
            continue
        if stripped.startswith(("```", "~~~")):
            flush()
            fence = stripped[0]
            buf.append(line)
            i += 1
            continue
        if stripped.startswith("|"):
            if not in_table:
                flush()
                in_table = True
            buf.append(line)
            i += 1
            continue
        if in_table:
            flush()
        if not stripped:
            flush()
            i += 1
            continue
        m = _ATX.match(line)
        if m and m.group(2).strip():
            flush()
            heading(len(m.group(1)), m.group(2).strip())
            i += 1
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if _SETEXT_H1.match(nxt) or _SETEXT_H2.match(nxt):
            flush()
            heading(1 if _SETEXT_H1.match(nxt) else 2, stripped)
            i += 2
            continue
        buf.append(line)
        i += 1
    flush()
    sections.append(cur)
    return [s for s in sections if s.blocks]


def _hard_split(text: str, budget_chars: int) -> list[str]:
    return [text[i : i + budget_chars] for i in range(0, len(text), budget_chars)]


def _split_table(text: str, budget_chars: int) -> list[_Block]:
    """Pack a table's rows into groups that each repeat the header (header + separator rows).

    A table with no separator line has no header to repeat. A row too big for one group even
    alone is hard-split by characters, each piece carrying the header.
    """
    lines = text.split("\n")
    header = ""
    if len(lines) > 1 and _TABLE_SEPARATOR.match(lines[1]):
        header = "\n".join(lines[:2])
        lines = lines[2:]
    if header and len(header) + 1 >= budget_chars:
        # a header that leaves no room for rows cannot be repeated; keep it as ordinary rows
        lines = [*header.split("\n"), *lines]
        header = ""
    lead = header + "\n" if header else ""
    room = budget_chars - len(lead)
    out: list[_Block] = []
    group: list[str] = []
    size = 0
    for row in lines:
        if len(row) > room:
            if group:
                out.append(_Block(lead + "\n".join(group), True))
                group, size = [], 0
            out.extend(_Block(lead + piece, True) for piece in _hard_split(row, room))
            continue
        if group and size + len(row) + 1 > room:
            out.append(_Block(lead + "\n".join(group), True))
            group, size = [], 0
        group.append(row)
        size += len(row) + 1
    if group:
        out.append(_Block(lead + "\n".join(group), True))
    return out


def _split_block(block: _Block, budget_chars: int) -> list[_Block]:
    """A block too big for one chunk is split: tables into header-repeating row groups,
    paragraphs by lines."""
    if len(block.text) <= budget_chars:
        return [block]
    if block.is_table:
        return _split_table(block.text, budget_chars)
    out: list[_Block] = []
    group: list[str] = []
    size = 0
    for line in block.text.split("\n"):
        pieces = _hard_split(line, budget_chars) if len(line) > budget_chars else [line]
        for piece in pieces:
            if group and size + len(piece) + 1 > budget_chars:
                out.append(_Block("\n".join(group), False))
                group, size = [], 0
            group.append(piece)
            size += len(piece) + 1
    if group:
        out.append(_Block("\n".join(group), False))
    return out


def _split_section(title: str, sec: _Section, max_tokens: int) -> list[str]:
    """Pack a section's blocks greedily into bodies that fit ``max_tokens`` with the prefix."""
    prefix = _prefix(title, sec.path)
    budget_chars = max(1, max_tokens * 4 - len(prefix))
    blocks: list[_Block] = []
    for b in sec.blocks:
        blocks.extend(_split_block(b, budget_chars))
    bodies: list[str] = []
    group: list[str] = []
    for b in blocks:
        cand = "\n\n".join([*group, b.text])
        if group and approx_tokens(prefix + cand) > max_tokens:
            bodies.append("\n\n".join(group))
            group = []
        group.append(b.text)
    if group:
        bodies.append("\n\n".join(group))
    return bodies


def _related(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Same parent path, or one is the parent of the other."""
    return a[:-1] == b[:-1] or a == b[:-1] or b == a[:-1]


def chunk_markdown(
    title: str, markdown: str, *, min_tokens: int = 300, max_tokens: int = 1200
) -> list[Chunk]:
    units: list[list] = []  # [path, body]
    for sec in _parse(title, markdown):
        for body in _split_section(title, sec, max_tokens):
            units.append([sec.path, body])

    def tokens(path: tuple[str, ...], body: str) -> int:
        return approx_tokens(_prefix(title, path) + body)

    def labelled(path: tuple[str, ...], body: str, target: tuple[str, ...]) -> str:
        if path != target and len(path) > 1:
            return f"{path[-1]}\n{body}"
        return body

    merged: list[list] = []
    for path, body in units:
        tiny = tokens(path, body) < min_tokens
        if merged and tiny and _related(merged[-1][0], path):
            tpath, tbody = merged[-1]
            combined = tbody + "\n\n" + labelled(path, body, tpath)
            if tokens(tpath, combined) <= max_tokens:
                merged[-1][1] = combined
                continue
        if (
            len(merged) == 1
            and tokens(*merged[0]) < min_tokens
            and _related(merged[0][0], path)
        ):
            fpath, fbody = merged[0]
            combined = labelled(fpath, fbody, path) + "\n\n" + body
            if tokens(path, combined) <= max_tokens:
                merged[0] = [path, combined]
                continue
        merged.append([path, body])

    chunks: list[Chunk] = []
    for ordinal, (path, body) in enumerate(merged):
        text = _prefix(title, path) + body
        chunks.append(Chunk(path, ordinal, text, approx_tokens(text)))
    return chunks
