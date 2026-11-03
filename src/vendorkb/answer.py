from __future__ import annotations

import logging
import re

from openai import AsyncOpenAI

from vendorkb.scope import ScopeIndex, detect_scope
from vendorkb.search import Hit, search_docs

log = logging.getLogger(__name__)

REFUSAL = "I don't have enough information to answer that from the available sources."

SYSTEM_RULES = (
    "You answer questions about backup-industry vendor documentation.\n"
    "Rules:\n"
    "- Answer only from the numbered sources in the user message; use no outside knowledge.\n"
    "- Cite with [n] markers after each claim, where n is the source number. "
    "Several sources may be cited as [1][2].\n"
    "- Never write URLs or links; only [n] markers identify sources.\n"
    "- If the sources do not answer the question, reply with exactly this sentence and "
    f"nothing else: {REFUSAL}"
)

_GROUP = re.compile(r"([ ]?)\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def build_prompt(question: str, hits: list[Hit]) -> str:
    """User message: numbered sources then the question. Rules live in SYSTEM_RULES."""
    blocks = []
    for n, h in enumerate(hits, start=1):
        trail = " > ".join(h.section_path)
        blocks.append(f"[{n}] {h.title} | {trail}\n{h.text}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n"


def resolve_markers(raw: str, hits: list[Hit]) -> tuple[str, list[dict], int]:
    valid = len(hits)
    order: list[int] = []
    unknown = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal unknown
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(2))]
        keep = []
        for n in nums:
            if 1 <= n <= valid:
                keep.append(n)
                if n not in order:
                    order.append(n)
            else:
                unknown += 1
        # A fully removed marker takes ONE preceding space with it; nothing else is touched.
        return f"{m.group(1)}[{', '.join(str(n) for n in keep)}]" if keep else ""

    text = _GROUP.sub(repl, raw)
    citations = []
    for n in order:
        h = hits[n - 1]
        citations.append({"marker": n, "chunk_id": h.chunk_id, "doc_id": h.doc_id,
                          "url": h.url, "title": h.title, "section_path": h.section_path})
    return text, citations, unknown


def _is_refusal(text: str) -> bool:
    return text.strip().startswith(REFUSAL)


async def _call_model(client: AsyncOpenAI, settings, user: str) -> tuple[str | None, dict, str | None]:
    """One model call with a single retry. Returns (content, usage_totals, error)."""
    totals: dict = {"prompt_tokens": 0, "completion_tokens": 0, "cost": None}
    error: str | None = None
    for attempt in (1, 2):
        try:
            resp = await client.chat.completions.create(
                model=settings.judge_model,
                messages=[{"role": "system", "content": SYSTEM_RULES},
                          {"role": "user", "content": user}],
                temperature=0,
                max_tokens=4000,
                extra_body={"reasoning": {"effort": settings.answer_reasoning_effort},
                            "provider": {"sort": "throughput", "allow_fallbacks": True},
                            "usage": {"include": True}},
            )
        except Exception as e:  # noqa: BLE001 - recorded, retried once, then reported
            error = f"{type(e).__name__}: {e}"
            log.warning("vendorkb answer model call failed (attempt %d): %s", attempt, error)
            continue
        u = getattr(resp, "usage", None)
        if u is not None:
            totals["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            totals["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            cost = getattr(u, "cost", None)
            if cost is not None:
                totals["cost"] = (totals["cost"] or 0) + cost
        choices = getattr(resp, "choices", None)
        message = choices[0].message if choices else None
        content = getattr(message, "content", None)
        if content is None or content.strip() == "":
            error = "empty reply"
            log.warning("vendorkb answer model returned an empty reply (attempt %d)", attempt)
            continue
        return content, totals, None
    return None, totals, error


async def answer(conn, question: str, *, settings, embedder, reranker,
                 scope_index: ScopeIndex, client: AsyncOpenAI) -> dict:
    vendors, products = detect_scope(question, scope_index)
    relaxed = False
    scoped = bool(vendors or products)
    result = await search_docs(conn, question, embedder=embedder, reranker=reranker,
                               vendor=vendors or None, product=products or None)
    if scoped and not result.hits:
        relaxed = True
        result = await search_docs(conn, question, embedder=embedder, reranker=reranker)
    out: dict = {"query": question, "answer": None, "citations": [], "unknown_markers": 0,
                 "uncited": False,
                 "scope": {"vendors": vendors, "products": products, "relaxed": relaxed},
                 "rerank_ok": result.rerank_ok, "retrieved": len(result.hits),
                 "usage": None, "error": None}
    if not result.hits:
        out["answer"] = REFUSAL
        return out
    content, usage, error = await _call_model(client, settings, build_prompt(question, result.hits))
    out["usage"] = usage
    if content is None:
        out["error"] = error
        return out
    if _is_refusal(content):
        out["answer"] = content.strip()
        return out
    text, citations, unknown = resolve_markers(content, result.hits)
    text = text.strip()
    if text == "":
        out["unknown_markers"] = unknown
        out["error"] = "empty reply"
        return out
    out.update(answer=text, citations=citations, unknown_markers=unknown,
               uncited=not citations)
    return out
