"""HTTP clients: embedder, reranker (Voyage), DocExtractor."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Literal, Self

import httpx

from vendorkb.config import VendorkbSettings
from vendorkb.feed import FeedItem, parse_line

logger = logging.getLogger(__name__)


class _Client:
    _http: httpx.AsyncClient

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc: BaseException | None, tb: TracebackType | None) -> None:
        await self.aclose()


InputType = Literal["document", "query"]
EMBED_RETRIES = 6
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class Embedder(_Client):
    def __init__(self, settings: VendorkbSettings, *,
                 transport: httpx.AsyncBaseTransport | None = None,
                 retry_base_delay: float = 2.0) -> None:
        self._s = settings
        self._http = httpx.AsyncClient(timeout=120.0, transport=transport)
        self._retry_base_delay = retry_base_delay
        self.tokens_used = 0  # from the provider's usage.total_tokens, when it reports one

    async def embed(self, texts: list[str], *,
                    input_type: InputType = "document") -> list[list[float]]:
        out: list[list[float]] = []
        step = max(1, self._s.embed_max_batch)
        for i in range(0, len(texts), step):
            out.extend(await self._embed_batch(texts[i:i + step], input_type))
        return out

    async def _embed_batch(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        payload: dict = {"model": self._s.embed_model, "input": batch}
        if self._s.embed_input_type:
            payload["input_type"] = input_type
        if self._s.embed_output_dimension is not None:
            payload["output_dimension"] = self._s.embed_output_dimension
        headers = ({"Authorization": f"Bearer {self._s.embed_api_key}"}
                   if self._s.embed_api_key else {})
        url = self._s.embed_base_url.rstrip("/") + "/embeddings"
        for attempt in range(EMBED_RETRIES):
            resp = await self._http.post(url, json=payload, headers=headers)
            if resp.status_code not in _RETRY_STATUS or attempt == EMBED_RETRIES - 1:
                break
            retry_after = resp.headers.get("retry-after", "")
            delay = (float(retry_after) if retry_after.replace(".", "", 1).isdigit()
                     else self._retry_base_delay * 2 ** attempt)
            logger.warning("embedder HTTP %s, retrying in %.1fs", resp.status_code, delay)
            await asyncio.sleep(delay)
        resp.raise_for_status()
        body = resp.json()
        self.tokens_used += int((body.get("usage") or {}).get("total_tokens") or 0)
        rows = sorted(body["data"], key=lambda r: r["index"])
        if len(rows) != len(batch):
            raise RuntimeError(f"embedder returned {len(rows)} vectors for {len(batch)} inputs")
        vectors = [[float(x) for x in r["embedding"]] for r in rows]
        for v in vectors:
            if len(v) != self._s.embed_dim:
                raise RuntimeError(
                    f"embedding has dimension {len(v)}, expected {self._s.embed_dim}")
        return vectors


class Reranker(_Client):
    def __init__(self, settings: VendorkbSettings, *,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._s = settings
        self._http = httpx.AsyncClient(timeout=30.0, transport=transport)

    async def rerank(self, query: str, documents: list[str],
                     top_k: int) -> list[tuple[int, float]] | None:
        """Best-first (index, score) pairs, or None when relevance could not be scored."""
        if not documents:
            return []
        payload = {"model": self._s.rerank_model, "query": query, "documents": documents,
                   "top_k": min(top_k, len(documents))}
        try:
            resp = await self._http.post(
                self._s.rerank_base_url.rstrip("/") + "/rerank", json=payload,
                headers={"Authorization": f"Bearer {self._s.rerank_api_key}"})
        except httpx.HTTPError as exc:
            logger.warning("rerank transport error (%s); caller must degrade visibly", exc)
            return None
        if resp.status_code != 200:
            logger.warning("rerank returned HTTP %s; caller must degrade visibly",
                           resp.status_code)
            return None
        try:
            out = [(int(r["index"]), float(r["relevance_score"]))
                   for r in resp.json()["data"]]
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("rerank payload unreadable (%s); caller must degrade visibly", exc)
            return None
        return [(i, s) for i, s in out if 0 <= i < len(documents)]


class DocExtractor(_Client):
    def __init__(self, settings: VendorkbSettings, *,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._s = settings
        self._http = httpx.AsyncClient(
            timeout=60.0, transport=transport,
            verify=settings.docext_verify_tls if transport is None else True)

    async def article(self, article_id: str) -> dict:
        resp = await self._http.get(
            f"{self._s.docext_base_url.rstrip('/')}/api/articles/{article_id}",
            headers={"X-API-Key": self._s.docext_read_key})
        resp.raise_for_status()
        return resp.json()


    async def delta(self, *, since: str | None = None,
                    bootstrap_after: str | None = None) -> AsyncIterator[FeedItem]:
        """Stream the delta feed. A malformed line raises ValueError and a transport
        error propagates; either way the caller must treat the stream as dropped."""
        params = {k: v for k, v in (("since", since), ("bootstrap_after", bootstrap_after))
                  if v is not None}
        async with self._http.stream(
                "GET", f"{self._s.docext_base_url.rstrip('/')}/api/articles/delta",
                params=params, headers={"X-API-Key": self._s.docext_read_key},
                timeout=httpx.Timeout(60.0, read=300.0)) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                item = parse_line(line)
                if item is not None:
                    yield item

    async def source_names(self) -> dict[str, str]:
        """source_id -> name. /api/sources caps at 200 and ignores paging, so ask per product."""
        base = self._s.docext_base_url.rstrip("/")
        headers = {"X-API-Key": self._s.docext_read_key}
        resp = await self._http.get(f"{base}/api/products", params={"limit": 200},
                                    headers=headers)
        resp.raise_for_status()
        product_ids = [p["id"] for p in resp.json()["products"]]
        sem = asyncio.Semaphore(8)

        async def one(pid: str) -> list[dict]:
            async with sem:
                r = await self._http.get(f"{base}/api/sources",
                                         params={"product_id": pid, "limit": 200},
                                         headers=headers)
                r.raise_for_status()
                sources = r.json()["sources"]
            if len(sources) >= 200:
                logger.warning("product %s returned %d sources (API cap 200); list may be "
                               "truncated", pid, len(sources))
            return sources

        out: dict[str, str] = {}
        for sources in await asyncio.gather(*(one(p) for p in product_ids)):
            for s in sources:
                out[str(s["id"])] = s["name"]
        return out
