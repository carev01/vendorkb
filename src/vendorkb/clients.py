"""HTTP clients: embedder, reranker (Voyage), DocExtractor, plus the corpus manifest."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import httpx

from vendorkb.config import VendorkbSettings

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


class Embedder(_Client):
    def __init__(self, settings: VendorkbSettings, *,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._s = settings
        self._http = httpx.AsyncClient(timeout=120.0, transport=transport)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        step = max(1, self._s.embed_max_batch)
        for i in range(0, len(texts), step):
            out.extend(await self._embed_batch(texts[i:i + step]))
        return out

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        resp = await self._http.post(
            self._s.embed_base_url.rstrip("/") + "/embeddings",
            json={"model": self._s.embed_model, "input": batch})
        resp.raise_for_status()
        rows = sorted(resp.json()["data"], key=lambda r: r["index"])
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


@dataclass
class CorpusEntry:
    article_id: str
    vendor: str
    product: str | None
    source: str | None
    title: str
    url: str | None
    graph_content_hash: str | None


def load_corpus(path: Path) -> list[CorpusEntry]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CorpusEntry(**row) for row in rows]
