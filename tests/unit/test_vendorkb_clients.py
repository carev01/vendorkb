import json

import httpx
import pytest

from vendorkb.clients import Embedder, Reranker
from vendorkb.config import VendorkbSettings


def _settings(**kw):
    return VendorkbSettings(
        _env_file=None, embed_base_url="http://e/v1", rerank_base_url="http://r/v1",
        rerank_api_key="k", **kw)


async def test_embedder_batches_at_the_cap():
    sizes = []

    def handler(req: httpx.Request) -> httpx.Response:
        inputs = json.loads(req.content)["input"]
        sizes.append(len(inputs))
        # reply out of order to prove results are ordered by index
        data = [{"index": i, "embedding": [float(len(t))] + [0.0] * 767}
                for i, t in enumerate(inputs)]
        return httpx.Response(200, json={"data": list(reversed(data))})

    texts = ["x" * (i + 1) for i in range(70)]
    async with Embedder(_settings(), transport=httpx.MockTransport(handler)) as emb:
        out = await emb.embed(texts)
    assert sizes == [32, 32, 6]
    assert [v[0] for v in out] == [float(i + 1) for i in range(70)]


async def test_embedder_rejects_wrong_dimension():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 5}]})

    emb = Embedder(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError):
        await emb.embed(["a"])
    await emb.aclose()


async def test_reranker_returns_none_on_http_500_transport_error_and_bad_payload():
    def h500(req):
        return httpx.Response(500, text="boom")

    def herr(req):
        raise httpx.ConnectError("down")

    def hbad(req):
        return httpx.Response(200, json={"nope": 1})

    for h in (h500, herr, hbad):
        rr = Reranker(_settings(), transport=httpx.MockTransport(h))
        assert await rr.rerank("q", ["a", "b"], 2) is None
        await rr.aclose()


async def test_reranker_drops_out_of_range_indices():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"data": [
            {"index": 1, "relevance_score": 0.9}, {"index": 7, "relevance_score": 0.8},
            {"index": -1, "relevance_score": 0.7}, {"index": 0, "relevance_score": 0.1}]})

    async with Reranker(_settings(), transport=httpx.MockTransport(handler)) as rr:
        out = await rr.rerank("q", ["a", "b"], 5)
    assert out == [(1, 0.9), (0, 0.1)]
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["top_k"] == 2
