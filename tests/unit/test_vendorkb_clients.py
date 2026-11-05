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
    assert sizes == [8] * 8 + [6]
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


async def test_embedder_sends_voyage_options_and_counts_tokens():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.headers.get("authorization"), json.loads(req.content)))
        n = len(json.loads(req.content)["input"])
        return httpx.Response(200, json={
            "data": [{"index": i, "embedding": [0.0] * 1024} for i in range(n)],
            "usage": {"total_tokens": 7 * n}})

    s = _settings(embed_api_key="vk", embed_input_type=True, embed_output_dimension=1024,
                  embed_dim=1024, embed_max_batch=2)
    async with Embedder(s, transport=httpx.MockTransport(handler)) as emb:
        await emb.embed(["a", "b", "c"])
        await emb.embed(["q"], input_type="query")
        assert emb.tokens_used == 28
    assert [a for a, _ in seen] == ["Bearer vk"] * 3
    assert [b["input_type"] for _, b in seen] == ["document", "document", "query"]
    assert all(b["output_dimension"] == 1024 for _, b in seen)


async def test_embedder_omits_voyage_options_by_default():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.headers.get("authorization"), json.loads(req.content)))
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 768}]})

    async with Embedder(_settings(), transport=httpx.MockTransport(handler)) as emb:
        await emb.embed(["a"], input_type="query")
        assert emb.tokens_used == 0
    auth, body = seen[0]
    assert auth is None and "input_type" not in body and "output_dimension" not in body


async def test_embedder_retries_rate_limits_then_succeeds():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 768}]})

    async with Embedder(_settings(), transport=httpx.MockTransport(handler),
                        retry_base_delay=0) as emb:
        assert len(await emb.embed(["a"])) == 1
    assert len(calls) == 3


async def test_embedder_gives_up_after_retries_and_does_not_retry_client_errors():
    calls = []

    def h429(req):
        calls.append("429")
        return httpx.Response(429)

    def h400(req):
        calls.append("400")
        return httpx.Response(400, text="bad")

    for h in (h429, h400):
        emb = Embedder(_settings(), transport=httpx.MockTransport(h), retry_base_delay=0)
        with pytest.raises(httpx.HTTPStatusError):
            await emb.embed(["a"])
        await emb.aclose()
    assert calls.count("429") == 6 and calls.count("400") == 1
