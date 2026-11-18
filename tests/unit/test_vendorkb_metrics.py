"""Usage metrics of the MCP app (metrics): counts, latency, the eval marker, tokens, cost,
vendors and build info; ``/metrics`` is never a route of the app."""
import httpx
import pytest

from tests.unit.test_mcp_tools import FakeBackend
from vendorkb import answer as answer_mod
from vendorkb import metrics
from vendorkb.config import VendorkbSettings
from vendorkb.mcp_app import build_app

HDRS = {"accept": "application/json, text/event-stream", "content-type": "application/json",
        "host": "vendorkb-mcp:8080"}


@pytest.fixture(autouse=True)
def _fresh_metrics():
    metrics.reset_for_tests()
    yield
    metrics.reset_for_tests()


def _settings(**over):
    return VendorkbSettings(_env_file=None, **{
        "mcp_bearer_token": "main-key", "mcp_full_access_token": "full-key", **over})


def _call(tool, arguments):
    return {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments}}


async def _post(app, body, token="full-key", **headers):
    async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://vendorkb-mcp:8080") as c:
        r = await c.post("/mcp", json=body,
                         headers={**HDRS, "authorization": f"Bearer {token}", **headers})
    assert r.status_code == 200
    return r


def _app(backend=None, **over):
    return build_app(_settings(**over), {"main": FakeBackend(), "full": backend or FakeBackend()})


def _value(name, **labels):
    return metrics.REGISTRY.get_sample_value(name, labels)


async def test_tool_call_counted():
    await _post(_app(), _call("search_docs", {"query": "immutability"}))
    assert _value("vendorkb_mcp_requests_total", tool="search_docs", key="full",
                  client="user", outcome="ok") == 1
    assert _value("vendorkb_mcp_request_seconds_count", tool="search_docs", client="user") == 1


@pytest.mark.parametrize("header, client", [("eval", "eval"), ("EVAL", "eval"),
                                            ("other", "user")])
async def test_eval_header_labels_client(header, client):
    name = "X-VENDORKB-CLIENT" if header == "EVAL" else "X-Vendorkb-Client"
    await _post(_app(), _call("search_docs", {"query": "immutability"}), **{name: header})
    assert _value("vendorkb_mcp_requests_total", tool="search_docs", key="full",
                  client=client, outcome="ok") == 1
    other = "user" if client == "eval" else "eval"
    assert _value("vendorkb_mcp_requests_total", tool="search_docs", key="full",
                  client=other, outcome="ok") is None


class Failing(FakeBackend):
    async def search(self, query, vendor, product, k):
        raise RuntimeError("database down")


async def test_tool_error_counted():
    await _post(_app(Failing()), _call("search_docs", {"query": "immutability"}))
    assert _value("vendorkb_mcp_requests_total", tool="search_docs", key="full",
                  client="user", outcome="error") == 1
    assert _value("vendorkb_mcp_request_seconds_count", tool="search_docs", client="user") == 1


async def test_abstention_counted():
    backend = FakeBackend(answer={"answer": answer_mod.REFUSAL, "citations": []})
    await _post(_app(backend), _call("answer_question", {"question": "What?"}))
    assert _value("vendorkb_mcp_requests_total", tool="answer_question", key="full",
                  client="user", outcome="abstained") == 1


def test_answer_tokens_and_cost():
    out = {"usage": {"prompt_tokens": 1000, "completion_tokens": 200}}
    metrics.record_answer(out, _settings(answer_price_in_per_mtok=2.0,
                                         answer_price_out_per_mtok=8.0))
    assert _value("vendorkb_answer_tokens_total", kind="prompt", client="user") == 1000
    assert _value("vendorkb_answer_tokens_total", kind="completion", client="user") == 200
    assert _value("vendorkb_answer_cost_usd_total", client="user") == pytest.approx(0.0036)

    metrics.reset_for_tests()
    metrics.record_answer(out, _settings())
    assert _value("vendorkb_answer_tokens_total", kind="prompt", client="user") == 1000
    assert _value("vendorkb_answer_cost_usd_total", client="user") is None


def test_answer_without_usage_records_nothing():
    metrics.record_answer({"usage": None, "scope": None}, _settings(answer_price_in_per_mtok=2.0))
    metrics.record_answer({"usage": {"prompt_tokens": None, "completion_tokens": None}},
                          _settings(answer_price_in_per_mtok=2.0))
    assert _value("vendorkb_answer_tokens_total", kind="prompt", client="user") is None
    assert _value("vendorkb_answer_cost_usd_total", client="user") is None


def test_vendor_counted():
    metrics.record_answer({"scope": {"vendors": ["Veeam", "Rubrik"]}}, _settings())
    assert _value("vendorkb_questions_by_vendor_total", vendor="Veeam", client="user") == 1
    assert _value("vendorkb_questions_by_vendor_total", vendor="Rubrik", client="user") == 1


async def test_answer_question_records_tokens_and_vendors():
    backend = FakeBackend(answer={"answer": "Yes [1].", "citations": [],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                                  "scope": {"vendors": ["Veeam"]}})
    await _post(_app(backend), _call("answer_question", {"question": "Veeam?"}),
                **{"X-Vendorkb-Client": "eval"})
    assert _value("vendorkb_answer_tokens_total", kind="prompt", client="eval") == 10
    assert _value("vendorkb_questions_by_vendor_total", vendor="Veeam", client="eval") == 1
    assert _value("vendorkb_mcp_requests_total", tool="answer_question", key="full",
                  client="eval", outcome="ok") == 1


async def test_metrics_not_on_app_routes():
    app = _app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://vendorkb-mcp:8080") as c:
        r = await c.get("/metrics")
    assert r.status_code == 404


def test_build_info():
    _app(vendorkb_version="sha-abc1234")
    assert _value("vendorkb_build_info", version="sha-abc1234") == 1


def test_start_does_nothing_on_port_zero(monkeypatch):
    started = []
    monkeypatch.setattr(metrics, "start_http_server", lambda *a, **kw: started.append(a))
    metrics.start(_settings(metrics_port=0))
    assert started == []
    metrics.start(_settings(metrics_port=9311))
    assert [a[0] for a in started] == [9311]


def test_agent_eval_sends_marker(monkeypatch):
    from eval import agent_eval

    sent = {}

    class Reply:
        status_code = 200
        text = ('{"jsonrpc": "2.0", "id": 1, "result": {"content": '
                '[{"type": "text", "text": "ok"}]}}')

    def post(*a, **kw):
        sent.update(kw.get("headers") or {})
        return Reply()

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("VENDORKB_MCP_TOKEN", "t")
    agent_eval.mcp_call("search_docs", {"query": "immutability"})
    assert sent.get("X-Vendorkb-Client") == "eval"


async def test_busy_metrics_port_keeps_mcp_serving(monkeypatch):
    def busy(*a, **kw):
        raise OSError(98, "Address already in use")

    refreshed = []

    async def refresh(*a, **kw):
        refreshed.append(a)

    monkeypatch.setattr(metrics, "start_http_server", busy)
    monkeypatch.setattr("vendorkb.mcp_app.refresh_loop", refresh)
    app = build_app(_settings(metrics_port=9311))  # no backends: the lifespan starts metrics
    async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://vendorkb-mcp:8080") as c:
        r = await c.get("/healthz")
    assert (r.status_code, r.text) == (200, "ok")
    assert refreshed == []  # no refresh loop without a metrics server
    # the failed bind left the pipeline collector unregistered, so a later start can register it
    assert metrics.REGISTRY.get_sample_value("vendorkb_metrics_refresh_ok") is None


def test_malformed_usage_never_raises():
    metrics.record_answer({"usage": {"prompt_tokens": "x", "completion_tokens": 2},
                           "scope": {"vendors": ["Veeam"]}},
                          _settings(answer_price_in_per_mtok=2.0))


async def test_bookkeeping_failure_keeps_the_tool_result(monkeypatch):
    class Broken:
        def labels(self, *a):
            raise ValueError("bad label")

    monkeypatch.setattr(metrics, "_requests", Broken())
    async with metrics.observe("search_docs", "full") as obs:
        result = 42
    assert (result, obs.outcome) == (42, "ok")
    with pytest.raises(RuntimeError, match="the tool's own"):
        async with metrics.observe("search_docs", "full"):
            raise RuntimeError("the tool's own")


async def test_answer_question_with_malformed_usage_still_answers():
    backend = FakeBackend(answer={"answer": "Yes [1].", "citations": [],
                                  "usage": {"prompt_tokens": "x"}})
    r = await _post(_app(backend), _call("answer_question", {"question": "Veeam?"}))
    assert "Yes [1]." in r.text
    assert _value("vendorkb_mcp_requests_total", tool="answer_question", key="full",
                  client="user", outcome="ok") == 1
