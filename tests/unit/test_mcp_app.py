"""The MCP HTTP app (mcp_app): bearer gate, host allow-list, health."""
import httpx
import pytest

from vendorkb.config import VendorkbSettings
from vendorkb.mcp_app import build_app
from tests.unit.test_mcp_tools import FakeBackend

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "0"}}}
HDRS = {"accept": "application/json, text/event-stream", "content-type": "application/json",
        "host": "vendorkb-mcp:8080"}


def _app(token="s3cret"):
    return build_app(VendorkbSettings(_env_file=None, mcp_bearer_token=token),
                     {"main": FakeBackend()})


async def _post(app, headers):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://vendorkb-mcp:8080") as c:
        return await c.post("/mcp", json=INIT, headers=headers)


def test_no_token_refuses_to_build():
    with pytest.raises(ValueError, match="MCP_BEARER_TOKEN"):
        _app(token="")


@pytest.mark.parametrize("auth", [None, "Bearer wrong", "Basic czNjcmV0", "Bearer"])
async def test_missing_or_wrong_token_is_401(auth):
    headers = dict(HDRS) if auth is None else {**HDRS, "authorization": auth}
    r = await _post(_app(), headers)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


async def test_the_right_token_reaches_the_server():
    app = _app()
    async with app.router.lifespan_context(app):
        r = await _post(app, {**HDRS, "authorization": "Bearer s3cret"})
    assert r.status_code == 200 and "vendorkb" in r.text  # serverInfo.name


async def test_a_foreign_host_is_rejected():
    app = _app()
    async with app.router.lifespan_context(app):
        r = await _post(app, {**HDRS, "host": "evil.example",
                              "authorization": "Bearer s3cret"})
    assert r.status_code == 421


async def test_health_needs_no_token():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app()),
                                 base_url="http://vendorkb-mcp:8080") as c:
        r = await c.get("/healthz")
    assert (r.status_code, r.text) == (200, "ok")


@pytest.mark.parametrize("headers", [{"x-api-key": "s3cret"},
                                     {"x-api-key": "s3cret", "authorization": "Bearer nope"},
                                     {"authorization": "Bearer s3cret", "x-api-key": "nope"}])
async def test_the_secret_as_an_api_key_reaches_the_server(headers):
    app = _app()
    async with app.router.lifespan_context(app):
        r = await _post(app, {**HDRS, **headers})
    assert r.status_code == 200 and "vendorkb" in r.text


@pytest.mark.parametrize("headers", [{"x-api-key": "nope"}, {"x-api-key": ""},
                                     {"x-api-key": "Bearer s3cret"},
                                     {"x-api-key": "nope", "authorization": "Bearer nope"}])
async def test_a_wrong_api_key_is_401(headers):
    r = await _post(_app(), {**HDRS, **headers})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


class Vendors(FakeBackend):
    def __init__(self, names):
        super().__init__()
        self.names = names

    async def vendors(self):
        return [{"vendor": n, "documents": 1, "capability_facts": 0,
                 "products": [{"product": n, "documents": 1}]} for n in self.names]


LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "list_vendors", "arguments": {}}}


def _two_keys(**over):
    s = VendorkbSettings(_env_file=None, **{
        "mcp_bearer_token": "old", "mcp_full_access_token": "new",
        "mcp_hidden_vendors": "Cohesity", "mcp_restricted_dsn": "postgresql://r@x/db", **over})
    # the same rows behind both keys: the restricted one is guarded on top
    return build_app(s, {"restricted": Vendors(["Cohesity", "Veeam"]),
                         "full": Vendors(["Cohesity", "Veeam"])})


async def _list(app, token):
    async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://vendorkb-mcp:8080") as c:
        r = await c.post("/mcp", json=LIST, headers={**HDRS, "authorization": f"Bearer {token}"})
    return r


async def test_each_key_reaches_its_own_backend():
    old, new = await _list(_two_keys(), "old"), await _list(_two_keys(), "new")
    assert old.status_code == new.status_code == 200
    assert "Veeam" in old.text and "Cohesity" not in old.text  # hidden from the old key
    assert "Veeam" in new.text and "Cohesity" in new.text
    assert (await _list(_two_keys(), "other")).status_code == 401


@pytest.mark.parametrize("over, match", [
    ({"mcp_restricted_dsn": ""}, "MCP_RESTRICTED_DSN"),
    ({"mcp_full_access_token": "old"}, "must differ"),
])
def test_contradictory_keys_refuse_to_build(over, match):
    with pytest.raises(ValueError, match=match):
        _two_keys(**over)


def test_a_restricted_dsn_without_hidden_vendors_is_unused():
    from vendorkb.mcp_app import mcp_keys
    keys = mcp_keys(VendorkbSettings(_env_file=None, mcp_bearer_token="old",
                                     mcp_restricted_dsn="postgresql://r@x/db",
                                     mcp_hidden_vendors=" , ", vendorkb_dsn="postgresql://o@x/db"))
    assert [(k.label, k.hidden, k.dsn) for k in keys] == [("main", (), "postgresql://o@x/db")]
