import json

import pytest

from vendorkb import cli
from vendorkb.claims import refresh


class FakeConn:
    async def close(self):
        pass


@pytest.fixture
def stubbed(monkeypatch):
    calls: dict = {"clients": []}

    async def connect(dsn):
        return FakeConn()

    async def init_schema(conn, *, embed_dim):
        pass

    def client(settings):
        calls["clients"].append((settings.extract_backend, settings.extract_base_url,
                                 settings.extract_api_key))
        return object()

    monkeypatch.setattr(cli, "connect", connect)
    monkeypatch.setattr(cli, "init_schema", init_schema)
    monkeypatch.setattr(cli, "_extract_client", client)
    return calls


def test_refresh_wires_both_endpoints_and_the_flags(stubbed, monkeypatch, capsys):
    waited = []

    async def fake_wait(base_url, api_key, *, timeout_s):
        waited.append((base_url, api_key, timeout_s))

    async def fake_refresh(conn, *, ready, **kw):
        await ready()
        stubbed.update(kw)
        return refresh.RefreshReport(new_chunks=3, due=True)

    monkeypatch.setattr(refresh, "wait_until_ready", fake_wait)
    monkeypatch.setattr(refresh, "refresh", fake_refresh)
    for name, value in {"EXTRACT_BACKEND": "openai", "EXTRACT_BASE_URL": "http://serve/v1",
                        "EXTRACT_API_KEY": "serve-key", "MAP_BACKEND": "openai",
                        "MAP_BASE_URL": "http://router/v1", "MAP_API_KEY": "map-key"}.items():
        monkeypatch.setenv(name, value)
    assert cli.main(["refresh", "--settle-minutes", "30", "--extract-limit", "500",
                     "--map-max-usd", "0.5", "--warmup-seconds", "60"]) == 0
    assert json.loads(capsys.readouterr().out)["new_chunks"] == 3
    assert stubbed["clients"] == [("openai", "http://serve/v1", "serve-key"),
                                  ("openai", "http://router/v1", "map-key")]
    assert waited == [("http://serve/v1", "serve-key", 60.0)]
    assert (stubbed["settle_minutes"], stubbed["extract_limit"],
            stubbed["map_max_usd"]) == (30.0, 500, 0.5)


def test_map_client_falls_back_to_the_extract_endpoint(stubbed, monkeypatch, capsys):
    from vendorkb.taxonomy import mapper

    async def fake_map(conn, **kw):
        return mapper.MapReport()

    monkeypatch.setattr(mapper, "map_claims", fake_map)
    monkeypatch.setenv("MAP_BACKEND", "openai")
    monkeypatch.setenv("EXTRACT_BASE_URL", "http://router/v1")
    monkeypatch.setenv("EXTRACT_API_KEY", "shared-key")
    assert cli.main(["taxonomy", "map"]) == 0
    assert stubbed["clients"] == [("openai", "http://router/v1", "shared-key")]


@pytest.mark.parametrize("extract, map_, code", [
    ({"errored": 0}, {"llm_invalid": 0}, 0),
    ({"errored": 2}, {"llm_invalid": 0}, 1),
    ({"errored": 0}, {"llm_invalid": 1}, 1),
    (None, None, 0),  # nothing to do
])
def test_refresh_exit_code(stubbed, monkeypatch, capsys, extract, map_, code):
    async def fake_refresh(conn, **kw):
        return refresh.RefreshReport(extract=extract, map=map_)

    monkeypatch.setattr(refresh, "refresh", fake_refresh)
    assert cli.main(["refresh"]) == code
