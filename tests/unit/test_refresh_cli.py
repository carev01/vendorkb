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
                     "--map-max-usd", "0.5", "--warmup-seconds", "60",
                     "--support-max-usd", "0.25", "--support-review-max-usd", "0.75"]) == 0
    assert json.loads(capsys.readouterr().out)["new_chunks"] == 3
    # the support client defaults to the Anthropic API
    assert stubbed["clients"] == [("openai", "http://serve/v1", "serve-key"),
                                  ("openai", "http://router/v1", "map-key"),
                                  ("anthropic", "http://router/v1", "map-key")]
    assert waited == [("http://serve/v1", "serve-key", 60.0)]
    assert (stubbed["settle_minutes"], stubbed["extract_limit"],
            stubbed["map_max_usd"]) == (30.0, 500, 0.5)
    assert (stubbed["support_max_usd"], stubbed["support_review_max_usd"]) == (0.25, 0.75)


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


@pytest.mark.parametrize("step", ["check", "review"])
def test_refresh_fails_on_an_unusable_support_call(stubbed, monkeypatch, capsys, step):
    async def fake_refresh(conn, **kw):
        sup = {"check": {"invalid": 0}, "review": {"invalid": 0}}
        sup[step]["invalid"] = 1
        return refresh.RefreshReport(extract={"errored": 0}, map={"llm_invalid": 0},
                                     support=sup)

    monkeypatch.setattr(refresh, "refresh", fake_refresh)
    assert cli.main(["refresh"]) == 1


def test_refresh_fails_when_the_support_step_raised(stubbed, monkeypatch, capsys):
    async def fake_refresh(conn, **kw):
        return refresh.RefreshReport(extract={"errored": 0}, map={"llm_invalid": 0},
                                     support={"error": "RuntimeError: boom"})

    monkeypatch.setattr(refresh, "refresh", fake_refresh)
    assert cli.main(["refresh"]) == 1


def test_the_support_endpoint_falls_back_as_a_url_and_key_pair(stubbed, monkeypatch):
    monkeypatch.setenv("SUPPORT_BACKEND", "openai")
    monkeypatch.setenv("MAP_BASE_URL", "http://router/v1")
    monkeypatch.setenv("MAP_API_KEY", "map-key")
    monkeypatch.setenv("EXTRACT_API_KEY", "gpu-key")
    cli._support_client(cli.VendorkbSettings(_env_file=None))
    monkeypatch.setenv("SUPPORT_BASE_URL", "http://other/v1")  # its key forgotten
    cli._support_client(cli.VendorkbSettings(_env_file=None))
    assert stubbed["clients"] == [("openai", "http://router/v1", "map-key"),
                                  ("openai", "http://other/v1", "")]


def test_claims_support_wires_its_endpoint_and_the_budgets(stubbed, monkeypatch, capsys):
    from vendorkb.claims import support

    async def fake_run(conn, *, client, settings, check_max_usd, review_max_usd, limit):
        stubbed.update(check=check_max_usd, review=review_max_usd, limit=limit)
        report = support.SupportReport(taxonomy_version="2-x")
        report.check.invalid = 1 if limit == 1 else 0
        return report

    monkeypatch.setattr(support, "run", fake_run)
    monkeypatch.setenv("SUPPORT_BACKEND", "openai")  # its endpoint falls back to the MAP_ one
    monkeypatch.setenv("MAP_BASE_URL", "http://router/v1")
    monkeypatch.setenv("MAP_API_KEY", "map-key")
    assert cli.main(["claims", "support", "--check-max-usd", "4", "--review-max-usd", "2",
                     "--limit", "20"]) == 0
    assert json.loads(capsys.readouterr().out)["taxonomy_version"] == "2-x"
    assert stubbed["clients"] == [("openai", "http://router/v1", "map-key")]
    assert (stubbed["check"], stubbed["review"], stubbed["limit"]) == (4.0, 2.0, 20)
    assert cli.main(["claims", "support", "--limit", "1"]) == 1  # an unusable call
