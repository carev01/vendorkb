import json
import logging

import pytest

from vendorkb import cli
from vendorkb.sync import SyncReport


class FakeConn:
    async def close(self):
        pass


@pytest.fixture
def fake_sync(monkeypatch):
    """Stub the database and the sync engine; the test sets the report to return."""
    calls = {}

    async def connect(dsn):
        return FakeConn()

    async def init_schema(conn, *, embed_dim):
        pass

    async def sync(conn, *, docext, embedder, page_size, max_tokens):
        calls["max_tokens"] = max_tokens
        return calls["report"]

    monkeypatch.setattr(cli, "connect", connect)
    monkeypatch.setattr(cli, "init_schema", init_schema)
    monkeypatch.setattr(cli, "sync", sync)
    return calls


@pytest.mark.parametrize(("flags", "code"), [
    ({"complete": True}, 0),
    ({"stopped_by_budget": True}, 0),
    ({"locked": True}, 0),
    ({}, 1),
])
def test_sync_exit_code(fake_sync, capsys, flags, code):
    fake_sync["report"] = SyncReport(mode="bootstrap", **flags)
    assert cli.main(["sync", "--max-tokens", "500"]) == code
    assert fake_sync["max_tokens"] == 500
    report = json.loads(capsys.readouterr().out)
    assert {k: report[k] for k in flags} == flags


def test_main_configures_info_logging(fake_sync, monkeypatch):
    seen = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: seen.update(kw))
    fake_sync["report"] = SyncReport(mode="incremental", complete=True)
    assert cli.main(["sync"]) == 0
    assert seen["level"] == logging.INFO and "format" in seen
