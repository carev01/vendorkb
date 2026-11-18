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

    async def sync(conn, *, docext, embedder, page_size, max_tokens, allow_prune):
        calls["max_tokens"] = max_tokens
        calls["allow_prune"] = allow_prune
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
    ({"prune_refused": 600}, 1),
])
def test_sync_exit_code(fake_sync, capsys, flags, code):
    fake_sync["report"] = SyncReport(mode="bootstrap", **flags)
    assert cli.main(["sync", "--max-tokens", "500"]) == code
    assert fake_sync["max_tokens"] == 500
    assert fake_sync["allow_prune"] is None
    report = json.loads(capsys.readouterr().out)
    assert {k: report[k] for k in flags} == flags


def test_main_configures_info_logging(fake_sync, monkeypatch):
    seen = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: seen.update(kw))
    fake_sync["report"] = SyncReport(mode="incremental", complete=True)
    assert cli.main(["sync"]) == 0
    assert seen["level"] == logging.INFO and "format" in seen
    for name in ("httpx", "httpcore"):  # one INFO line per request otherwise
        assert logging.getLogger(name).level == logging.WARNING
        logging.getLogger(name).setLevel(logging.NOTSET)


def test_sync_passes_allow_prune(fake_sync):
    fake_sync["report"] = SyncReport(mode="bootstrap", complete=True)
    assert cli.main(["sync", "--allow-prune", "600"]) == 0
    assert fake_sync["allow_prune"] == 600


def test_extract_passes_retry_errored(monkeypatch, tmp_path):
    from vendorkb.claims import extract as extract_mod
    from vendorkb.claims.extract import ExtractReport
    seen = {}

    async def connect(dsn):
        return FakeConn()

    async def init_schema(conn, *, embed_dim):
        pass

    async def run(conn, ids, *, client, settings, mode, retry_truncated=False):
        seen.update(ids=ids, mode=mode, retry=retry_truncated)
        return ExtractReport(chunks=len(ids))

    monkeypatch.setattr(cli, "connect", connect)
    monkeypatch.setattr(cli, "init_schema", init_schema)
    monkeypatch.setattr(cli, "_extract_client", lambda settings: object())
    monkeypatch.setattr(extract_mod, "run", run)
    ids = tmp_path / "ids.txt"
    ids.write_text("a\nb\n")
    assert cli.main(["extract", "--ids-file", str(ids), "--mode", "sync",
                     "--retry-errored"]) == 0
    assert seen == {"ids": ["a", "b"], "mode": "sync", "retry": True}
    assert cli.main(["extract", "--ids-file", str(ids), "--mode", "sync"]) == 0
    assert seen["retry"] is False


def test_sigterm_cancels_the_command_and_fails_its_run(monkeypatch):
    """A Job deadline sends SIGTERM: the command is cancelled, so ``track`` ends its row
    ``failed`` at once, and the process exits 143."""
    import asyncio
    import os
    import signal

    from vendorkb import task_runs

    finished = []

    async def start(dsn, kind, stage, parent_id):
        return task_runs.TaskRun()

    async def finish(self, state, details=None):
        finished.append(state)

    async def command(settings, args):
        async with task_runs.track("unused", "sync"):
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.sleep(10)
        return 0

    monkeypatch.setattr(task_runs, "_start", start)
    monkeypatch.setattr(task_runs.TaskRun, "finish", finish)
    monkeypatch.setitem(cli._COMMANDS, "stats", command)
    assert cli.main(["stats"]) == 143
    assert finished == ["failed"]
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
