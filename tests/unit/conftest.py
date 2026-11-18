"""Unit tests never record task runs: the CLI's DSN would reach a real database, if any."""
import pytest

from vendorkb import task_runs


@pytest.fixture(autouse=True)
def _no_task_runs(monkeypatch):
    async def start(dsn, kind, stage, parent_id):
        return task_runs.TaskRun()

    monkeypatch.setattr(task_runs, "_start", start)
