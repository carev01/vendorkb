import json

import pytest

from vendorkb import cli
from vendorkb.taxonomy import capabilities, load, mapper, propose


class FakeConn:
    async def close(self):
        pass


@pytest.fixture
def stubbed(monkeypatch):
    calls: dict = {}

    async def connect(dsn):
        return FakeConn()

    async def init_schema(conn, *, embed_dim):
        calls["init_schema"] = True

    monkeypatch.setattr(cli, "connect", connect)
    monkeypatch.setattr(cli, "init_schema", init_schema)
    monkeypatch.setattr(cli, "_extract_client", lambda settings: object())
    return calls


@pytest.mark.parametrize(("invalid", "budget", "code"), [(0, False, 0), (0, True, 0),
                                                          (2, False, 1)])
def test_taxonomy_propose_args_and_exit(stubbed, monkeypatch, capsys, invalid, budget, code):
    async def fake_propose(conn, *, client, embedder, settings, kind, distance_threshold,
                           out_path, max_usd, limit):
        stubbed.update(kind=kind, threshold=distance_threshold, out=out_path, max_usd=max_usd,
                       backend=settings.extract_backend, model=settings.extract_model,
                       limit=limit)
        return propose.ProposeReport(kind=kind, invalid=invalid, stopped_by_budget=budget)

    monkeypatch.setattr(propose, "propose", fake_propose)
    assert cli.main(["taxonomy", "propose", "--kind", "target", "--threshold", "0.3",
                     "--out", "x.yaml", "--max-usd", "1.5", "--backend", "openai",
                     "--limit", "3"]) == code
    assert (stubbed["kind"], stubbed["threshold"], stubbed["out"], stubbed["max_usd"],
            stubbed["backend"], stubbed["limit"]) == ("target", 0.3, "x.yaml", 1.5, "openai", 3)
    assert json.loads(capsys.readouterr().out)["kind"] == "target"


def test_taxonomy_propose_defaults(stubbed, monkeypatch, capsys):
    async def fake_propose(conn, **kw):
        stubbed.update(kw)
        return propose.ProposeReport()

    monkeypatch.setattr(propose, "propose", fake_propose)
    assert cli.main(["taxonomy", "propose", "--kind", "feature"]) == 0
    assert (stubbed["distance_threshold"], stubbed["out_path"], stubbed["max_usd"],
            stubbed["limit"]) == (0.25, "taxonomy/taxonomy.yaml", 2.0, None)


def test_taxonomy_load_and_map(stubbed, monkeypatch, capsys):
    async def fake_load(conn, path):
        return {"version": "2-ab12cd34", "path": path}

    async def fake_map(conn, *, embedder, threshold):
        return mapper.MapReport(taxonomy_version="1", claims=int(threshold * 100))

    monkeypatch.setattr(load, "load_taxonomy", fake_load)
    monkeypatch.setattr(mapper, "map_claims", fake_map)
    assert cli.main(["taxonomy", "load", "t.yaml"]) == 0
    assert json.loads(capsys.readouterr().out) == {"version": "2-ab12cd34", "path": "t.yaml"}
    assert cli.main(["taxonomy", "map", "--threshold", "0.7"]) == 0
    assert json.loads(capsys.readouterr().out)["claims"] == 70
    assert stubbed["init_schema"]


def test_taxonomy_load_lists_every_problem(stubbed, monkeypatch, capsys):
    async def fake_load(conn, path):
        raise load.TaxonomyError(["first problem", "second problem"])

    monkeypatch.setattr(load, "load_taxonomy", fake_load)
    assert cli.main(["taxonomy", "load", "t.yaml"]) == 1
    out = capsys.readouterr()
    assert out.out == "" and "first problem" in out.err and "second problem" in out.err


def test_capabilities_build_and_query(stubbed, monkeypatch, capsys):
    async def fake_build(conn):
        return 7

    async def fake_version(conn):
        return "3"

    async def fake_query(conn, feature, target=None, vendor=None, support=None):
        stubbed["query"] = (feature, target, vendor, support)
        return [capabilities.CapRow("veeam", "VBR", None, feature, target, "Amazon RDS",
                                    "supported", None, "q", "c1", "http://u", "T", 1.0,
                                    "alias")]

    monkeypatch.setattr(capabilities, "build", fake_build)
    monkeypatch.setattr(load, "current_version", fake_version)
    monkeypatch.setattr(capabilities, "query", fake_query)
    assert cli.main(["capabilities", "build"]) == 0
    assert json.loads(capsys.readouterr().out) == {"rows": 7, "taxonomy_version": "3"}
    assert cli.main(["capabilities", "query", "immutable_backups", "--target", "aws",
                     "--vendor", "veeam", "--support", "partial"]) == 0
    assert stubbed["query"] == ("immutable_backups", "aws", "veeam", "partial")
    row = json.loads(capsys.readouterr().out)
    assert (row["vendor"], row["url"], row["chunk_id"]) == ("veeam", "http://u", "c1")
