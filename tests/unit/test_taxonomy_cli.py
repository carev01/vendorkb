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


@pytest.mark.parametrize("kind", ["dimension", "target", "object_type", "offering"])
def test_taxonomy_propose_kinds(stubbed, monkeypatch, capsys, kind):
    async def fake_propose(conn, **kw):
        stubbed.update(kw)
        return propose.ProposeReport(kind=kw["kind"])

    monkeypatch.setattr(propose, "propose", fake_propose)
    assert cli.main(["taxonomy", "propose", "--kind", kind]) == 0
    assert stubbed["kind"] == kind


def test_taxonomy_propose_refuses_v1_kinds(stubbed):
    with pytest.raises(SystemExit):
        cli.main(["taxonomy", "propose", "--kind", "feature"])


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
    assert cli.main(["taxonomy", "propose", "--kind", "dimension"]) == 0
    assert (stubbed["distance_threshold"], stubbed["out_path"], stubbed["max_usd"],
            stubbed["limit"]) == (0.25, "taxonomy/taxonomy.yaml", 2.0, None)


def test_taxonomy_load_and_map(stubbed, monkeypatch, capsys):
    async def fake_load(conn, path):
        return {"version": "2-ab12cd34", "path": path}

    async def fake_map(conn, *, embedder, client, settings, max_usd, remap, vendor):
        stubbed.update(map_backend=settings.map_backend,
                       extract_backend=settings.extract_backend, remap=remap, vendor=vendor)
        return mapper.MapReport(taxonomy_version="1", claims=int(max_usd * 100))

    monkeypatch.setattr(load, "load_taxonomy", fake_load)
    monkeypatch.setattr(mapper, "map_claims", fake_map)
    assert cli.main(["taxonomy", "load", "t.yaml"]) == 0
    assert json.loads(capsys.readouterr().out) == {"version": "2-ab12cd34", "path": "t.yaml"}
    assert cli.main(["taxonomy", "map", "--max-usd", "0.7", "--backend", "openai"]) == 0
    assert json.loads(capsys.readouterr().out)["claims"] == 70
    assert (stubbed["map_backend"], stubbed["extract_backend"]) == ("openai", "anthropic")
    assert cli.main(["taxonomy", "map"]) == 0
    assert json.loads(capsys.readouterr().out)["claims"] == 100  # default --max-usd 1.0
    assert stubbed["remap"] is False  # only claims not yet mapped under this version
    assert stubbed["init_schema"]
    assert cli.main(["taxonomy", "map", "--all"]) == 0
    assert stubbed["remap"] is True and stubbed["vendor"] is None
    assert cli.main(["taxonomy", "map", "--vendor", "Veeam"]) == 0
    assert stubbed["vendor"] == "Veeam"


def test_taxonomy_map_exit_code_on_invalid_replies(stubbed, monkeypatch, capsys):
    async def fake_map(conn, **kw):
        return mapper.MapReport(llm_invalid=2)

    monkeypatch.setattr(mapper, "map_claims", fake_map)
    assert cli.main(["taxonomy", "map"]) == 1


def test_taxonomy_map_client_uses_the_map_backend(stubbed, monkeypatch, capsys):
    built = []
    monkeypatch.setattr(cli, "_extract_client", lambda s: built.append(s.extract_backend))

    async def fake_map(conn, **kw):
        return mapper.MapReport()

    monkeypatch.setattr(mapper, "map_claims", fake_map)
    monkeypatch.setenv("MAP_BACKEND", "openai")
    assert cli.main(["taxonomy", "map"]) == 0
    assert built == ["openai"]


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

    async def fake_query(conn, f):
        stubbed["filter"] = f
        fact = capabilities.Fact(
            "c1", "veeam", "VBR", None, None, "coverage.backup", "coverage", "cloud.database",
            "aws.rds", None, "database", "capability", "supported", "unstated", None, None,
            None, None, None, "product", None, True, "backs up", "Amazon RDS", "q", "k1",
            "http://u", "T", "2026-01-01T00:00:00+00:00", "2-abc")
        return capabilities.CapResult(facts=[fact], hint=None)

    monkeypatch.setattr(capabilities, "build", fake_build)
    monkeypatch.setattr(load, "current_version", fake_version)
    monkeypatch.setattr(capabilities, "query", fake_query)
    assert cli.main(["capabilities", "build"]) == 0
    assert json.loads(capsys.readouterr().out) == {"rows": 7, "taxonomy_version": "3"}
    assert cli.main(["capabilities", "query", "--target", "aws", "--vendor", "veeam",
                     "--support", "partial", "--offering", "o", "--edition", "e",
                     "--dimension", "d", "--family", "coverage", "--object-type", "database",
                     "--availability", "preview", "--value-min", "1",
                     "--value-max", "9.5"]) == 0
    assert stubbed["filter"] == capabilities.CapFilter(
        vendor="veeam", offering="o", edition="e", dimension="d", family="coverage",
        target="aws", object_type="database", support="partial", availability="preview",
        value_min=1.0, value_max=9.5)
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"facts", "product_level", "product_wide", "conflicts", "hint", "totals"}
    fact = out["facts"][0]
    assert (fact["vendor"], fact["url"], fact["evidence_quote"]) == ("veeam", "http://u", "q")


def test_capabilities_query_choices_match_the_sources():
    from vendorkb.claims.extract import AVAILABILITY_VALUES, SUPPORT_VALUES

    assert cli.SUPPORT_VALUES == SUPPORT_VALUES
    assert cli.AVAILABILITY_VALUES == AVAILABILITY_VALUES
    assert cli.FAMILIES == load.FAMILIES


def test_taxonomy_aspects_json_lines_and_unknown_exit(stubbed, monkeypatch, capsys):
    from vendorkb.taxonomy import aspects as aspects_mod

    async def fake_aspects(conn, *, target=None, cls=None):
        stubbed["args"] = (target, cls)
        if target == "nope":
            raise LookupError("unknown target 'nope'")
        return [aspects_mod.Aspect("coverage", "coverage.backup", "coverage", "Backup", "d",
                                   "user", None)]

    monkeypatch.setattr(aspects_mod, "aspects", fake_aspects)
    assert cli.main(["taxonomy", "aspects", "--target", "okta"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert (row["kind"], row["object_type_id"], row["value_unit"]) == ("coverage", "user", None)
    assert cli.main(["taxonomy", "aspects", "--class", "saas.identity"]) == 0
    assert stubbed["args"] == (None, "saas.identity")
    capsys.readouterr()
    assert cli.main(["taxonomy", "aspects", "--target", "nope"]) == 2
    out = capsys.readouterr()
    assert out.out == "" and "unknown target" in out.err
    with pytest.raises(SystemExit) as exc:  # neither or both of --target/--class
        cli.main(["taxonomy", "aspects"])
    assert exc.value.code == 2
