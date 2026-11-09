"""``eval/map_replay.py --taxonomy`` against a real Postgres, with a fake embedder: an edited
taxonomy YAML decides the dimensions the replay offers (names, definitions, applies_to, search
aliases, new and removed dimensions), ranks them as the mapper does once that YAML is loaded,
and writes nothing to the taxonomy tables.

Every phrase here is invented for the tests; none comes from vendor documentation."""
import yaml

from eval import map_replay as replay
from tests.integration.test_taxonomy_mapper_v2 import (
    VECTORS, FakeEmbedder, FakeOpenAI, _axis, _candidates, _claim, _extended, _fresh, _phrase,
    _run, _settings)
from tests.unit.test_map_replay import NEW_DEFINITION, _write_yaml, edited_doc
from vendorkb.db import connect
from vendorkb.taxonomy import load, mapper

TABLE = VECTORS | {
    "minimum retention period": _axis(3), "shortest retention": _axis(3, 0.05, 13),
    "keep copies briefly": _axis(3, 0.1, 14),
    "crm restore rehearsal": _axis(4), "rehearse crm restore": _axis(4, 0.05, 15),
    "try a crm restore without writing": _axis(4, 0.1, 16),
}
TARGETS = {"saas.identity": "Okta", "saas.crm": "Salesforce", "(none)": None}


def _item(item_id: str, phrase: str, class_id: str) -> dict:
    return {"item_id": item_id, "claim_id": f"c-{item_id}", "phrase": phrase,
            "class_id": class_id, "vendor": "examplevendor", "target_text": "t",
            "evidence_quote": "q", "acceptable": [None]}


ITEMS = [
    _item("i1", "keep copies briefly", "saas.identity"),
    _item("i2", "try a crm restore without writing", "saas.crm"),
    _item("i3", "try a crm restore without writing", "saas.identity"),
    _item("i4", "keep copies for ages", "(none)"),
]


class _Ctx:
    """What ``Embedder(settings)`` is in ``_setup``: an async context manager."""

    def __init__(self, embedder):
        self.embedder = embedder

    async def __aenter__(self):
        return self.embedder

    async def __aexit__(self, *exc):
        return None


async def _stored(conn) -> tuple:
    """The taxonomy tables a replay must leave alone, and the taxonomy version."""
    cur = await conn.execute(
        "SELECT dimension_id, name, definition, applies_to FROM dimensions ORDER BY 1")
    dimensions = await cur.fetchall()
    cur = await conn.execute(
        "SELECT alias, entity_type, scope, entity_id, search FROM aliases ORDER BY 1, 2, 3")
    return dimensions, await cur.fetchall(), await load.current_version(conn)


async def test_override_offers_the_edited_dimensions_and_stores_nothing(
        vkb_dsn, tmp_path, monkeypatch):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        edited = load.validate(edited_doc(yaml.safe_load(_extended(tmp_path).read_text())))
        embedder = FakeEmbedder(TABLE)
        monkeypatch.setattr("vendorkb.clients.Embedder", lambda settings: _Ctx(embedder))
        settings = _settings()
        stored = await _stored(conn)

        base, _, base_ranked, _ = await replay._setup(conn, settings, ITEMS)
        assert "max retention" in base.search[("dimension", "")]
        assert mapper._details(base, "dimension", "retention.maximum_period")[1] == (
            "The longest time a backup copy can be kept.")
        assert "retention.minimum_period" not in base.dimensions
        assert "retention.minimum_period" not in base_ranked["i1"][2]
        assert len(embedder.calls) == 1
        calls_before = len(embedder.calls)

        tax, _, ranked, rank = await replay._setup(conn, settings, ITEMS, edited)

        # a changed definition is the one the candidate details show
        assert mapper._details(tax, "dimension", "retention.maximum_period") == (
            "Maximum retention period", NEW_DEFINITION)
        # a new dimension can be a candidate, one limited to CRM only for a CRM class
        assert ranked["i1"][2][0] == "retention.minimum_period"
        assert ranked["i2"][2][0] == "recovery.crm_dry_run"
        assert "recovery.crm_dry_run" not in ranked["i3"][2]
        assert "recovery.crm_dry_run" in ranked["i4"][2]  # no class: every dimension
        assert ranked["i4"][2][0] == "retention.maximum_period"
        # a lookup_only dimension is known to the taxonomy but never offered
        assert "operations.manual_only" in tax.dimensions
        assert all("operations.manual_only" not in r[2] for r in ranked.values())
        # a removed alias is no longer a key, and the keyword retriever reads the new text
        assert "max retention" not in tax.search[("dimension", "")]
        assert rank("shortest", ranked["i1"][1], set(), 5) == ["retention.minimum_period"]
        # only the new keys were embedded (the rest came from the cache); a removed alias, a
        # lookup alias and a lookup_only dimension's words were not
        assert len(embedder.calls) == calls_before + 1
        assert set(embedder.calls[-1][0]) == {
            "minimum retention period", "shortest retention", "crm restore rehearsal",
            "rehearse crm restore"}

        assert await _stored(conn) == stored  # the taxonomy tables are as they were
    finally:
        await conn.close()


def _candidate_lines(user: str) -> list[str]:
    block = user.split("Candidates (number. id | name | details):\n", 1)[1]
    return block.split("\n\n", 1)[0].splitlines()


async def test_override_candidates_are_those_the_mapper_offers_once_the_yaml_is_loaded(
        vkb_dsn, tmp_path, monkeypatch):
    conn = await connect(vkb_dsn)
    try:
        await _fresh(conn, tmp_path)
        path = _write_yaml(tmp_path / "edited.yaml",
                           edited_doc(yaml.safe_load(_extended(tmp_path).read_text())))
        monkeypatch.setattr("vendorkb.clients.Embedder",
                            lambda settings: _Ctx(FakeEmbedder(TABLE)))
        settings = _settings()
        tax, _, ranked, _ = await replay._setup(conn, settings, ITEMS, load.validate(
            load.read_doc(path)))

        await load.load_taxonomy(conn, path)  # now the database holds the edited YAML
        loaded = await mapper._snapshot(conn)
        # the override is what the loader stores for the dimensions
        assert tax.dimensions == loaded.dimensions
        for part in ("aliases", "search"):
            assert getattr(tax, part)[("dimension", "")] == getattr(loaded, part)[("dimension", "")]

        for it in ITEMS:
            await _claim(conn, it["phrase"], TARGETS[it["class_id"]])
        client = FakeOpenAI({it["phrase"]: (lambda candidates: "none") for it in ITEMS})
        await _run(conn, client, FakeEmbedder(TABLE))
        assert len(client.calls) == len(ITEMS)

        by_question = {}
        for call in client.calls:
            user = call["messages"][-1]["content"]
            cls = user.split("Workload class: ", 1)[1].split("\n", 1)[0]
            by_question[(_phrase(user), cls)] = user
        for it in ITEMS:
            user = by_question[(it["phrase"], it["class_id"])]
            offered = ranked[it["item_id"]][2][:settings.map_candidates]  # the replay's k
            assert len(offered) == settings.map_candidates
            assert _candidates(user) == offered
            assert _candidate_lines(user) == [
                f"{n}. {eid} | {' | '.join(mapper._details(tax, 'dimension', eid))}"
                for n, eid in enumerate(offered, 1)]
    finally:
        await conn.close()
