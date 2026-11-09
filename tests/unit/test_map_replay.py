import copy
import json
from pathlib import Path

import pytest
import yaml

from eval import map_replay as replay
from eval.map_replay import bm25_ranker, score, words
from vendorkb.config import VendorkbSettings
from vendorkb.taxonomy import load, mapper

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"
NEW_DEFINITION = "The longest span a backup copy is kept before it expires."


def test_words_drop_stopwords_short_words_and_plurals():
    assert words("The backups of SQL databases, by a DB") == ["backup", "sql", "database"]


def test_bm25_ranks_by_shared_words_within_allowed_minus_excluded():
    rank = bm25_ranker({"a": "immutable backup copies", "b": "backup schedule frequency",
                        "c": "immutable object lock retention", "d": "nothing shared"})
    # b and c share one word each with the query; the shorter document ranks first
    assert rank("immutable backups", {"a", "b", "c", "d"}, set(), 5) == ["a", "b", "c"]
    assert rank("immutable backups", {"a", "b", "c", "d"}, {"a"}, 5) == ["b", "c"]
    assert rank("immutable backups", {"b", "d"}, set(), 5) == ["b"]  # zero scores left out


def test_score_precision_recall_and_unlabelled_answers():
    items = [
        {"item_id": "1", "acceptable": ["x"], "assigned_2026_11_08": "x"},
        {"item_id": "2", "acceptable": [None], "assigned_2026_11_08": None},
        {"item_id": "3", "acceptable": ["y", None], "assigned_2026_11_08": "z"},
        {"item_id": "4", "acceptable": ["w"], "assigned_2026_11_08": None},
    ]
    s = score({"1": "x", "2": "q", "3": None, "4": None}, items)
    # answered 1 and 2: 1 is right, 2 should be none (and is a dimension the labels never saw)
    assert (s["answered"], s["precision"], s["unlabelled"]) == (2, 0.5, 1)
    # items with an acceptable dimension: 1, 3 (none also fine there) and 4
    assert (s["recall"], s["right"]) == (2 / 3, 2)


# ---- --labels and --taxonomy ----------------------------------------------------------------

def edited_doc(doc: dict) -> dict:
    """``doc`` (a taxonomy with the fixture's dimensions) with one definition changed, one
    search alias removed (and a lookup alias added), and three dimensions added: one for every
    class, one for CRM applications only, one that is never a candidate."""
    doc = copy.deepcopy(doc)
    dims = {d["id"]: d for d in doc["dimensions"]}
    dims["retention.maximum_period"].update(definition=NEW_DEFINITION,
                                            aliases=["longest retention"])  # "max retention" gone
    dims["schedule.backup_frequency"]["lookup_aliases"] = ["runs every day"]
    doc["dimensions"] += [
        {"id": "retention.minimum_period", "name": "Minimum retention period",
         "definition": "The shortest time a backup copy must be kept.",
         "aliases": ["shortest retention"], "applies_to": "all"},
        {"id": "recovery.crm_dry_run", "name": "CRM restore rehearsal",
         "definition": "Whether a restore of CRM records can be tried without writing them.",
         "aliases": ["rehearse crm restore"], "applies_to": ["saas.crm"]},
        {"id": "operations.manual_only", "name": "Manual only",
         "definition": "Work that only a person can do.", "aliases": ["manual step"],
         "applies_to": "all", "lookup_only": True},
    ]
    return doc


def _write_yaml(path: Path, doc: dict) -> Path:
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _tax() -> mapper._Tax:
    """A snapshot with one dimension (and an alias that is gone in the edited YAML)."""
    return mapper._Tax(
        aliases={("dimension", ""): {"old alias": "retention.maximum_period"},
                 ("target", ""): {"okta": "okta"}},
        search={("dimension", ""): {"old alias": "retention.maximum_period"},
                ("target", ""): {"okta": "okta"}},
        target_parent={"okta": None}, target_class={"okta": "saas.identity"},
        target_name={"okta": "Okta"},
        class_parent={"saas": None, "saas.identity": "saas", "saas.crm": "saas"},
        dimensions={"retention.maximum_period": ("Old name", "old definition", None)},
        vendor_offerings={"examplevendor": ["examplevendor.cloud_suite"]})


def test_with_dimensions_replaces_the_dimensions_and_keeps_the_rest():
    taxonomy = load.validate(edited_doc(yaml.safe_load(FIXTURE.read_text())))
    tax = _tax()
    before = copy.deepcopy(tax)
    out = mapper._with_dimensions(tax, taxonomy)
    assert tax == before  # the snapshot it was given is left as it was
    assert set(out.dimensions) == {d.id for d in taxonomy.dimensions}
    assert out.dimensions["retention.maximum_period"] == (
        "Maximum retention period", NEW_DEFINITION, None)
    assert out.dimensions["recovery.crm_dry_run"][2] == ["saas.crm"]
    assert mapper._details(out, "dimension", "retention.maximum_period")[1] == NEW_DEFINITION
    # the targets, classes and offerings stay
    assert (out.aliases[("target", "")], out.search[("target", "")]) == (
        tax.aliases[("target", "")], tax.search[("target", "")])
    assert (out.target_class, out.class_parent, out.vendor_offerings) == (
        tax.target_class, tax.class_parent, tax.vendor_offerings)


def test_with_dimensions_search_keys_are_names_and_search_aliases_only():
    taxonomy = load.validate(edited_doc(yaml.safe_load(FIXTURE.read_text())))
    out = mapper._with_dimensions(_tax(), taxonomy)
    keys = out.search[("dimension", "")]
    assert keys["longest retention"] == "retention.maximum_period"
    assert keys["maximum retention period"] == "retention.maximum_period"  # its own name
    assert keys["shortest retention"] == "retention.minimum_period"  # a new dimension
    assert "max retention" not in keys and "old alias" not in keys  # removed aliases
    # a lookup alias and a lookup_only dimension are found by exact lookup, never searched
    assert "runs every day" not in keys and "manual step" not in keys
    assert "manual only" not in keys
    assert out.lookup("dimension", "", "runs every day") == "schedule.backup_frequency"
    assert out.lookup("dimension", "", "manual step") == "operations.manual_only"
    assert out.lookup("dimension", "", "max retention") is None
    assert out.lookup("dimension", "", "old alias") is None


def test_with_dimensions_applies_to_filters_by_class_lineage():
    taxonomy = load.validate(edited_doc(yaml.safe_load(FIXTURE.read_text())))
    out = mapper._with_dimensions(_tax(), taxonomy)
    assert "recovery.crm_dry_run" in out.applicable("saas.crm")
    assert "recovery.crm_dry_run" not in out.applicable("saas.identity")
    assert "retention.minimum_period" in out.applicable("saas.identity")
    assert "recovery.crm_dry_run" in out.applicable(None)  # no target: every dimension


def _labels(path: Path, items: list[dict]) -> Path:
    path.write_text(json.dumps({"taxonomy_version": "x", "items": items}))
    return path


def _item(item_id: str, acceptable: list, **extra) -> dict:
    return {"item_id": item_id, "claim_id": f"c-{item_id}", "phrase": f"phrase {item_id}",
            "class_id": "(none)", "vendor": "v", "target_text": "t", "evidence_quote": "q",
            "acceptable": acceptable, **extra}


def test_score_reads_the_given_labels(tmp_path, capsys):
    labels = _labels(tmp_path / "mine.json", [_item("x1", ["a.b"], note="ignored"),
                                              _item("x2", [None])])
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"k5": {"x1": "a.b", "x2": "c.d"}}))
    assert replay.main(["score", str(run), "--labels", str(labels)]) == 0
    rows = capsys.readouterr().out.splitlines()
    # no mapped-2026-11-08 row: these items carry no assigned answer
    assert [r.split()[0] for r in rows] == ["variant", "k5"]
    assert rows[1].split() == ["k5", "0.500", "of", "2", "1.000", "1", "1"]


def test_score_without_labels_option_uses_the_default_labels(tmp_path, capsys):
    run = tmp_path / "run.json"
    run.write_text("{}")
    assert replay.main(["score", str(run)]) == 0
    rows = capsys.readouterr().out.splitlines()
    assert [r.split()[0] for r in rows] == ["variant", "mapped"]  # "mapped 2026-11-08"


class _Stop(Exception):
    pass


@pytest.fixture
def stubbed_io(monkeypatch):
    """No database and no .env: ``connect`` and the settings are fakes, and ``_setup`` records
    what it was given (and stops a ``run`` there when ``seen["stop"]`` is set)."""
    seen: dict = {}
    real = VendorkbSettings

    class Conn:
        async def execute(self, *args):
            pass

        async def close(self):
            pass

    async def connect(dsn):
        seen["connected"] = True
        return Conn()

    async def setup(conn, settings, items, taxonomy=None):
        seen["items"], seen["taxonomy"] = items, taxonomy
        if seen.get("stop"):
            raise _Stop
        return None, {}, {it["item_id"]: (None, set(), []) for it in items}, (lambda *a: [])

    monkeypatch.setattr("vendorkb.config.VendorkbSettings", lambda: real(_env_file=None))
    monkeypatch.setattr("vendorkb.db.connect", connect)
    monkeypatch.setattr(replay, "_setup", setup)
    return seen


def test_recall_and_run_take_the_given_labels_and_taxonomy(stubbed_io, tmp_path, capsys):
    labels = _labels(tmp_path / "mine.json", [_item("x1", ["a.b"]), _item("x2", [None])])
    taxonomy = _write_yaml(tmp_path / "edited.yaml",
                           edited_doc(yaml.safe_load(FIXTURE.read_text())))
    options = ["--labels", str(labels), "--taxonomy", str(taxonomy)]

    assert replay.main(["recall", *options]) == 0
    assert [it["item_id"] for it in stubbed_io["items"]] == ["x1", "x2"]
    assert "retention.minimum_period" in {d.id for d in stubbed_io["taxonomy"].dimensions}
    assert "acceptable dimension among the top  5: 0/1" in capsys.readouterr().out

    stubbed_io.update(stop=True, items=None, taxonomy=None)
    with pytest.raises(_Stop):
        replay.main(["run", str(tmp_path / "out.json"), "k5", "--max-usd", "0.1", *options])
    assert [it["item_id"] for it in stubbed_io["items"]] == ["x1", "x2"]
    assert stubbed_io["taxonomy"] is not None
    assert capsys.readouterr().out.splitlines()[0] == f"taxonomy: {taxonomy}"


def test_without_options_run_and_recall_use_the_database_and_the_default_labels(
        stubbed_io, tmp_path, capsys):
    default_ids = [it["item_id"] for it in json.loads(replay.LABELS.read_text())["items"]]

    assert replay.main(["recall"]) == 0
    assert stubbed_io["taxonomy"] is None
    assert [it["item_id"] for it in stubbed_io["items"]] == default_ids
    capsys.readouterr()

    stubbed_io.update(stop=True, items=None)
    with pytest.raises(_Stop):
        replay.main(["run", str(tmp_path / "out.json"), "k5", "--max-usd", "0.1"])
    assert stubbed_io["taxonomy"] is None
    assert [it["item_id"] for it in stubbed_io["items"]] == default_ids
    assert capsys.readouterr().out.splitlines()[0] == "taxonomy: database"


def test_an_invalid_taxonomy_stops_before_anything_is_connected(stubbed_io, tmp_path, capsys):
    bad = _write_yaml(tmp_path / "bad.yaml", {"version": 2})
    for cmd in (["recall"], ["run", str(tmp_path / "out.json"), "k5", "--max-usd", "0.1"]):
        assert replay.main([*cmd, "--taxonomy", str(bad)]) == 1
    assert "connected" not in stubbed_io
    assert "dimensions: missing" in capsys.readouterr().err
