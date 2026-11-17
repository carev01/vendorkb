"""The in-place YAML editing behind ``taxonomy propose``: a round trip keeps the file as
written, appended entries follow their siblings' style, and an undo restores the text."""
from pathlib import Path

import yaml

from ruamel.yaml.comments import CommentedMap

from vendorkb.taxonomy import load, propose

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"

TEXT = """\
version: 2
# a comment the review must keep
dimensions:
- {id: coverage.backup, name: B, definition: d, applies_to: all}
- {id: coverage.restore, name: R, definition: d, applies_to: all}
- {id: coverage.export, name: E, definition: d, applies_to: all}
- {id: coverage.compare, name: C, definition: d, applies_to: all}
classes:
- id: saas
  name: SaaS
  definition: d
  object_types: []
- id: cloud
  name: Cloud
  definition: d

targets:
- id: thing
  name: Thing
  class: saas
  aliases: [a thing]

offerings: []
"""


def _entry(*pairs) -> CommentedMap:
    m = CommentedMap()
    propose._set(m, *pairs)
    return m


def test_round_trip_is_identical():
    for text in (TEXT, FIXTURE.read_text()):
        assert propose._Doc(text).dump() == text


def test_append_follows_sibling_style_and_undo_restores():
    doc = propose._Doc(TEXT)
    undos = [
        propose._append(doc.data, "dimensions", _entry(
            ("id", "cyber.scan"), ("name", "S"), ("definition", "d"), ("applies_to", "all"),
            ("proposed", True)), named=False),
        propose._append(doc.data["classes"][0], "object_types", _entry(
            ("id", "user"), ("name", "User"), ("proposed", True)), named=True),
        propose._append(doc.data["classes"][1], "object_types", _entry(
            ("id", "vm"), ("name", "VM"), ("proposed", True)), named=True),
        propose._append(doc.data, "targets", _entry(
            ("id", "other"), ("name", "Other"), ("class", "cloud"), ("proposed", True)),
            named=False),
    ]
    out = doc.dump()
    load.validate(load.parse_doc(out.encode()))
    assert "- {id: cyber.scan, name: S, definition: d, applies_to: all, proposed: true}\n" \
        "classes:" in out  # flow, like its siblings
    assert "  object_types:\n  - {id: user, name: User, proposed: true}\n- id: cloud" in out
    assert "  object_types:\n  - {id: vm, name: VM, proposed: true}\n\ntargets:" in out
    # the blank line that closed the targets list now follows the appended target
    assert "  aliases: [a thing]\n- id: other\n  name: Other\n  class: cloud\n" \
        "  proposed: true\n\nofferings:" in out
    for undo in reversed(undos):
        undo()
    assert doc.dump() == TEXT


def test_add_aliases_dedupes_and_quotes():
    doc = propose._Doc(TEXT)
    target = doc.data["targets"][0]
    undo = propose._add_aliases(target, ["a thing", "yes", "another thing", "yes"])
    assert "  aliases: [a thing, 'yes', another thing]\n" in doc.dump()
    undo()
    assert doc.dump() == TEXT
    cloud = doc.data["classes"][1]
    undo = propose._add_aliases(cloud, ["clouds"])  # no aliases key: added after definition
    assert list(cloud) == ["id", "name", "definition", "aliases"]
    assert "  definition: d\n  aliases: [clouds]\n\ntargets:" in doc.dump()
    undo()
    assert doc.dump() == TEXT


def test_add_aliases_with_nothing_new_changes_nothing():
    doc = propose._Doc(TEXT)
    cloud = doc.data["classes"][1]
    propose._add_aliases(cloud, [])()
    propose._add_aliases(doc.data["targets"][0], ["a thing"])
    assert "aliases" not in cloud and doc.dump() == TEXT


def test_reads_back_catches_what_yaml_1_1_misreads():
    doc = propose._Doc(TEXT)
    assert not propose._reads_back(doc, _entry(("id", "on"), ("name", "null")))
    assert propose._reads_back(doc, _entry(("id", propose._scalar("on")),
                                           ("name", propose._scalar("null"))))
    assert propose._reads_back(doc, propose._flow_seq(["a b", propose._scalar("yes")]))


WRAPPED = """\
version: 2
dimensions:
- id: coverage.backup
  name: Backup
  definition: d
  aliases: [first phrase, second phrase, third phrase,
      fourth phrase, fifth phrase]
  applies_to: all
- id: coverage.restore
  name: Restore
  definition: d
  aliases: [restore phrase one, restore phrase two,
    restore phrase three]
  applies_to: all
- id: coverage.export
  name: Export
  definition: d
  aliases: [export phrase, another export phrase, a third export phrase, and a much longer export phrase that runs on]
  applies_to: all
"""


def test_dump_keeps_the_original_text_of_every_entry_it_did_not_change():
    """ruamel keeps no line breaks inside a flow list, and a file wrapped at mixed widths
    matches no single width: only the changed entry's lines may change."""
    doc = propose._Doc(WRAPPED)
    assert doc.dump() == WRAPPED  # nothing changed: the file as written
    propose._add_aliases(doc.data["dimensions"][1], ["restore phrase four"])
    entry = CommentedMap()
    propose._set(entry, ("id", "coverage.compare"), ("name", "Compare"), ("definition", "d"))
    entry["applies_to"] = "all"
    propose._append(doc.data, "dimensions", entry, named=False)
    out = doc.dump()
    removed = [line for line in WRAPPED.splitlines() if line not in out.splitlines()]
    assert removed == ["  aliases: [restore phrase one, restore phrase two,",
                       "    restore phrase three]"]
    assert out.startswith(WRAPPED.split("- id: coverage.restore")[0])  # entry 1 as written
    assert "and a much longer export phrase that runs on]\n  applies_to: all\n" in out
    data = yaml.safe_load(out)
    assert data["dimensions"][1]["aliases"][-1] == "restore phrase four"
    assert data["dimensions"][-1]["id"] == "coverage.compare"
    assert data == yaml.safe_load(doc._dump(doc.data))


def test_dump_falls_back_to_the_full_dump_when_the_splice_reads_back_wrong(monkeypatch):
    doc = propose._Doc(WRAPPED)
    propose._add_aliases(doc.data["dimensions"][0], ["sixth phrase"])
    monkeypatch.setattr(propose, "_splice", lambda original, baseline, new: original)
    assert doc.dump() == doc._dump(doc.data)
