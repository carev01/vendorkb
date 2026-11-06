"""The in-place YAML editing behind ``taxonomy propose``: a round trip keeps the file as
written, appended entries follow their siblings' style, and an undo restores the text."""
from pathlib import Path

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
