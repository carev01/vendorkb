import json
from pathlib import Path

import pytest
import yaml

from eval.findability import coverage_report, load_map, main
from vendorkb.taxonomy.load import parse_doc, validate

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"


def _taxonomy():
    return validate(parse_doc(FIXTURE.read_bytes(), FIXTURE))


def _rows():
    return [
        {"table": "T1", "row": "r1", "vendor": "v", "ids": ["coverage.backup"],
         "out_of_scope": None},
        {"table": "T1", "row": "r2", "vendor": None,
         "ids": ["target:okta", "object:saas.identity:user"], "out_of_scope": None},
        {"table": "T2", "row": "r3", "vendor": "v", "ids": [], "out_of_scope": None},
        {"table": "T2", "row": "r4", "vendor": "v", "ids": ["retention.nonesuch"],
         "out_of_scope": None},
        {"table": "T3", "row": "r5", "vendor": "v", "ids": [],
         "out_of_scope": "not a backup feature"},
    ]


def test_counts_on_five_row_map():
    rep = coverage_report(_rows(), _taxonomy())
    assert rep["rows"] == 5
    assert rep["out_of_scope"] == 1
    assert rep["in_scope"] == 4
    assert rep["mapped"] == 3
    assert rep["unmapped"] == [{"table": "T2", "row": "r3"}]
    assert rep["unknown_ids"] == [{"table": "T2", "row": "r4", "id": "retention.nonesuch"}]


def test_object_id_from_parent_class_is_accepted():
    rows = [{"table": "T", "row": "r", "ids": ["object:saas.crm:tenant_config"]}]
    rep = coverage_report(rows, _taxonomy())
    assert rep["unknown_ids"] == []
    assert rep["mapped"] == 1


def test_object_id_outside_lineage_or_malformed_is_unknown():
    bad = ["object:saas.crm:user", "object:nope:user", "object:saas.crm", "target:nope",
           "target:", "nonsense"]
    rows = [{"table": "T", "row": "r", "ids": bad}]
    rep = coverage_report(rows, _taxonomy())
    assert [u["id"] for u in rep["unknown_ids"]] == bad


def test_all_ids_valid_kinds():
    ids = ["coverage.compare", "target:aws.rds.oracle", "object:cloud.database:database"]
    rep = coverage_report([{"table": "T", "row": "r", "ids": ids}], _taxonomy())
    assert rep["unknown_ids"] == [] and rep["unmapped"] == []


def _write(tmp_path, rows):
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump(rows))
    return path


def test_exit_1_when_anything_unmapped(tmp_path, capsys):
    path = _write(tmp_path, _rows())
    assert main(["coverage", "--map", str(path), "--taxonomy", str(FIXTURE)]) == 1
    out = capsys.readouterr().out
    assert "r3" in out and "retention.nonesuch" in out


def test_exit_0_and_json_when_clean(tmp_path, capsys):
    rows = [r for r in _rows() if r["row"] in ("r1", "r2", "r5")]
    path = _write(tmp_path, rows)
    out_json = tmp_path / "report.json"
    code = main(["coverage", "--map", str(path), "--taxonomy", str(FIXTURE),
                 "--json", str(out_json)])
    assert code == 0
    rep = json.loads(out_json.read_text())
    assert rep["rows"] == 3 and rep["in_scope"] == 2 and rep["mapped"] == 2
    assert "in_scope" in capsys.readouterr().out


def test_invalid_yaml_map_names_only_the_path(tmp_path):
    path = tmp_path / "map.yaml"
    path.write_text("- table: T\n  row: SECRET-ROW-LABEL\n  ids: [unclosed\n")
    with pytest.raises(ValueError) as info:
        load_map(path)
    assert str(info.value) == f"{path}: not valid YAML"
    assert info.value.__cause__ is None and info.value.__suppress_context__
    assert "SECRET" not in repr(info.value)
