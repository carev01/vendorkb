"""Findability checks for a comparison-table coverage map (the map stays outside the repo).

The map is a YAML list of ``{table, row, vendor, ids, out_of_scope}``: ``vendor`` is the
table's vendor (or null), ``ids`` the taxonomy v2 ids the row maps to, ``out_of_scope`` a
reason text (or null). An id is a dimension id, ``target:<target_id>`` or
``object:<class_id>:<object_type_id>`` (the object type must be in that class's lineage).

    uv run python eval/findability.py coverage --map PATH --taxonomy taxonomy/taxonomy.yaml \
        [--json OUT]

``coverage`` checks that every in-scope row has at least one id and that every id exists in the
taxonomy; it exits 1 if a row is unmapped or an id is unknown. It prints only table names, row
labels and ids, never anything else from the map.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from vendorkb.taxonomy.load import Taxonomy, parse_doc, validate


def _known_ids(taxonomy: Taxonomy) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    dimensions = {d.id for d in taxonomy.dimensions}
    targets = {t.id for t in taxonomy.targets}
    objects = {(c.id, o.id) for c in taxonomy.classes for o in taxonomy.object_types(c.id)}
    return dimensions, targets, objects


def _id_is_known(id_: str, dimensions: set[str], targets: set[str],
                 objects: set[tuple[str, str]]) -> bool:
    if id_.startswith("target:"):
        return id_.removeprefix("target:") in targets
    if id_.startswith("object:"):
        class_id, sep, object_type = id_.removeprefix("object:").rpartition(":")
        return bool(sep) and (class_id, object_type) in objects
    return id_ in dimensions


def coverage_report(map_rows: list[dict[str, Any]], taxonomy: Taxonomy) -> dict[str, Any]:
    """Count the map's rows and list the unmapped rows and unknown ids.

    A row with a non-empty ``out_of_scope`` is out of scope (its ids are not checked). An
    in-scope row is ``mapped`` if it has any id and ``unmapped`` if it has none; an id the
    taxonomy lacks is listed in ``unknown_ids`` whether or not the row has other ids.
    """
    dimensions, targets, objects = _known_ids(taxonomy)
    out_of_scope = 0
    mapped = 0
    unmapped: list[dict[str, str]] = []
    unknown: list[dict[str, str]] = []
    for row in map_rows:
        if row.get("out_of_scope"):
            out_of_scope += 1
            continue
        ids = row.get("ids") or []
        if not ids:
            unmapped.append({"table": row["table"], "row": row["row"]})
            continue
        mapped += 1
        for id_ in ids:
            if not isinstance(id_, str) or not _id_is_known(id_, dimensions, targets, objects):
                unknown.append({"table": row["table"], "row": row["row"], "id": str(id_)})
    return {"rows": len(map_rows), "out_of_scope": out_of_scope,
            "in_scope": len(map_rows) - out_of_scope, "mapped": mapped,
            "unmapped": unmapped, "unknown_ids": unknown}


def load_map(path: str | Path) -> list[dict[str, Any]]:
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: the map must be a YAML list of rows")
    for n, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("table") or not row.get("row"):
            raise ValueError(f"{path}: row {n} needs a table and a row label")
        if not isinstance(row.get("ids") or [], list):
            raise ValueError(f"{path}: row {n} ids must be a list")
    return rows


def cmd_coverage(args: argparse.Namespace) -> int:
    taxonomy = validate(parse_doc(Path(args.taxonomy).read_bytes(), args.taxonomy))
    report = coverage_report(load_map(args.map), taxonomy)
    for key in ("rows", "out_of_scope", "in_scope", "mapped"):
        print(f"{key}: {report[key]}")
    print(f"unmapped: {len(report['unmapped'])}")
    for u in report["unmapped"]:
        print(f"  {u['table']} | {u['row']}")
    print(f"unknown_ids: {len(report['unknown_ids'])}")
    for u in report["unknown_ids"]:
        print(f"  {u['table']} | {u['row']} | {u['id']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if report["unmapped"] or report["unknown_ids"] else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage", help="check every map row points at taxonomy v2 ids")
    c.add_argument("--map", required=True, help="the coverage map YAML (kept outside the repo)")
    c.add_argument("--taxonomy", required=True)
    c.add_argument("--json", help="also write the report here")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"coverage": cmd_coverage}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
