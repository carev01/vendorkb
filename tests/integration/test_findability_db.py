"""``eval/findability.py`` claims, chunks-candidates and chunks-score on a synthetic database,
the fixture taxonomy and a synthetic map (every name here is invented)."""
import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pgvector import Vector

from eval.findability import (CountingEmbedder, CountingReranker, build_question,
                              chunk_candidates, chunks_score, claims_report, load_map, main,
                              row_filters)
from vendorkb.db import connect, init_schema
from vendorkb.taxonomy import capabilities, load
from vendorkb.taxonomy.capabilities import CapFilter

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"
NOTE = "SECRET-NOTE-do-not-print"
USER = "object:saas.identity:user"


def _taxonomy():
    return load.validate(load.parse_doc(FIXTURE.read_bytes(), FIXTURE))


async def _fresh(conn) -> str:
    await conn.execute("DROP TABLE IF EXISTS capabilities, raw_claims, rejected_claims,"
                       " extractions, extract_batches, chunks, documents CASCADE")
    await init_schema(conn, embed_dim=8)
    await load.load_taxonomy(conn, FIXTURE)
    version = await load.current_version(conn)
    assert version is not None
    return version


async def _doc(conn, vendor: str, n_chunks: int = 0, text: str = "text") -> list[str]:
    doc_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO documents (doc_id, vendor, product, source, title, url, content_hash,"
        " fetched_at) VALUES (%s,%s,'P','S','Doc','http://docs/x',%s,%s)",
        (doc_id, vendor, str(doc_id), datetime(2026, 1, 1, tzinfo=UTC)))
    chunk_ids = []
    for n in range(n_chunks):
        chunk_id = uuid.uuid4()
        chunk_ids.append(str(chunk_id))
        await conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, section_path, ordinal, text, token_count,"
            " embedding) VALUES (%s,%s,%s,%s,%s,10,%s)",
            (chunk_id, doc_id, ["A"], n, f"{text} {n}", Vector([0.1] * 8)))
    return chunk_ids


async def _claim(conn, version: str, dimension: str, target: str | None = None, *,
                 vendor="examplevendor", class_id=None, object_type=None) -> None:
    (chunk_id,) = await _doc(conn, vendor, 1)
    ext_id, claim_id = uuid.uuid4(), uuid.uuid4()
    await conn.execute(
        "INSERT INTO extractions (extraction_id, chunk_id, extractor_model, prompt_version,"
        " status) VALUES (%s,%s,'m','claims-v3.1','done')", (ext_id, chunk_id))
    await conn.execute(
        "INSERT INTO raw_claims (claim_id, extraction_id, chunk_id, vendor, product, kind,"
        " capability_raw, target_raw, support, availability, evidence_quote, gate,"
        " extractor_model, prompt_version, dimension_id, dimension_method, class_id,"
        " target_id, object_type_id, mapped_version)"
        " VALUES (%s,%s,%s,%s,'P','capability','some capability',%s,'supported','unstated',"
        " 'quote','exact','m','claims-v3.1',%s,'alias',%s,%s,%s,%s)",
        (claim_id, ext_id, chunk_id, vendor, target, dimension, class_id, target, object_type,
         version))


def _row(table, row, vendor, ids, out_of_scope=None):
    return {"table": table, "row": row, "vendor": vendor, "ids": ids,
            "out_of_scope": out_of_scope, "notes": NOTE}


def _map():
    return [
        _row("T1", "has target+object", "examplevendor", ["target:okta", USER]),
        _row("T1", "has wide dimension", "examplevendor", ["retention.maximum_period"]),
        _row("T1", "dimension with target", "examplevendor",
             ["target:okta", "retention.maximum_period"]),
        _row("T1", "other target", "examplevendor", ["target:entra_id", USER]),
        _row("T2", "other vendor", "othervendor", ["target:okta", USER]),
        _row("T2", "no vendor", None, ["target:okta"]),
        _row("T2", "out of scope", "examplevendor", [], "pricing"),
        _row("T2", "no ids", "examplevendor", []),
    ]


async def _seed_claims(conn) -> None:
    v = await _fresh(conn)
    await _claim(conn, v, "coverage.backup", "okta", class_id="saas.identity",
                 object_type="user")
    await _claim(conn, v, "retention.maximum_period")
    await capabilities.build(conn)


def test_row_filters_cross_targets_with_aspects():
    filters = row_filters("v", ["target:a", "target:b", USER, "retention.maximum_period"])
    assert len(filters) == 2 * (4 + 1)
    assert CapFilter(vendor="v", target="a", dimension="coverage.compare",
                     object_type="user") in filters
    assert CapFilter(vendor="v", target="b", dimension="retention.maximum_period") in filters
    assert row_filters("v", ["target:a"]) == [CapFilter(vendor="v", target="a")]
    assert row_filters("v", ["retention.maximum_period"]) == [
        CapFilter(vendor="v", dimension="retention.maximum_period")]


def test_question_uses_taxonomy_names():
    t = _taxonomy()
    assert build_question("examplevendor", t, ["target:entra_id", USER]) == (
        "examplevendor Microsoft Entra ID User")
    assert build_question("v", t, ["target:okta", "retention.maximum_period"]) == (
        "v Okta Maximum retention period")
    assert build_question("v", t, ["retention.maximum_period"]) == (
        "v Maximum retention period")


async def test_claims_counts(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _seed_claims(conn)
        report = await claims_report(conn, _map())
    finally:
        await conn.close()
    assert (report["rows"], report["out_of_scope"], report["no_vendor"], report["no_ids"],
            report["checked"], report["with_fact"]) == (8, 1, 1, 1, 5, 3)
    assert report["with_fact_by_group"] == {"facts": 1, "product_level": 0, "product_wide": 2}
    assert [w["row_key"] for w in report["without_fact"]] == [
        "T1|other target", "T2|other vendor"]
    assert NOTE not in json.dumps(report)


async def test_claims_cli_prints_only_labels_and_writes_json(vkb_dsn, tmp_path, capsys):
    conn = await connect(vkb_dsn)
    try:
        await _seed_claims(conn)
    finally:
        await conn.close()
    map_path, out = tmp_path / "map.yaml", tmp_path / "claims.json"
    map_path.write_text(yaml.safe_dump(_map()), encoding="utf-8")
    assert await asyncio.to_thread(
        main, ["claims", "--map", str(map_path), "--dsn", vkb_dsn, "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "with_fact: 3" in printed and "  T1|other target" in printed
    assert NOTE not in printed
    assert json.loads(out.read_text())["with_fact"] == 3


class FakeEmbedder:
    def __init__(self):
        self.questions: list[str] = []

    async def embed(self, texts, *, input_type="document"):
        assert input_type == "query"
        self.questions.extend(texts)
        return [[0.1] * 8 for _ in texts]


class FakeReranker:
    def __init__(self, fail=False):
        self.fail = fail

    async def rerank(self, query, documents, top_k):
        if self.fail:
            return None
        order = sorted(range(len(documents)), key=lambda i: documents[i])
        return [(i, 1.0 - n / 100) for n, i in enumerate(order[:top_k])]


async def _candidates(conn, *, reranker, limit=None):
    claims = await claims_report(conn, _map())
    embedder = CountingEmbedder(FakeEmbedder())
    counted = CountingReranker(reranker)
    result = await chunk_candidates(conn, _map(), claims, _taxonomy(), embedder=embedder,
                                    reranker=counted, limit=limit)
    return result, embedder, counted


async def test_candidates_only_for_rows_without_claims(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _seed_claims(conn)
        await _doc(conn, "examplevendor", 12, text="entra user")
        await _doc(conn, "othervendor", 3, text="other user")
        result, embedder, reranker = await _candidates(conn, reranker=FakeReranker())
    finally:
        await conn.close()
    by_row: dict[str, list[dict]] = {}
    for c in result["candidates"]:
        assert set(c) == {"row_key", "chunk_id", "text"}
        by_row.setdefault(c["row_key"], []).append(c)
    assert set(by_row) == {"T1|other target", "T2|other vendor"}  # not the rows with claims
    assert len(by_row["T1|other target"]) == 10  # top 10 of 14 examplevendor chunks
    assert all(c["text"].startswith("entra user") for c in by_row["T1|other target"])
    assert len(by_row["T2|other vendor"]) == 3  # scoped to othervendor (4 with its claim doc)
    assert not any(c["text"].startswith("entra") for c in by_row["T2|other vendor"])
    assert embedder._inner.questions == ["examplevendor Microsoft Entra ID User",
                                         "othervendor Okta User"]
    assert (result["rows"], embedder.calls, reranker.calls) == (2, 2, 2)


async def test_candidates_limit_and_rerank_failure(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await _seed_claims(conn)
        await _doc(conn, "examplevendor", 12)
        limited, embedder, _ = await _candidates(conn, reranker=FakeReranker(), limit=1)
        failed, _, _ = await _candidates(conn, reranker=FakeReranker(fail=True))
    finally:
        await conn.close()
    assert limited["rows"] == 1 and embedder.calls == 1
    assert {c["row_key"] for c in limited["candidates"]} == {"T1|other target"}
    assert failed["candidates"] == []
    # othervendor has no chunk here: no hits is not a rerank failure
    assert (failed["rerank_failed"], failed["no_hits"]) == (
        ["T1|other target"], ["T2|other vendor"])


def test_chunks_score_arithmetic(tmp_path, capsys):
    candidates = [{"row_key": "T|a", "chunk_id": "1", "text": "x"},
                  {"row_key": "T|a", "chunk_id": "2", "text": "x"},
                  {"row_key": "T|b", "chunk_id": "3", "text": "x"},
                  {"row_key": "T|c", "chunk_id": "4", "text": "x"},
                  {"row_key": "T|c", "chunk_id": "5", "text": "x"}]
    labels = [{"row_key": "T|a", "chunk_id": "1", "relevant": False},
              {"row_key": "T|a", "chunk_id": "2", "relevant": True},
              {"row_key": "T|b", "chunk_id": "3", "relevant": False},
              {"row_key": "T|c", "chunk_id": "4", "relevant": True},
              {"row_key": "T|z", "chunk_id": "9", "relevant": True}]
    assert chunks_score(candidates, labels) == {
        "rows": 3, "with_relevant": 2, "without_relevant": ["T|b"], "unlabelled": 1,
        "stray_labels": 1}
    cand_path, lab_path, out = (tmp_path / n for n in ("c.jsonl", "l.jsonl", "o.json"))
    cand_path.write_text("".join(json.dumps(c) + "\n" for c in candidates), encoding="utf-8")
    lab_path.write_text("".join(json.dumps(x) + "\n" for x in labels), encoding="utf-8")
    assert main(["chunks-score", "--candidates", str(cand_path), "--labels", str(lab_path),
                 "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "with_relevant: 2" in printed and "  T|b" in printed
    assert json.loads(out.read_text())["with_relevant"] == 2


def test_load_map_roundtrip(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(_map()), encoding="utf-8")
    assert len(load_map(p)) == 8
