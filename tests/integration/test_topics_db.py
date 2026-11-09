from pathlib import Path

import pytest
import yaml

from vendorkb.db import connect, init_schema
from vendorkb.taxonomy import load
from vendorkb.topics import detect_topic, load_topics

FIXTURE = Path(__file__).parent.parent / "fixtures" / "taxonomy_v2_small.yaml"
V2_TABLES = ("dimensions", "classes", "object_types", "targets", "variants", "offerings",
             "editions", "aliases", "topics", "topic_aspects")


def _topic(topic_id: str, triggers: list[str]) -> dict:
    """A topic over the fixture's dimensions: two aspects, the second without cues."""
    return {"id": topic_id, "name": f"Topic {topic_id}", "triggers": triggers,
            "aspects": [
                {"id": "keeping", "name": "Keeping copies", "query": "How long copies are kept",
                 "cues": ["Retention", "keep  copies"],
                 "dimensions": ["retention.maximum_period", "schedule.backup_frequency"]},
                {"id": "getting_back", "name": "Getting data back",
                 "query": "Restoring objects, and writing them out of a backup",
                 "dimensions": ["coverage.restore", "coverage.export"]}]}


@pytest.fixture
async def conn(vkb_dsn):
    conn = await connect(vkb_dsn)
    try:
        await conn.execute(f"DROP TABLE IF EXISTS capabilities, features, {', '.join(V2_TABLES)}"
                           " CASCADE")
        await init_schema(conn, embed_dim=768)
        await conn.execute("DELETE FROM taxonomy_meta")
        yield conn
    finally:
        await conn.close()


async def _load(conn, tmp_path: Path, topics: list[dict]) -> None:
    doc = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    if topics:
        doc["topics"] = topics
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    await load.load_taxonomy(conn, path)


async def test_load_topics_reads_the_loaded_taxonomy(conn, tmp_path):
    # YAML order is b before a: load_topics orders by topic id, so "a" wins a double match
    await _load(conn, tmp_path, [_topic("b_loss", ["Data Loss", "lost data"]),
                                 _topic("a_loss", ["Data-Loss"])])
    topics = await load_topics(conn)
    assert [t.id for t in topics] == ["a_loss", "b_loss"]
    b = topics[1]
    assert isinstance(b, load.Topic) and b.name == "Topic b_loss"
    assert b.triggers == ("data loss", "lost data")  # stored normalized
    assert [a.id for a in b.aspects] == ["keeping", "getting_back"]  # in ordinal order
    assert all(isinstance(a, load.Aspect) for a in b.aspects)
    keeping, getting_back = b.aspects
    assert keeping == load.Aspect(
        "keeping", "Keeping copies", "How long copies are kept", ("retention", "keep copies"),
        ("retention.maximum_period", "schedule.backup_frequency"))
    assert getting_back.cues == ()
    assert getting_back.dimensions == ("coverage.restore", "coverage.export")
    assert getting_back.query == "Restoring objects, and writing them out of a backup"
    # what the database holds detects the topic, and a cue of its aspects still blocks it
    assert detect_topic("Is there protection against data loss?", topics) is topics[0]
    assert detect_topic("Is there lost data in the retention period?", topics) is None


async def test_load_topics_is_empty_for_a_taxonomy_without_topics(conn, tmp_path):
    await _load(conn, tmp_path, [])
    assert await load_topics(conn) == []
