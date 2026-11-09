"""Topics: the taxonomy's broad subjects (a fixed kit of aspects), and spotting one in a question.

A topic fires only on a broad question: one of its triggers is in the question and the question
names none of its aspects (no aspect cue). Triggers and cues are compared with the question after
the same normalising: lowercase, hyphens as spaces, whitespace collapsed. A trigger matches as a
whole phrase, with an optional plural suffix; a cue matches at a word start, so a stem such as
``immutab`` covers every form of the word.
"""
from __future__ import annotations

import re
from functools import lru_cache

import psycopg

from vendorkb.taxonomy.cluster import normalize
from vendorkb.taxonomy.load import Aspect, Topic

_HYPHENS = re.compile("[-‐‑]")  # hyphen-minus, hyphen, non-breaking hyphen


def _norm(text: str) -> str:
    return normalize(_HYPHENS.sub(" ", text))


@lru_cache(maxsize=512)
def _trigger(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(_norm(phrase))}(?:s|es)?(?!\w)")


@lru_cache(maxsize=512)
def _cue(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(_norm(phrase))}")


def detect_topic(question: str, topics: list[Topic]) -> Topic | None:
    """The first of ``topics`` whose trigger the question holds and none of whose aspects it
    names, else None (the question is handled as it is without topics)."""
    text = _norm(question)
    for topic in topics:
        if not any(_trigger(t).search(text) for t in topic.triggers):
            continue
        if not any(_cue(c).search(text) for a in topic.aspects for c in a.cues):
            return topic
    return None


async def load_topics(conn: psycopg.AsyncConnection) -> list[Topic]:
    """Every topic of the loaded taxonomy, by topic id, its aspects in YAML order."""
    cur = await conn.execute(  # one statement: a reload cannot split a topic from its aspects
        "SELECT t.topic_id, t.name, t.triggers, a.aspect_id, a.name, a.query, a.cues,"
        " a.dimensions FROM topics t JOIN topic_aspects a USING (topic_id)"
        ' ORDER BY t.topic_id COLLATE "C", a.ordinal')
    heads: dict[str, tuple[str, tuple[str, ...]]] = {}
    aspects: dict[str, list[Aspect]] = {}
    for topic_id, name, triggers, aspect_id, aspect_name, query, cues, dimensions in (
            await cur.fetchall()):
        heads[topic_id] = (name, tuple(triggers))
        aspects.setdefault(topic_id, []).append(
            Aspect(aspect_id, aspect_name, query, tuple(cues), tuple(dimensions)))
    return [Topic(topic_id, name, triggers, tuple(aspects[topic_id]))
            for topic_id, (name, triggers) in heads.items()]
