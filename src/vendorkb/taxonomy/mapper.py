"""Map every raw claim to a feature and a target of the loaded taxonomy.

For each claim's normalized ``capability_raw`` (and ``target_raw``): an exact match on an
alias or a normalized name is ``alias`` (conf 1.0); otherwise the embedding nearest
neighbour among the names and aliases, when its cosine is at least ``threshold``, is ``nn``
(conf = the cosine); otherwise ``unmapped``. A claim with no target gets no target mapping.
Every claim is (re)mapped and stamped with the taxonomy version, in one transaction, so
``capabilities.build`` only ever uses mappings made under the current version. LLM
adjudication of the unmapped is deferred (spec §5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import psycopg

from vendorkb.taxonomy.cluster import normalize
from vendorkb.taxonomy.load import current_version

log = logging.getLogger(__name__)

Kind = Literal["feature", "target"]
Match = tuple[str | None, str, float | None]  # (entity id, method, conf)
_NN_BLOCK = 1024  # query rows per similarity matmul


class _Embedder(Protocol):
    async def embed(self, texts: list[str], *, input_type: Literal["document", "query"] = ...
                    ) -> list[list[float]]: ...


@dataclass
class MapReport:
    taxonomy_version: str = ""
    claims: int = 0
    feature_alias: int = 0
    feature_nn: int = 0
    feature_unmapped: int = 0
    target_alias: int = 0
    target_nn: int = 0
    target_unmapped: int = 0
    target_none: int = 0  # claims with no target_raw
    embedded_strings: int = 0


async def _exact_index(conn: psycopg.AsyncConnection, kind: Kind) -> dict[str, str]:
    """normalized alias or name -> entity id (an alias wins over another entity's name)."""
    cur = await conn.execute(
        "SELECT alias, entity_id FROM aliases WHERE entity_type=%s ORDER BY alias", (kind,))
    index = {normalize(a): eid for a, eid in await cur.fetchall()}
    table, col = ("features", "feature_id") if kind == "feature" else ("targets", "target_id")
    cur = await conn.execute(f"SELECT {col}, name FROM {table} ORDER BY {col}")
    for eid, name in await cur.fetchall():
        index.setdefault(normalize(name), eid)
    return index


def _unit(vectors: list[list[float]]) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def nearest(queries: list[list[float]], index: list[list[float]]) -> list[tuple[int, float]]:
    """(index position, cosine) of each query's nearest index vector."""
    q, ix = _unit(queries), _unit(index)
    out: list[tuple[int, float]] = []
    for start in range(0, len(q), _NN_BLOCK):
        sims = q[start:start + _NN_BLOCK] @ ix.T
        best = sims.argmax(axis=1)
        out.extend(zip(best.tolist(), sims[np.arange(len(best)), best].tolist()))
    return out


async def _match(strings: set[str], exact: dict[str, str], *, embedder: _Embedder,
                 threshold: float, report: MapReport) -> dict[str, Match]:
    out: dict[str, Match] = {s: (exact[s], "alias", 1.0) for s in strings if s in exact}
    rest = sorted(strings - out.keys())
    if rest and exact:
        keys = sorted(exact)
        vectors = await embedder.embed(keys + rest, input_type="document")
        report.embedded_strings += len(keys) + len(rest)
        for s, (j, cos) in zip(rest, nearest(vectors[len(keys):], vectors[:len(keys)])):
            out[s] = (exact[keys[j]], "nn", cos) if cos >= threshold else (None, "unmapped",
                                                                              None)
    for s in rest:
        out.setdefault(s, (None, "unmapped", None))
    return out


async def map_claims(conn: psycopg.AsyncConnection, *, embedder: _Embedder,
                     threshold: float) -> MapReport:
    version = await current_version(conn)
    if version is None:
        raise RuntimeError("no taxonomy loaded: run `vendorkb taxonomy load PATH` first")
    report = MapReport(taxonomy_version=version)
    cur = await conn.execute("SELECT claim_id, capability_raw, target_raw FROM raw_claims")
    claims = [(cid, normalize(cap), None if tgt is None or not normalize(tgt)
               else normalize(tgt)) for cid, cap, tgt in await cur.fetchall()]
    report.claims = len(claims)
    features = await _match({c for _, c, _ in claims}, await _exact_index(conn, "feature"),
                            embedder=embedder, threshold=threshold, report=report)
    targets = await _match({t for _, _, t in claims if t is not None},
                           await _exact_index(conn, "target"),
                           embedder=embedder, threshold=threshold, report=report)
    rows = []
    for cid, cap, tgt in claims:
        fid, fmethod, fconf = features[cap]
        setattr(report, f"feature_{fmethod}", getattr(report, f"feature_{fmethod}") + 1)
        if tgt is None:
            tid, tmethod, tconf = None, None, None
            report.target_none += 1
        else:
            tid, tmethod, tconf = targets[tgt]
            setattr(report, f"target_{tmethod}", getattr(report, f"target_{tmethod}") + 1)
        rows.append((cid, fid, fmethod, fconf, tid, tmethod, tconf))
    async with conn.transaction():
        await conn.execute(
            "CREATE TEMP TABLE _claim_map (claim_id uuid PRIMARY KEY, feature_id text,"
            " feature_method text, feature_conf real, target_id text, target_method text,"
            " target_conf real) ON COMMIT DROP")
        async with conn.cursor() as c:
            async with c.copy("COPY _claim_map FROM STDIN") as copy:
                for row in rows:
                    await copy.write_row(row)
        await conn.execute(
            "UPDATE raw_claims r SET feature_id=m.feature_id, feature_method=m.feature_method,"
            " feature_conf=m.feature_conf, target_id=m.target_id,"
            " target_method=m.target_method, target_conf=m.target_conf, mapped_version=%s"
            " FROM _claim_map m WHERE r.claim_id = m.claim_id", (version,))
    log.info("taxonomy map: %s", report)
    return report
