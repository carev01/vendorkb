"""Replay the mapper's dimension adjudication on a labelled sample, under variants, without
storing anything; score the answers against the labels.

    uv run python eval/map_replay.py run OUT.json k5 k8 k5+second --max-usd 0.3
    uv run python eval/map_replay.py score OUT.json [OUT2.json ...]
    uv run python eval/map_replay.py recall          # candidate recall only, no model call

The labels (``eval/labels/dimension-sample-2026-11-08.json``) give each claim the acceptable
answers (dimension ids, ``null`` for none). A variant is ``k<N>`` (the N nearest applicable
dimensions, as the mapper computes them, without pinning), optionally ``+second`` (a none answer
is asked once more over the five best keyword matches between the claim's phrase, target and
evidence quote and the dimensions' names, definitions and aliases). Calls go through the
mapper's own request path (MAP_* settings), embeddings come from ``embedding_cache``, and no
decision is written. A run costs about $0.03 per variant on Haiku.

An answer the labels do not cover counts as wrong (``unlabelled`` in the score): label those
before trusting a variant that produces many.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = Path(__file__).resolve().parent / "labels" / "dimension-sample-2026-11-08.json"
STOP = frozenset("a an and are as at be by can for from in into is it its of on or that the this to "
                 "with when which you your not no all any each per been being do does".split())


def words(text: str) -> list[str]:
    """Lowercase content words, a plural "s" dropped from words longer than four letters."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w not in STOP and len(w) > 2:
            out.append(w[:-1] if w.endswith("s") and len(w) > 4 else w)
    return out


def bm25_ranker(docs: dict[str, str]):
    """rank(query, allowed, exclude, k): the best ``k`` doc ids by BM25, positive scores only."""
    tf = {d: Counter(words(t)) for d, t in docs.items()}
    n, avg = len(tf), sum(sum(c.values()) for c in tf.values()) / max(len(tf), 1)
    df = Counter(w for c in tf.values() for w in c)

    def rank(query: str, allowed: set[str], exclude: set[str], k: int) -> list[str]:
        q = set(words(query))
        scores = {}
        for d, c in tf.items():
            if d in allowed and d not in exclude:
                length = sum(c.values())
                scores[d] = sum(math.log(1 + (n - df[w] + .5) / (df[w] + .5)) * c[w] * 2.2
                                / (c[w] + 1.2 * (.25 + .75 * length / avg)) for w in q if w in c)
        return [d for d, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])) if s > 0][:k]
    return rank


def score(answers: dict[str, str | None], items: list[dict[str, Any]]) -> dict[str, Any]:
    """Precision over the answered items, recall over the items with an acceptable dimension,
    and how many answers fall outside the labels."""
    acceptable = {it["item_id"]: it["acceptable"] for it in items}
    answered = [i for i in acceptable if answers.get(i) is not None]
    with_dim = [i for i in acceptable if any(a is not None for a in acceptable[i])]
    right = [i for i in answered if answers[i] in acceptable[i]]
    return {"answered": len(answered), "precision": len(right) / len(answered) if answered else None,
            "recall": sum(1 for i in with_dim if answers.get(i) in acceptable[i]) / len(with_dim),
            "right": sum(1 for i in acceptable if answers.get(i) in acceptable[i]),
            "unlabelled": sum(1 for i in answered if answers[i] not in acceptable[i]
                              and answers[i] != next(it["assigned_2026_11_08"] for it in items
                                                     if it["item_id"] == i))}


async def _setup(conn, settings, items):
    """The taxonomy snapshot, dimension keys with vectors, and each item's phrase, class mask
    and nearest applicable dimensions."""
    import numpy as np

    from vendorkb.clients import Embedder
    from vendorkb.taxonomy import mapper as M
    from vendorkb.taxonomy.cluster import normalize

    tax = await M._snapshot(conn)
    keys = M._Keys(tax.search.get(("dimension", ""), {}))
    phrases = {it["item_id"]: normalize(it["phrase"]) for it in items}
    texts = list(dict.fromkeys([*keys.texts, *phrases.values()]))
    report = M.MapReport(taxonomy_version="replay")
    async with Embedder(settings) as emb:
        vecs = await M._embed_all(emb, texts, conn=conn,
                                  model=f"{settings.embed_model}@{settings.embed_dim}",
                                  report=report)
    print(f"embeddings: {report.embeddings_cached} cached, {report.embeddings_new} new")
    row = {t: i for i, t in enumerate(texts)}
    keys.vectors = vecs[[row[t] for t in keys.texts]]
    searchable = set(keys.entities)
    ranked = {}
    for it in items:
        cid = None if it["class_id"] == "(none)" else it["class_id"]
        allowed = tax.applicable(cid) & searchable
        scores = keys.scores(vecs[[row[phrases[it["item_id"]]]]])[0]
        ranked[it["item_id"]] = (cid, allowed, [keys.entities[j] for j in np.argsort(-scores)
                                                if keys.entities[j] in allowed])
    aliases: dict[str, list[str]] = {}
    for a, eid in tax.search.get(("dimension", ""), {}).items():
        aliases.setdefault(eid, []).append(a)
    docs = {eid: f"{name} {definition} " + " ".join(aliases.get(eid, []))
            for eid, (name, definition, _) in tax.dimensions.items() if eid in searchable}
    return tax, phrases, ranked, bm25_ranker(docs)


def _query(it: dict[str, Any]) -> str:
    return f"{it['phrase']} {it['target_text']} {it['evidence_quote']}"


async def _run(args: argparse.Namespace) -> int:
    from vendorkb.cli import _map_client
    from vendorkb.config import VendorkbSettings
    from vendorkb.db import connect
    from vendorkb.taxonomy import mapper as M

    items = json.loads(LABELS.read_text())["items"]
    settings = VendorkbSettings()
    conn = await connect(settings.vendorkb_dsn)
    try:
        await conn.execute("SET statement_timeout = '300s'")
        tax, phrases, ranked, rank = await _setup(conn, settings, items)
        contexts = {}
        for it in items:  # the labelled claim first, then others with the same phrase and class
            cid = ranked[it["item_id"]][0]
            cur = await conn.execute(
                "SELECT capability_raw, coalesce(target_raw,'-'), coalesce(target_scope,'-'),"
                " vendor, evidence_quote FROM raw_claims WHERE prompt_version LIKE %s"
                " AND (claim_id::text = %s OR (lower(capability_raw) = lower(%s)"
                "  AND class_id IS NOT DISTINCT FROM %s))"
                " ORDER BY claim_id::text <> %s, claim_id LIMIT %s",
                (M.PROMPT_FAMILY, it["claim_id"], it["phrase"], cid, it["claim_id"],
                 M.MAX_CONTEXTS))
            contexts[it["item_id"]] = [(c, t, s, v, " ".join((q or "").split())[:M.QUOTE_CHARS])
                                       for c, t, s, v, q in await cur.fetchall()]
        client = _map_client(settings)
        out: dict[str, Any] = {}
        for variant in args.variants:
            base, *flags = variant.split("+")
            k = int(base.removeprefix("k"))
            adj = M._Adjudicator(tax, client, settings, args.max_usd,
                                 M.MapReport(taxonomy_version="replay"), conn)

            def ask(iid: str, cands: list[str]) -> Any:
                a = M._Adjudication("dimension", phrases[iid], tuple(cands), ranked[iid][0])
                a.contexts.extend(contexts[iid])
                return a
            first = {it["item_id"]: ask(it["item_id"], ranked[it["item_id"]][2][:k])
                     for it in items if ranked[it["item_id"]][2]}
            await asyncio.gather(*(adj._one(a) for a in first.values()))  # _one stores nothing
            answers = {i: a.choice for i, a in first.items()}
            if "second" in flags:
                by_id = {it["item_id"]: it for it in items}
                again = {i: ask(i, rank(_query(by_id[i]), ranked[i][1], set(a.candidates), 5))
                         for i, a in first.items() if a.choice is None and not a.failed}
                again = {i: a for i, a in again.items() if a.candidates}
                await asyncio.gather(*(adj._one(a) for a in again.values()))
                answers.update({i: a.choice for i, a in again.items() if a.choice is not None})
            out[variant] = answers
            print(f"{variant}: {adj.report.llm_calls} calls, {adj.report.llm_invalid} invalid,"
                  f" ${adj.spent:.4f}")
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    finally:
        await conn.close()
    return 0


async def _recall(args: argparse.Namespace) -> int:
    from vendorkb.config import VendorkbSettings
    from vendorkb.db import connect

    items = json.loads(LABELS.read_text())["items"]
    settings = VendorkbSettings()
    conn = await connect(settings.vendorkb_dsn)
    try:
        _, _, ranked, rank = await _setup(conn, settings, items)
    finally:
        await conn.close()
    with_dim = [it for it in items if any(a is not None for a in it["acceptable"])]
    for name, cands in (
            ("embedding", lambda it: ranked[it["item_id"]][2]),
            ("keywords", lambda it: rank(_query(it), ranked[it["item_id"]][1], set(), 60))):
        for k in (5, 8, 10):
            found = sum(1 for it in with_dim if set(it["acceptable"]) & set(cands(it)[:k]))
            print(f"{name:10s} acceptable dimension among the top {k:2d}: {found}/{len(with_dim)}")
    return 0


def _score(args: argparse.Namespace) -> int:
    items = json.loads(LABELS.read_text())["items"]
    assigned = {it["item_id"]: it["assigned_2026_11_08"] for it in items}
    runs = {"mapped 2026-11-08": assigned}
    for path in args.runs:
        runs.update(json.loads(Path(path).read_text()))
    print(f"{'variant':20s} precision        recall  right  unlabelled")
    for name, answers in runs.items():
        s = score(answers, items)
        print(f"{name:20s} {s['precision']:.3f} of {s['answered']:3d}  {s['recall']:.3f}  "
              f"{s['right']:3d}  {s['unlabelled']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="replay variants (paid: MAP_* settings)")
    r.add_argument("out")
    r.add_argument("variants", nargs="+")
    r.add_argument("--max-usd", type=float, required=True)
    sub.add_parser("recall", help="candidate recall of the retrievers (no model call)")
    s = sub.add_parser("score", help="score run outputs against the labels")
    s.add_argument("runs", nargs="*")
    args = ap.parse_args(argv)
    if args.cmd == "score":
        return _score(args)
    return asyncio.run(_run(args) if args.cmd == "run" else _recall(args))


if __name__ == "__main__":
    sys.exit(main())
