"""Topic eval: blind grading items and a scorer for the broad ransomware questions.

    uv run python eval/answer_eval.py --questions eval/topic_questions.yaml run --out RUN.jsonl
    uv run python eval/topic_eval.py build before.jsonl after.jsonl --out items.json --key key.json
    # one fresh grader per grades file, given items.json and eval/TOPIC_GRADING_BRIEF.md only
    uv run python eval/topic_eval.py score key.json opus.json fable.json \
        --before before --after after [--corpus corpus.json]

``build`` makes one item per answer row of every answers file (the run label is the file's
stem), shuffled with ``--seed`` and keyed by an opaque id. ``items.json`` carries no run label,
file name, model name, cost or timing; ``key.json`` maps each item id to ``{run, question_id}``.
An answer that errored or is empty, or one to a question whose text differs from the questions
file, stops the build (exit message names it): there is nothing to grade, or it is not the same
question.

The questions file is ``eval/topic_questions.yaml``: ``id``, ``type``, ``question``, ``topic``
(a topic id of ``taxonomy/taxonomy.yaml``, whose aspects are read from there) and ``sides`` (the
parties graded separately). ``grading`` is ``sides`` (the default when ``sides`` is not empty),
``by_aspect`` (graded across the whole answer) or ``enumeration`` (graded on sampled
vendor-aspect pairs); it is required when ``sides`` is empty, because a vendor-neutral question
and a "which vendors" question share the type ``cross_vendor``.

Grades, one JSON file per grader, a list of objects (``eval/TOPIC_GRADING_BRIEF.md`` defines the
labels). ``sides`` and ``by_aspect`` items (the side key is ``"*"`` for ``by_aspect``)::

    {"item": id, "claims_total": int, "unsupported_claims": [str, ...],
     "aspects": {"<side>": {"<aspect_id>": "covered_supported" | "covered_unsupported"
                                           | "not_covered" | "stated_not_covered"}}}

``enumeration`` items carry pairs, not claims. ``pairs_total`` is the number of distinct
(vendor, aspect) pairs the answer covers; ``pairs`` is an even-stride sample of them in answer
order: every ``ceil(pairs_total / 30)``-th pair starting with the first (all of them up to 30),
with no (vendor, aspect_id) twice. The count is checked (``sample_size``)::

    {"item": id, "pairs_total": int,
     "pairs": [{"vendor": str, "aspect_id": str, "supported": bool}, ...]}

``score`` prints JSON per grades file and run: coverage (``covered_supported`` slots over the
graded side-aspect slots, less the ``stated_not_covered`` slots that ``--corpus`` marks as not
documented) and the unsupported rate, pooled over every item::

    (unsupported claims of sides and by_aspect items
     + sum of pairs_total x (1 - supported share) over enumeration items)
    / (claims_total of sides and by_aspect items + sum of pairs_total over enumeration items)

The enumeration supported share and estimate (per item ``pairs_total`` times the supported share
of its sampled pairs; per run the sum over its enumeration items, null with none) are reported
separately. For two or more grades files it prints the agreement of the first two (the share of
identical aspect labels, with the slots that differ). With exactly two runs and
``--before``/``--after`` it also prints the spec's pass check on the pooled rate: coverage rises
by at least 10 points (this scorer's reading of "clearly") and the unsupported rate by at most
2. A question missing from one run is dropped from the pass check, and the output says which.
The corpus file is ``{"<side>": {"<aspect_id>": true | false}}``, true meaning the corpus
documents it; its sides and aspect ids must exist in the questions file and the taxonomy, and
without it the output warns when ``stated_not_covered`` slots exist. A grades file with a
missing or invalid slot exits 2 and lists every problem.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from vendorkb.taxonomy.load import TaxonomyError, read_doc, validate

REPO = Path(__file__).resolve().parent.parent
TAXONOMY = REPO / "taxonomy" / "taxonomy.yaml"
DEFAULT_QUESTIONS = REPO / "eval" / "topic_questions.yaml"

LABELS = ("covered_supported", "covered_unsupported", "not_covered", "stated_not_covered")
KINDS = ("sides", "by_aspect", "enumeration")
ANY_SIDE = "*"  # the side key of a by_aspect item
MIN_COVERAGE_RISE = 10.0  # points: this scorer's reading of the spec's "rises clearly"
MAX_UNSUPPORTED_RISE = 2.0  # points: the spec's "by no more than two points"
MAX_PAIRS = 30  # an enumeration item lists at most this many of its pairs
EPSILON = 1e-6


class GradeError(ValueError):
    """A grades file that cannot be scored; ``problems`` lists every defect found."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


def _norm(name: Any) -> str:
    return " ".join(str(name).split()).casefold()


def _ratio(n: float, d: float) -> float | None:
    return n / d if d else None


def _round(value: Any) -> Any:
    """Floats to four places, for printing."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


# ---- questions and aspects ------------------------------------------------------------------

def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """The questions in file order, each with ``sides`` (a list) and ``grading`` (one of
    ``KINDS``); every defect is reported at once."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: not a YAML list")
    problems: list[str] = []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for q in raw:
        qid = str(q.get("id"))
        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)
        for field in ("question", "topic"):
            if not q.get(field):
                problems.append(f"{qid}: no {field}")
        sides = [str(s) for s in q.get("sides") or []]
        grading = q.get("grading") or ("sides" if sides else None)
        if grading not in KINDS:
            problems.append(f"{qid}: grading must be one of {', '.join(KINDS)}"
                            " (a question with no sides needs it)")
        elif grading == "sides" and not sides:
            problems.append(f"{qid}: grading sides needs a non-empty sides list")
        elif grading != "sides" and sides:
            problems.append(f"{qid}: sides are for grading sides, not {grading}")
        out.append({**q, "id": qid, "sides": sides, "grading": grading})
    if problems:
        raise SystemExit("\n".join(problems))
    return out


def load_aspects(topic_ids: list[str],
                 taxonomy: str | Path = TAXONOMY) -> dict[str, list[dict[str, str]]]:
    """``{topic id: [{id, name, description}]}`` from the taxonomy file, aspects in file order;
    ``description`` is the aspect's retrieval phrasing, which says what it covers."""
    try:
        topics = {t.id: t for t in validate(read_doc(taxonomy)).topics}
    except TaxonomyError as e:
        raise SystemExit(f"{taxonomy}: {e}") from e
    unknown = sorted(set(topic_ids) - topics.keys())
    if unknown:
        raise SystemExit(f"{taxonomy}: no such topic: {', '.join(unknown)}")
    return {tid: [{"id": a.id, "name": a.name, "description": a.query}
                  for a in topics[tid].aspects] for tid in sorted(set(topic_ids))}


# ---- build ----------------------------------------------------------------------------------

def read_answers(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{n}: not JSON ({e})") from e
    return rows


def runs_from_files(paths: list[str | Path]) -> dict[str, list[dict[str, Any]]]:
    """``{run label: answer rows}``; the label is the file's stem, so stems must differ."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        label = Path(path).stem
        if label in runs:
            raise SystemExit(f"two answers files are called {label}: the run label is the stem")
        rows = read_answers(path)
        ids = Counter(r.get("id") for r in rows)
        twice = sorted(str(i) for i, n in ids.items() if n > 1)
        if twice:
            raise SystemExit(f"{path}: {twice[0]} twice")
        runs[label] = rows
    return runs


def _marker_order(citation: dict[str, Any]) -> tuple[int, str]:
    marker = str(citation.get("marker"))
    return (int(marker), "") if marker.isdigit() else (1 << 30, marker)


def build_items(runs: dict[str, list[dict[str, Any]]], questions: list[dict[str, Any]],
                aspects: dict[str, list[dict[str, str]]], seed: int = 1
                ) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """``(items, key)``: one blind item per answer row, shuffled; ``key`` maps item id to
    ``{run, question_id}``. The ids are drawn from the same seeded generator after the shuffle,
    so they say nothing about the run or the question and cannot be recomputed from either."""
    by_id = {q["id"]: q for q in questions}
    problems: list[str] = []
    entries: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for run, rows in runs.items():
        for row in rows:
            qid = row.get("id")
            question = by_id.get(qid)
            if question is None:
                problems.append(f"{run}: {qid}: not in the questions file")
            elif row.get("error") or not str(row.get("answer") or "").strip():
                problems.append(f"{run}: {qid}: no answer (error: {row.get('error')})")
            elif row.get("question") not in (None, question["question"]):
                problems.append(f"{run}: {qid}: the question asked differs from the questions"
                                " file")
            else:
                entries.append((run, row, question))
    if problems:
        raise SystemExit("\n".join(problems))
    rng = random.Random(seed)
    rng.shuffle(entries)
    items: list[dict[str, Any]] = []
    key: dict[str, dict[str, str]] = {}
    for run, row, question in entries:
        item_id = f"{rng.getrandbits(32):08x}"
        while item_id in key:
            item_id = f"{rng.getrandbits(32):08x}"
        key[item_id] = {"run": run, "question_id": question["id"]}
        sources = [{"marker": c.get("marker"), "title": c.get("title"), "vendor": c.get("vendor"),
                    "url": c.get("url"), "text": c.get("text")}
                   for c in sorted(row.get("citations") or [], key=_marker_order)]
        items.append({"item": item_id, "kind": question["grading"],
                      "question": question["question"], "sides": question["sides"],
                      "aspects": aspects[question["topic"]], "answer": row["answer"],
                      "sources": sources})
    return items, key


# ---- grades ---------------------------------------------------------------------------------

def _count_problem(item: str, grade: dict[str, Any], problems: list[str]) -> int | None:
    raw = grade.get("claims_total")
    total = raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else None
    if total is None:
        problems.append(f"{item}: claims_total must be a non-negative integer")
    unsupported = grade.get("unsupported_claims")
    if not isinstance(unsupported, list) or not all(isinstance(c, str) for c in unsupported):
        problems.append(f"{item}: unsupported_claims must be a list of strings")
    elif total is not None and len(unsupported) > total:
        problems.append(f"{item}: more unsupported claims than claims_total")
    return total


def _grade_sides(item: str, grade: dict[str, Any], question: dict[str, Any],
                 aspect_ids: list[str], problems: list[str]) -> dict[tuple[str, str], str]:
    expected = question["sides"] if question["grading"] == "sides" else [ANY_SIDE]
    canonical = {_norm(s): s for s in expected}
    given = grade.get("aspects")
    if not isinstance(given, dict):
        problems.append(f"{item}: aspects must be an object of side -> aspect -> label")
        return {}
    slots: dict[tuple[str, str], str] = {}
    seen: set[str] = set()
    for raw_side, labels in given.items():
        side = canonical.get(_norm(raw_side))
        if side is None:
            problems.append(f"{item}: unknown side {raw_side!r}")
            continue
        if side in seen:
            problems.append(f"{item}: side {side} graded twice")
            continue
        seen.add(side)
        if not isinstance(labels, dict):
            problems.append(f"{item}: {side}: expected an object of aspect -> label")
            continue
        for aspect, label in labels.items():
            if aspect not in aspect_ids:
                problems.append(f"{item}: {side}: unknown aspect {aspect!r}")
            elif label not in LABELS:
                problems.append(f"{item}: {side}: {aspect}: unknown label {label!r}")
            else:
                slots[(side, aspect)] = label
        problems.extend(f"{item}: {side}: aspect {a} not graded" for a in aspect_ids
                        if a not in labels)
    problems.extend(f"{item}: side {s} not graded" for s in expected if s not in seen)
    return slots


def pair_stride(pairs_total: int) -> int:
    """The stride of the enumeration sample: every this-many-th pair, starting with the first."""
    return max(1, -(-pairs_total // MAX_PAIRS))


def sample_size(pairs_total: int) -> int:
    """How many pairs the even-stride sample of ``pairs_total`` holds: all of them up to
    ``MAX_PAIRS``, else every ``pair_stride``-th one (so 20 of 40, 30 of 60, 25 of 100)."""
    return -(-pairs_total // pair_stride(pairs_total))


def _grade_pairs(item: str, grade: dict[str, Any], aspect_ids: list[str],
                 problems: list[str]) -> tuple[int | None, list[dict[str, Any]]]:
    total = grade.get("pairs_total")
    pairs = grade.get("pairs")
    if not (isinstance(total, int) and not isinstance(total, bool) and total >= 0):
        problems.append(f"{item}: pairs_total must be a non-negative integer")
        total = None
    if not isinstance(pairs, list):
        problems.append(f"{item}: pairs must be a list")
        return total, []
    good = []
    listed: set[tuple[str, str]] = set()
    for n, pair in enumerate(pairs):
        if (not isinstance(pair, dict) or not isinstance(pair.get("vendor"), str)
                or pair.get("aspect_id") not in aspect_ids
                or not isinstance(pair.get("supported"), bool)):
            problems.append(f"{item}: pair {n} needs a vendor, an aspect_id of the item and a "
                            "boolean supported")
            continue
        who = (_norm(pair["vendor"]), pair["aspect_id"])
        if who in listed:
            problems.append(f"{item}: pair {n} repeats {pair['vendor']} / {pair['aspect_id']}")
        listed.add(who)
        good.append(pair)
    if total is not None and len(pairs) != sample_size(total):
        stride = pair_stride(total)
        how = (f"stride {stride} over pairs_total {total}, starting with the first pair"
               if stride > 1 else f"all of pairs_total {total}")
        problems.append(f"{item}: {len(pairs)} pairs listed, expected {sample_size(total)} "
                        f"({how})")
    return total, good


def parse_grades(grades: Any, key: dict[str, dict[str, str]],
                 questions: list[dict[str, Any]],
                 aspects: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    """The validated records of one grades file, one per graded item. Every item of ``key``
    must be graded, every side and aspect of a ``sides``/``by_aspect`` item labelled."""
    by_id = {q["id"]: q for q in questions}
    problems: list[str] = []
    records: list[dict[str, Any]] = []
    if not isinstance(grades, list):
        raise GradeError(["the grades file must be a list of objects"])
    seen: set[str] = set()
    for n, grade in enumerate(grades):
        if not isinstance(grade, dict):
            problems.append(f"entry {n}: not an object")
            continue
        item = grade.get("item")
        if not isinstance(item, str) or item not in key:
            problems.append(f"{item!r}: not in the key" if not isinstance(item, str)
                            else f"{item}: not in the key")
            continue
        if item in seen:
            problems.append(f"{item}: graded twice")
            continue
        seen.add(item)
        question = by_id[key[item]["question_id"]]
        aspect_ids = [a["id"] for a in aspects[question["topic"]]]
        record: dict[str, Any] = {"item": item, "run": key[item]["run"],
                                  "question_id": question["id"], "kind": question["grading"],
                                  "slots": {}, "pairs_total": None, "pairs": [],
                                  "claims_total": 0, "unsupported": []}
        if question["grading"] == "enumeration":
            record["pairs_total"], record["pairs"] = _grade_pairs(item, grade, aspect_ids,
                                                                   problems)
        else:
            record["claims_total"] = _count_problem(item, grade, problems)
            unsupported = grade.get("unsupported_claims")
            record["unsupported"] = list(unsupported) if isinstance(unsupported, list) else []
            record["slots"] = _grade_sides(item, grade, question, aspect_ids, problems)
        records.append(record)
    problems.extend(f"{item}: not graded" for item in key if item not in seen)
    if problems:
        raise GradeError(problems)
    return records


# ---- score ----------------------------------------------------------------------------------

def load_corpus(path: str | Path) -> dict[str, dict[str, bool]]:
    """``{side: {aspect id: documented?}}``, as written in the file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not (isinstance(raw, dict) and all(
            isinstance(v, dict) and all(isinstance(b, bool) for b in v.values())
            for v in raw.values())):
        raise SystemExit(f"{path}: expected {{side: {{aspect_id: true|false}}}}")
    return raw


def _is_excluded(side: str, aspect: str, label: str,
                 corpus: dict[str, dict[str, bool]]) -> bool:
    """A stated_not_covered slot the corpus does not document is not held against the answer.
    ``corpus`` has normalised side names."""
    return label == "stated_not_covered" and corpus.get(_norm(side), {}).get(aspect) is False


def _coverage(slots: list[tuple[str, str, str]],
              corpus: dict[str, dict[str, bool]]) -> dict[str, Any]:
    excluded = sum(_is_excluded(side, aspect, label, corpus) for side, aspect, label in slots)
    supported = sum(label == "covered_supported" for _, _, label in slots)
    denominator = len(slots) - excluded
    return {"slots": len(slots), "excluded": excluded, "denominator": denominator,
            "covered_supported": supported, "coverage": _ratio(supported, denominator)}


def _enumeration(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_item = []
    for r in records:
        supported = sum(p["supported"] for p in r["pairs"])
        share = _ratio(supported, len(r["pairs"]))
        total = r["pairs_total"]
        estimate = 0.0 if not total else None if share is None else total * share
        unsupported = 0.0 if not total else None if share is None else total * (1 - share)
        per_item.append({"item": r["item"], "question_id": r["question_id"],
                         "pairs_total": total, "sampled": len(r["pairs"]),
                         "supported": supported, "share": share, "estimate": estimate,
                         "unsupported_estimate": unsupported})

    def total_of(name: str) -> float | None:
        values = [i[name] for i in per_item]
        return None if not values or any(v is None for v in values) else sum(values)

    sampled = sum(i["sampled"] for i in per_item)
    supported = sum(i["supported"] for i in per_item)
    return {"items": len(records), "pairs_total": sum(i["pairs_total"] for i in per_item),
            "sampled": sampled, "supported": supported, "share": _ratio(supported, sampled),
            "estimate": total_of("estimate"), "unsupported_estimate": total_of(
                "unsupported_estimate"), "per_item": per_item}


def score_records(records: list[dict[str, Any]],
                  corpus: dict[str, dict[str, bool]]) -> dict[str, dict[str, Any]]:
    """The metrics of each run in ``records``: coverage, labels, the pooled unsupported rate,
    the enumeration estimate and a per-question table. The rate is the unsupported claims of
    ``sides`` and ``by_aspect`` items plus the estimated unsupported pairs of ``enumeration``
    items, over those items' ``claims_total`` plus the enumeration ``pairs_total`` (None when
    an enumeration item has no sampled pair to estimate from). ``corpus`` sides are matched
    without regard to case and spacing."""
    corpus = {_norm(side): aspects for side, aspects in corpus.items()}
    out: dict[str, dict[str, Any]] = {}
    for run in dict.fromkeys(r["run"] for r in records):
        mine = [r for r in records if r["run"] == run]
        graded = [r for r in mine if r["kind"] != "enumeration"]
        enumerated = [r for r in mine if r["kind"] == "enumeration"]
        enum = _enumeration(enumerated)

        def slots_of(rs: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
            return [(side, aspect, label) for r in rs
                    for (side, aspect), label in r["slots"].items()]

        claims = sum(r["claims_total"] for r in graded)
        unsupported = sum(len(r["unsupported"]) for r in graded)
        counts = Counter(label for _, _, label in slots_of(graded))
        by_question: dict[str, dict[str, Any]] = {}
        for r in graded:
            by_question[r["question_id"]] = {
                **_coverage(slots_of([r]), corpus), "claims_total": r["claims_total"],
                "unsupported_claims": len(r["unsupported"])}
        for row in enum["per_item"]:
            by_question[row["question_id"]] = {k: row[k] for k in ("pairs_total", "share",
                                                                   "estimate")}
        pair_unsupported = 0.0 if not enumerated else enum["unsupported_estimate"]
        rate = (None if pair_unsupported is None else
                _ratio(unsupported + pair_unsupported, claims + enum["pairs_total"]))
        out[run] = {"items": len(mine), "coverage": _coverage(slots_of(graded), corpus),
                    "labels": {label: counts[label] for label in LABELS},
                    "unsupported": {"claims_total": claims, "unsupported_claims": unsupported,
                                    "pairs_total": enum["pairs_total"],
                                    "unsupported_pairs_estimate": pair_unsupported,
                                    "rate": rate},
                    "enumeration": enum, "by_question": by_question}
    return out


def agreement(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    """The share of identical aspect labels over the slots both grades files graded, and the
    slots they differ on (item ids only, so adjudication can stay blind)."""
    other = {r["item"]: r for r in second}
    slots = identical = 0
    disagreements = []
    for r in first:
        theirs = other.get(r["item"])
        if theirs is None:
            continue
        for (side, aspect), label in r["slots"].items():
            if (side, aspect) not in theirs["slots"]:
                continue
            slots += 1
            if label == theirs["slots"][(side, aspect)]:
                identical += 1
            else:
                disagreements.append({"item": r["item"], "side": side, "aspect": aspect,
                                      "labels": [label, theirs["slots"][(side, aspect)]]})
    disagreements.sort(key=lambda d: (d["item"], d["side"], d["aspect"]))
    return {"slots": slots, "identical": identical, "share": _ratio(identical, slots),
            "disagreements": disagreements}


def pass_check(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """The spec's bar: coverage rises by at least ``MIN_COVERAGE_RISE`` points and the pooled
    unsupported rate by at most ``MAX_UNSUPPORTED_RISE``. A rate that cannot be computed (no
    claims in one run) fails the check."""
    def points(metric: str, name: str) -> float | None:
        a, b = before[metric][name], after[metric][name]
        return None if a is None or b is None else round((b - a) * 100, 6)

    cov = points("coverage", "coverage")
    rate = points("unsupported", "rate")
    cov_ok = None if cov is None else cov >= MIN_COVERAGE_RISE - EPSILON
    rate_ok = None if rate is None else rate <= MAX_UNSUPPORTED_RISE + EPSILON
    return {"bar": f"coverage rises by at least {MIN_COVERAGE_RISE:g} points (\"clearly\") and "
                   f"the pooled unsupported rate by at most {MAX_UNSUPPORTED_RISE:g} points",
            "coverage": {"before": before["coverage"]["coverage"],
                         "after": after["coverage"]["coverage"]},
            "unsupported_rate": {"before": before["unsupported"]["rate"],
                                 "after": after["unsupported"]["rate"]},
            "coverage_rise_points": cov, "coverage_ok": cov_ok,
            "unsupported_rise_points": rate, "unsupported_ok": rate_ok,
            "pass": bool(cov_ok and rate_ok)}


def check_corpus(corpus: dict[str, dict[str, bool]], questions: list[dict[str, Any]],
                 aspects: dict[str, list[dict[str, str]]]) -> None:
    """Every side of ``corpus`` must be one a question grades (or ``"*"``, the vendor-neutral
    side) and every aspect id one of that side's topic; else a ``GradeError``."""
    known: dict[str, set[str]] = {}
    for q in questions:
        sides = q["sides"] if q["grading"] == "sides" else [ANY_SIDE] if (
            q["grading"] == "by_aspect") else []
        for side in sides:
            known.setdefault(_norm(side), set()).update(a["id"] for a in aspects[q["topic"]])
    problems = []
    for side, flags in corpus.items():
        if _norm(side) not in known:
            problems.append(f"corpus: unknown side {side!r}")
            continue
        problems.extend(f"corpus: {side}: unknown aspect {a!r}" for a in flags
                        if a not in known[_norm(side)])
    if problems:
        raise GradeError(problems)


def report_for(records: list[dict[str, Any]], corpus: dict[str, dict[str, bool]] | None,
               before: str | None, after: str | None) -> dict[str, Any]:
    """The metrics of one grades file, a pass check, and ``warnings``. ``corpus`` None means
    no corpus file was given; the pass check compares only the questions both runs have."""
    runs = score_records(records, corpus or {})
    warnings: list[str] = []
    stated = sum(label == "stated_not_covered" for r in records for label in r["slots"].values())
    if stated and corpus is None:
        warnings.append(f"{stated} stated_not_covered slots count against coverage: no --corpus "
                        "file was given to excuse the aspects a side's documents lack")
    if len(runs) != 2:
        check: dict[str, Any] = {"skipped": f"the pass check needs exactly two runs, found "
                                            f"{len(runs)}"}
    elif before is None or after is None:
        check = {"skipped": "give --before and --after to run the pass check"}
    else:
        missing = [label for label in (before, after) if label not in runs]
        if missing:
            raise SystemExit(f"no such run: {', '.join(missing)} (runs: {', '.join(runs)})")
        asked = {run: {r["question_id"] for r in records if r["run"] == run}
                 for run in (before, after)}
        common = asked[before] & asked[after]
        dropped = sorted(asked[before] ^ asked[after])
        warnings.extend(f"pass check: {qid} has no graded answer in run "
                        f"{after if qid in asked[before] else before}; left out of the "
                        "comparison" for qid in dropped)
        compared = runs
        if dropped:
            compared = score_records([r for r in records if r["question_id"] in common],
                                     corpus or {})
        if not common:
            check = {"skipped": "the two runs have no question in common"}
        else:
            check = {"before": before, "after": after, "questions_compared": sorted(common),
                     "questions_dropped": dropped,
                     **pass_check(compared[before], compared[after])}
    return {"runs": runs, "pass_check": check, "warnings": warnings}


# ---- command line ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions)
    aspects = load_aspects([q["topic"] for q in questions], args.taxonomy)
    runs = runs_from_files(args.answers)
    items, key = build_items(runs, questions, aspects, args.seed)
    graded = {(v["run"], v["question_id"]) for v in key.values()}
    for run in runs:
        absent = [q["id"] for q in questions if (run, q["id"]) not in graded]
        if absent:
            print(f"warning: {run} has no answer for {', '.join(absent)}", file=sys.stderr)
    Path(args.out).write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    Path(args.key).write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    print(f"{len(items)} items from {len(runs)} runs -> {args.out}; key -> {args.key}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions)
    aspects = load_aspects([q["topic"] for q in questions], args.taxonomy)
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    unknown = sorted({v["question_id"] for v in key.values()} - {q["id"] for q in questions})
    if unknown:
        raise SystemExit(f"the key names questions not in {args.questions}: {', '.join(unknown)}")
    corpus = load_corpus(args.corpus) if args.corpus else None
    if corpus is not None:
        try:
            check_corpus(corpus, questions, aspects)
        except GradeError as e:
            print(f"{args.corpus}:", *e.problems, sep="\n  ", file=sys.stderr)
            return 2
    parsed: dict[str, list[dict[str, Any]]] = {}
    failed = False
    for path in args.grades:
        try:
            parsed[path] = parse_grades(json.loads(Path(path).read_text(encoding="utf-8")),
                                        key, questions, aspects)
        except GradeError as e:
            failed = True
            print(f"{path}:", *e.problems, sep="\n  ", file=sys.stderr)
    if failed:
        return 2
    report: dict[str, Any] = {"grades": {
        path: report_for(records, corpus, args.before, args.after)
        for path, records in parsed.items()}}
    if len(parsed) >= 2:
        first, second = list(parsed)[:2]
        report["agreement"] = {"files": [first, second],
                               **agreement(parsed[first], parsed[second])}
    print(json.dumps(_round(report), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="blind grading items from answers files")
    b.add_argument("answers", nargs="+", help="answers JSONL, one file per run (label = stem)")
    b.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    b.add_argument("--taxonomy", default=str(TAXONOMY))
    b.add_argument("--out", required=True, help="the blind items, for the graders")
    b.add_argument("--key", required=True, help="item id -> run and question id; not for graders")
    b.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("score", help="coverage, unsupported rate, enumeration and the pass check")
    s.add_argument("key")
    s.add_argument("grades", nargs="+", help="one grades JSON per grader")
    s.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    s.add_argument("--taxonomy", default=str(TAXONOMY))
    s.add_argument("--corpus", help='{"<side>": {"<aspect_id>": true|false}}, true = documented')
    s.add_argument("--before", help="run label of the baseline, for the pass check")
    s.add_argument("--after", help="run label of the run under test, for the pass check")
    args = ap.parse_args(argv)
    return cmd_build(args) if args.cmd == "build" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
