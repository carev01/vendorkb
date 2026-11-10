"""Three-arm agent eval (BACKLOG 42): does vendorkb need its own answer model?

    uv run python eval/bypass_eval.py prepare --out eval/runs/bypass-2026-11-10
    # one fresh agent per brief (eval/runs/.../briefs/*.md), nothing else in its prompt
    uv run python eval/bypass_eval.py items eval/runs/bypass-2026-11-10
    # one fresh grader per batch (grading/batch-*.json) and grader model, given the batch and
    # eval/BYPASS_GRADING_BRIEF.md only; each writes grading/grades-<grader>-<batch>.json
    uv run python eval/bypass_eval.py score eval/runs/bypass-2026-11-10

Every question is answered three times, each by a fresh agent with one set of tools:

- ``relay``: ``answer_question`` only; the agent relays vendorkb's written answer (today's
  Copilot agent, docs/deploy/copilot-studio.md §5).
- ``evidence``: ``gather_evidence`` (and ``read_section``); the agent writes the answer itself
  from the numbered passages.
- ``raw``: the search and capability tools; the agent gathers and writes (``agent_eval``'s arm
  ``full``).

An agent writes its reply as the user would read it (``results/<arm>-<id>.md``) and the vendors
it says support what a "which vendors" question asks (``results/<arm>-<id>.json``). The agents
run ``agent_eval.py call``, which logs every tool call with its result.

``items`` draws up to ``--statements`` cited statements from every reply (an even stride over
the sentences that carry a ``[N]``), resolves each marker to the chunk it cites and fetches the
chunk's whole text from the MCP server (``read_section``, no rerank). ``items.json`` carries no
arm; ``key.json`` maps each item to its arm and question. ``score`` reports, per arm: vendor
recall and precision on the labelled "which vendors" questions, whether the question reached
vendorkb word for word, the share of sentences that carry a marker, the share of markers that
resolve to a passage a tool returned, and the graders' unsupported and uncited claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.agent_eval import ARMS, REPO, mcp_call, score_answer  # noqa: E402

# arm -> the agent_eval arm whose tools it may call
ARM_TOOLS = {"relay": "relay", "evidence": "evidence", "raw": "full"}
# The lean pilot (2026-11-10): six "which vendors" questions (the largest and weakest, a
# precision-weak one, and three between), three comparisons and three topic questions.
PILOT = ["ex-rt-10", "cv-01", "cv-07", "cv-08", "cv43-04", "cv43-10",
         "ex-rt-07", "cv-12", "cv-14", "u-01", "tp-01", "tp-07"]
# The user's question of BACKLOG 40, in no labelled file.
EXTRA = [{"id": "u-01", "shape": "comparison", "topic": "ransomware_protection",
          "question": "Compare ransomware protection capabilities between Veeam and Commvault"}]
VERBATIM = ("Pass it the user's question word for word: add no instructions, detail, examples "
            "or formatting requests.")
NEVER = ("Never add a fact, vendor, product, version or number from general knowledge that the "
         "tools did not return.")
NAMES = "Keep vendor and product names exactly as vendorkb gives them."
RULES = {
    "relay": [
        f"Answer with answer_question. {VERBATIM} Relay its answer and its Sources list "
        "unchanged: keep every [N] marker and every source link.",
        NEVER,
        "When answer_question refuses or finds nothing, say that it is not found in the "
        "documentation vendorkb covers. Do not guess.",
        NAMES],
    "evidence": [
        f"Answer with gather_evidence. {VERBATIM} When the reply says \"Part 1 of N\", call it "
        "again with part 2 … N (same question) before you answer. Write the answer from the "
        "numbered passages only. Put the [N] of the passage after every statement it supports, "
        "and end with a Sources list: `[N] title — link` for each [N] used. State for a vendor "
        "only what its own passages show.",
        NEVER,
        "When gather_evidence finds no passage, or none for a vendor or aspect, say that it is "
        "not found in the documentation vendorkb covers; never say the product lacks it. Do "
        "not guess.",
        NAMES],
    "raw": [
        "Every factual statement in your reply must cite a source the tools returned: number "
        "your sources, put [N] after each statement, and end with a Sources list: `[N] title — "
        "link`. If nothing supports a statement, leave it out. Never invent a link.",
        "For a question about which vendors support something, run list_vendors, call "
        "query_capabilities first, then check the vendors it does not list with search_docs: a "
        "vendor missing from the table does not mean the vendor lacks the feature.",
        "For a broad subject such as ransomware protection, query_capabilities with `topic` "
        "returns facts grouped by aspect. For one vendor's details, use search_docs with the "
        "vendor filter, then read_section for the surrounding text.",
        NEVER,
        NAMES],
}
TOOL_HELP = {
    "relay": "`answer_question` takes `question`.",
    "evidence": ("`gather_evidence` takes `question` and `part` (a number); `read_section` takes "
                 "`chunk_id` and `expand` \"0\"/\"1\"/\"full\", to read a passage in its section."),
    "raw": ("`list_vendors` takes no arguments; `search_docs` takes `query`, optional `vendor` "
            "(a list of vendor names) and `k`; `read_section` takes `chunk_id` and `expand` "
            "\"0\"/\"1\"/\"full\"; `query_capabilities` takes taxonomy filters such as `dimension`, "
            "`target`, `vendor`, `support`, or `topic`; `list_dimensions` / `list_targets` take "
            "`query`. You have at most {max_searches} `search_docs` calls; the command refuses "
            "more."),
}
BRIEF = """# Agent brief: answer one question with the vendorkb tools

You are an assistant that answers a user's question about backup products, using only the
vendorkb tools. Call a tool from a shell with:

    {command} TOOL '{{"arg": "value"}}'

Tools available to you: {tools}. {tool_help}

Rules:
{rules}

When you have your reply:
1. Write it, exactly as the user should read it (markdown), to {md} with your file-writing
   tool.
2. Write {json} with a short Python script and json.dump:
   `{{"question_id": "{qid}", "supported": ["<vendor>", ...]}}`. `supported` lists every vendor
   your reply says supports what the question asks about, spelled as the tools spell it. Use an
   empty list when the question does not ask which vendors do something.

Open or list no file other than this brief, and run no command other than the one above and the
script that writes the JSON. Then reply to me with just "done" and the number of tool calls you
made.

## The user's question (id {qid})

{question}
"""

MARKERS = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SOURCES_HEAD = re.compile(r"^\s*(?:#+\s*)?\**\s*sources?\s*:?\**\s*:?\s*$", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s<>()\[\]]+")
_CHUNK = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
# "[3] title — url (chunk id)" (answer_question's Sources) and "[3] head\n    source: url (chunk
# id)" (gather_evidence, search_docs)
_TOOL_SOURCE = re.compile(rf"^\[(\d+)\] [^\n]*? — (\S+) \(chunk ({_CHUNK})\)\s*$"
                          rf"|^\[(\d+)\] [^\n]*\n\s+source: (\S+) \(chunk ({_CHUNK})", re.MULTILINE)
_ANY_SOURCE = re.compile(rf"(https?://\S+?)[)\s][^\n]*?\(?chunk ({_CHUNK})|"
                         rf"(https?://\S+) \(chunk ({_CHUNK})")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\[(\"`])")
SOURCE_CHARS = 6000  # of a cited chunk's text shown to a grader
BATCH_CHARS = 220_000  # of items in one grader's batch


def load_set(ids: list[str]) -> list[dict[str, Any]]:
    """The questions ``ids`` name, in that order, from the labelled files and ``EXTRA``: each
    ``{id, question, kind}``, ``kind`` "vendors" for a labelled "which vendors" question (with
    its ``expected_vendors`` and ``unclear_vendors``), else "other"."""
    pool: dict[str, dict[str, Any]] = {q["id"]: q for q in EXTRA}
    for name in ("questions.yaml", "topic_questions.yaml"):
        for q in yaml.safe_load((REPO / "eval" / name).read_text()):
            pool.setdefault(q["id"], q)
    if unknown := [i for i in ids if i not in pool]:
        raise SystemExit(f"unknown question id: {', '.join(unknown)}")
    out = []
    for i in ids:
        q = pool[i]
        row = {"id": i, "question": " ".join(q["question"].split()),
               "kind": "vendors" if q.get("expected_vendors") else "other"}
        if row["kind"] == "vendors":
            row.update(expected_vendors=q["expected_vendors"],
                       unclear_vendors=q.get("unclear_vendors") or [])
        out.append(row)
    return out


def brief(arm: str, q: dict[str, Any], out: Path, max_searches: int) -> str:
    name = f"{arm}-{q['id']}"
    command = (f"cd {REPO} && CALL_LOG={out / 'logs' / name}.jsonl ARM={ARM_TOOLS[arm]}"
               f" MAX_SEARCHES={max_searches} {sys.executable} eval/agent_eval.py call")
    return BRIEF.format(
        command=command, tools=", ".join(f"`{t}`" for t in ARMS[ARM_TOOLS[arm]]),
        tool_help=TOOL_HELP[arm].format(max_searches=max_searches),
        rules="\n".join(f"{n}. {r}" for n, r in enumerate(RULES[arm], start=1)),
        md=out / "results" / f"{name}.md", json=out / "results" / f"{name}.json",
        qid=q["id"], question=q["question"])


def _prepare(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    for sub in ("briefs", "results", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    questions = load_set(args.ids or PILOT)
    for q in questions:
        for arm in args.arms:
            (out / "briefs" / f"{arm}-{q['id']}.md").write_text(
                brief(arm, q, out, args.max_searches))
    manifest = {"arms": args.arms, "max_searches": args.max_searches,
                "questions": [{k: q[k] for k in ("id", "question", "kind")} for q in questions],
                "label": args.label}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(questions) * len(args.arms)} briefs in {out / 'briefs'}")
    return 0


# --- reading a reply ---------------------------------------------------------------------------

def split_reply(reply: str) -> tuple[str, dict[int, str]]:
    """(the body, the reply's own Sources as marker -> URL). The Sources section starts at the
    last line that is just "Sources" (as a heading, bold or with a colon); a source line is
    ``[N] …`` with a URL in it. A reply without the section is all body."""
    lines = reply.splitlines()
    heads = [i for i, line in enumerate(lines) if _SOURCES_HEAD.match(line)]
    if not heads:
        return reply, {}
    sources: dict[int, str] = {}
    for line in lines[heads[-1] + 1:]:
        m = re.match(r"\s*(?:[-*]\s*)?\[(\d+)\]", line)
        url = _URL.search(line)
        if m and url:
            sources.setdefault(int(m.group(1)), url.group().rstrip(".,;"))
    return "\n".join(lines[:heads[-1]]), sources


def _numbers(text: str) -> list[int]:
    return list(dict.fromkeys(int(n) for m in MARKERS.finditer(text)
                              for n in re.split(r"\s*,\s*", m.group(1))))


def sentences(body: str) -> list[str]:
    """The body's sentences that could state something: each line split at sentence ends,
    without headings, table rules and lines of under four words. A line whose markers follow
    its last full stop is one statement, cited as a whole."""
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or re.fullmatch(r"[|\s:-]+", line):
            continue
        if re.fullmatch(r"\*\*[^*]+\*\*:?", line):  # a bold line used as a heading
            continue
        line = re.sub(r"^(?:[-*+]|\d+\.)\s+", "", line)
        parts = _SENTENCE.split(line)
        if len(parts) > 1 and not re.sub(r"[\s.,;]", "", MARKERS.sub("", parts[-1])):
            parts = [line]  # "One. Two. [3]": the marker after the last stop cites the whole item
        for part in parts:
            if len(MARKERS.sub("", part).split()) >= 4:
                out.append(part.strip())
    return out


def stride(items: list[Any], n: int) -> list[Any]:
    """Up to ``n`` of ``items``, evenly spaced, the first always among them."""
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[math.floor(i * step)] for i in range(n)]


def tool_sources(log: list[dict[str, Any]], arm: str) -> tuple[dict[int, tuple[str, str]],
                                                               dict[str, list[str]]]:
    """From an agent's tool results: (marker -> (URL, chunk id) as answer_question or
    gather_evidence numbered them, nothing for the raw arm, whose tools number each result from
    1; URL -> the chunk ids shown with it by any tool)."""
    numbered: dict[int, tuple[str, str]] = {}
    by_url: dict[str, list[str]] = {}
    for call in log:
        text = call.get("result") or ""
        if arm != "raw" and call.get("tool") in ("answer_question", "gather_evidence"):
            for m in _TOOL_SOURCE.finditer(text):
                n, url, chunk = (m.group(1), m.group(2), m.group(3)) if m.group(1) else (
                    m.group(4), m.group(5), m.group(6))
                numbered.setdefault(int(n), (url, chunk))
        for m in _ANY_SOURCE.finditer(text):
            url, chunk = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            chunks = by_url.setdefault(url.rstrip(".,;"), [])
            if chunk not in chunks:
                chunks.append(chunk)
    return numbered, by_url


def resolve(marker: int, own: dict[int, str], numbered: dict[int, tuple[str, str]],
            by_url: dict[str, list[str]]) -> tuple[str | None, list[str]]:
    """(URL, chunk ids) a reply's ``[marker]`` cites: the reply's own Sources line if it has
    one (the chunk the tool numbered so when the URL agrees, else the chunks any tool showed
    for that URL, two at most), otherwise the tool's numbering. No chunk: the marker cites
    nothing a tool returned."""
    if marker in own:
        url = own[marker]
        if marker in numbered and numbered[marker][0] == url:
            return url, [numbered[marker][1]]
        return url, by_url.get(url, [])[:2]
    if marker in numbered:
        return numbered[marker][0], [numbered[marker][1]]
    return None, []


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reply_facts(reply: str, log: list[dict[str, Any]], arm: str,
                question: str) -> dict[str, Any]:
    """What code can tell of one reply: its sentences and the share carrying a marker, its
    markers and the share citing a passage a tool returned, the markers of answer_question's
    text it kept (relay), and whether the question reached vendorkb word for word."""
    body, own = split_reply(reply)
    numbered, by_url = tool_sources(log, arm)
    sents = sentences(body)
    cited = [s for s in sents if MARKERS.search(s)]
    used = _numbers(body)
    resolved = [n for n in used if resolve(n, own, numbered, by_url)[1]]
    asked = [c["arguments"].get("question") for c in log
             if c.get("tool") in ("answer_question", "gather_evidence")]
    out: dict[str, Any] = {
        "sentences": len(sents), "sentences_cited": len(cited), "markers": len(used),
        "markers_resolved": len(resolved), "has_sources_list": bool(own),
        "tool_calls": len(log), "tool_seconds": round(sum(c.get("seconds") or 0 for c in log), 1),
        "verbatim": None if not asked else all(
            " ".join((a or "").split()) == " ".join(question.split()) for a in asked),
        "asked": asked[:1]}
    if arm == "relay":
        given = _numbers(split_reply(next(
            (c.get("result") or "" for c in log if c.get("tool") == "answer_question"), ""))[0])
        out["markers_given"] = len(given)
        out["markers_kept"] = len(set(given) & set(used))
    return out


# --- grading items -----------------------------------------------------------------------------

def _chunk_text(chunk: str, cache: dict[str, str | None]) -> str | None:
    if chunk not in cache:
        code, text = mcp_call("read_section", {"chunk_id": chunk, "expand": "0"})
        # the reply is two head lines (vendor · product — title; source), a blank, the text
        cache[chunk] = None if code else text.split("\n\n", 1)[-1][:SOURCE_CHARS]
    return cache[chunk]


def build_item(reply: str, log: list[dict[str, Any]], arm: str, question: str, n: int,
               fetch: Any) -> dict[str, Any]:
    """One grading item: the question, the reply and up to ``n`` of its cited statements, each
    with the text of what its markers cite (three markers a statement at most)."""
    body, own = split_reply(reply)
    numbered, by_url = tool_sources(log, arm)
    checks = []
    for i, statement in enumerate(stride([s for s in sentences(body) if MARKERS.search(s)], n),
                                  start=1):
        sources = []
        for marker in _numbers(statement)[:3]:
            url, chunks = resolve(marker, own, numbered, by_url)
            texts = [t for t in map(fetch, chunks) if t]
            sources.append({"marker": marker, "url": url, "texts": texts} if texts
                           else {"marker": marker, "url": url, "unresolvable": True})
        checks.append({"n": i, "statement": statement, "sources": sources})
    return {"question": question, "reply": reply, "checks": checks}


def _items(args: argparse.Namespace) -> int:
    run = Path(args.run).resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    cache: dict[str, str | None] = {}
    items, key, missing = [], {}, []
    for q in manifest["questions"]:
        for arm in manifest["arms"]:
            name = f"{arm}-{q['id']}"
            md = run / "results" / f"{name}.md"
            if not md.exists() or not md.read_text().strip():
                missing.append(name)
                continue
            item = build_item(md.read_text(), read_log(run / "logs" / f"{name}.jsonl"), arm,
                              q["question"], args.statements, lambda c: _chunk_text(c, cache))
            item_id = hashlib.sha256(f"{args.seed}:{name}".encode()).hexdigest()[:8]
            items.append({"item": item_id, **item})
            key[item_id] = {"arm": arm, "question_id": q["id"], "checks": len(item["checks"])}
    random.Random(args.seed).shuffle(items)
    out = run / "grading"
    out.mkdir(exist_ok=True)
    batches: list[list[dict[str, Any]]] = [[]]
    size = 0
    for item in items:
        chars = len(json.dumps(item))
        if batches[-1] and size + chars > BATCH_CHARS:
            batches.append([])
            size = 0
        batches[-1].append(item)
        size += chars
    for i, batch in enumerate(batches, start=1):
        (out / f"batch-{i:02d}.json").write_text(json.dumps(batch, indent=1, ensure_ascii=False))
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"{len(items)} items in {len(batches)} batches under {out}"
          + (f"; no reply for: {', '.join(missing)}" if missing else ""))
    return 0


# --- scoring -----------------------------------------------------------------------------------

def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def score_run(run: Path) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    labelled = {q["id"]: q for q in load_set([q["id"] for q in manifest["questions"]])}
    arms: dict[str, Any] = {}
    for arm in manifest["arms"]:
        rows = {}
        for q in manifest["questions"]:
            name = f"{arm}-{q['id']}"
            md = run / "results" / f"{name}.md"
            if not md.exists():
                rows[q["id"]] = {"missing": True}
                continue
            row = reply_facts(md.read_text(), read_log(run / "logs" / f"{name}.jsonl"), arm,
                              q["question"])
            if q["kind"] == "vendors":
                try:
                    answer = json.loads((run / "results" / f"{name}.json").read_text())
                except (OSError, json.JSONDecodeError):
                    answer = {"supported": []}
                    row["no_vendor_list"] = True
                row.update(score_answer(labelled[q["id"]], answer))
            rows[q["id"]] = row
        done = [r for r in rows.values() if not r.get("missing")]
        total = {k: sum(r.get(k) or 0 for r in done) for k in (
            "sentences", "sentences_cited", "markers", "markers_resolved", "markers_given",
            "markers_kept", "tool_calls")}
        arms[arm] = {"questions": rows, "summary": {
            "answered": len(done), "missing": len(rows) - len(done),
            "recall": _mean([r["recall"] for r in done if r.get("recall") is not None]),
            "precision": _mean([r["precision"] for r in done if r.get("precision") is not None]),
            "verbatim": _share(sum(r["verbatim"] is True for r in done),
                               sum(r["verbatim"] is not None for r in done)),
            "sentences_cited": _share(total["sentences_cited"], total["sentences"]),
            "markers_resolved": _share(total["markers_resolved"], total["markers"]),
            "markers_kept": _share(total["markers_kept"], total["markers_given"]),
            "replies_without_markers": sum(r["markers"] == 0 for r in done),
            "tool_calls": _mean([r["tool_calls"] for r in done]),
            "tool_seconds": _mean([r["tool_seconds"] for r in done])}}
    return {"arms": arms}


def score_grades(run: Path) -> dict[str, Any]:
    """Per grader and arm: checks graded, the share unsupported and partly supported, and the
    uncited share of the claims the grader counted. A grade for an unknown item or check is an
    error; a missing one is reported."""
    grading = run / "grading"
    key = json.loads((grading / "key.json").read_text())
    out: dict[str, Any] = {}
    for path in sorted(grading.glob("grades-*.json")):
        grader = path.stem.split("-")[1]
        for g in json.loads(path.read_text()):
            if g["item"] not in key:
                raise SystemExit(f"{path.name}: unknown item {g['item']}")
            k = key[g["item"]]
            verdicts = [c["verdict"] for c in g.get("checks") or []]
            if len(verdicts) != k["checks"] or set(verdicts) - {"supported", "partly",
                                                                "unsupported"}:
                raise SystemExit(f"{path.name}: item {g['item']} needs {k['checks']} verdicts of"
                                 " supported / partly / unsupported")
            cell = out.setdefault(grader, {}).setdefault(k["arm"], {
                "items": 0, "checks": 0, "unsupported": 0, "partly": 0, "claims": 0,
                "uncited": 0})
            cell["items"] += 1
            cell["checks"] += len(verdicts)
            cell["unsupported"] += verdicts.count("unsupported")
            cell["partly"] += verdicts.count("partly")
            cell["claims"] += int(g.get("claims_total") or 0)
            cell["uncited"] += len(g.get("uncited_claims") or [])
    for arms in out.values():
        for cell in arms.values():
            cell["unsupported_rate"] = _share(cell["unsupported"], cell["checks"])
            cell["partly_rate"] = _share(cell["partly"], cell["checks"])
            cell["uncited_rate"] = _share(cell["uncited"], cell["claims"])
    return {"graders": out, "items": len(key)}


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1%}"


def _score(args: argparse.Namespace) -> int:
    run = Path(args.run).resolve()
    result = score_run(run)
    if (run / "grading" / "key.json").exists():
        result["grades"] = score_grades(run)
    (run / "score.json").write_text(json.dumps(result, indent=1) + "\n")
    for arm, a in result["arms"].items():
        s = a["summary"]
        print(f"{arm:9s} answered {s['answered']} (missing {s['missing']}) | vendor recall "
              f"{_pct(s['recall'])} precision {_pct(s['precision'])} | question verbatim "
              f"{_pct(s['verbatim'])} | sentences with a marker {_pct(s['sentences_cited'])} | "
              f"markers resolved {_pct(s['markers_resolved'])} | markers kept "
              f"{_pct(s['markers_kept'])} | replies with no marker "
              f"{s['replies_without_markers']} | tool calls {s['tool_calls'] or 0:.1f}, "
              f"{s['tool_seconds'] or 0:.0f} s")
    for grader, arms in (result.get("grades") or {}).get("graders", {}).items():
        for arm, c in arms.items():
            print(f"grader {grader:6s} {arm:9s} items {c['items']} checks {c['checks']} | "
                  f"unsupported {_pct(c['unsupported_rate'])} partly {_pct(c['partly_rate'])} | "
                  f"uncited claims {c['uncited']}/{c['claims']} ({_pct(c['uncited_rate'])})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="write one agent brief per question and arm")
    p.add_argument("--out", required=True)
    p.add_argument("--arms", nargs="+", choices=sorted(ARM_TOOLS), default=list(ARM_TOOLS))
    p.add_argument("--ids", nargs="+", default=None, help="question ids (default: the pilot)")
    p.add_argument("--max-searches", type=int, default=6)
    p.add_argument("--label", default=None)
    i = sub.add_parser("items", help="build the blind grading batches from the replies")
    i.add_argument("run")
    i.add_argument("--statements", type=int, default=8)
    i.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score", help="score the replies and any grades")
    s.add_argument("run")
    args = ap.parse_args(argv)
    return {"prepare": _prepare, "items": _items, "score": _score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
