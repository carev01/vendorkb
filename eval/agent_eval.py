"""Agent-level eval: fresh agents answer the labelled cross-vendor questions through the
deployed MCP server, and their answers are scored against ``expected_vendors``.

    uv run python eval/agent_eval.py prepare --out eval/runs/agent-2026-11-08
    # one fresh agent per brief (eval/runs/.../briefs/*.md), nothing else in its prompt
    uv run python eval/agent_eval.py score eval/runs/agent-2026-11-08 --baseline eval/agent_baseline.json
    uv run python eval/agent_eval.py score eval/runs/agent-2026-11-08 --write-baseline eval/agent_baseline.json

The questions are the cross-vendor ones in ``eval/questions.yaml`` with ``expected_vendors``
(``--ids`` narrows them). Arm ``full`` offers every MCP tool; ``search`` only
``list_vendors``, ``search_docs`` and ``read_section``. The agents run ``call``, which talks
to the MCP server (``VENDORKB_MCP_URL``; token from ``VENDORKB_MCP_TOKEN`` or
``~/.config/vendorkb/mcp-token``), refuses tools outside the arm and more than
``--max-searches`` ``search_docs`` calls (each reranks ``RERANK_CANDIDATES`` chunks, about 20k
rerank tokens), and logs every call as a JSON line. ``score`` exits 1 when a mean drops more
than ``--tolerance`` below the baseline, and 2 when an answer is missing or unreadable.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

ARMS = {"full": ("list_vendors", "list_dimensions", "list_targets", "query_capabilities",
                 "search_docs", "read_section"),
        "search": ("list_vendors", "search_docs", "read_section")}
DEFAULT_URL = "http://vendorkb-mcp.k3s.home.lan/mcp"
TOKEN_FILE = "~/.config/vendorkb/mcp-token"
REPO = Path(__file__).resolve().parent.parent

BRIEF = """# Agent brief: answer a cross-vendor question with the vendorkb tools

You are the agent a backup administrator asks. Answer ONE question about which backup vendors
support something, using only the vendorkb tools, called from a shell:

    {command} TOOL '{{"arg": "value"}}'

Tools available to you: {tools}. (`list_vendors` takes no arguments; `search_docs` takes
`query`, optional `vendor` (a list of vendor names) and `k`; `read_section` takes `chunk_id`
and `expand` "0"/"1"/"full"; `query_capabilities` takes taxonomy filters such as `dimension`,
`target`, `vendor`, `support`; `list_dimensions` / `list_targets` take `query`.) Run
`list_vendors` first for the vendor names. You have at most {max_searches} `search_docs`
calls; the command refuses more.

Rules:
- A vendor counts as supporting the capability only when a tool result shows documentation
  that says so; cite that source's URL. Do not use outside knowledge.
- Try to cover every vendor that documents it, not just the first few you find.
- Open or list no file other than this brief, and run no command other than the one above
  and the script that writes your answer.

Write your answer as JSON to {output} (build it with a short Python script and json.dump):
`{{"question_id": "{qid}", "supported": [{{"vendor": "<name as list_vendors spells it>",
"how": "<one sentence>", "sources": ["<url>", ...]}}], "not_supported": [{{"vendor": ...,
"sources": [...]}}], "notes": "<one sentence on what you could not check>"}}`.
Reply with just the number of supported vendors and the number of tool calls you made.

## Your question (id {qid})

{question}
"""


def load_questions(path: str | Path, ids: list[str] | None = None) -> list[dict[str, Any]]:
    """The cross-vendor questions with expected vendors, in file order (``ids`` narrows them;
    an unknown id is an error)."""
    qs = [q for q in yaml.safe_load(Path(path).read_text())
          if q.get("type") == "cross_vendor" and q.get("expected_vendors")]
    if ids:
        known = {q["id"] for q in qs}
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise SystemExit(f"not a labelled cross-vendor question: {', '.join(unknown)}")
        qs = [q for q in qs if q["id"] in ids]
    return qs


def _norm(vendor: str) -> str:
    return " ".join(vendor.split()).lower()


def _vendors(entries: list[Any] | None) -> set[str]:
    return {_norm(e["vendor"] if isinstance(e, dict) else e) for e in entries or []}


def score_answer(question: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """Vendor recall and precision of one answer. Unclear vendors count on neither side;
    precision is None when the answer lists no vendor outside them."""
    expected, unclear = _vendors(question["expected_vendors"]), _vendors(
        question.get("unclear_vendors"))
    got = _vendors(answer.get("supported"))
    exp = expected - unclear
    listed = got - unclear
    return {"recall": len(exp & listed) / len(exp) if exp else None,
            "precision": len(listed & exp) / len(listed) if listed else None,
            "listed": len(got), "false": sorted(listed - exp),
            "missed": sorted(exp - listed)}


def _mean(values: list[float | None]) -> float | None:
    vs = [v for v in values if v is not None]
    return statistics.fmean(vs) if vs else None


def summarize(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"questions": len(rows),
            "recall": _mean([r["recall"] for r in rows.values()]),
            "precision": _mean([r["precision"] for r in rows.values()])}


def gate(current: dict[str, Any], baseline: dict[str, Any], tolerance: float,
         question_tolerance: float) -> tuple[list[str], list[str]]:
    """(failures, warnings) of a run against a baseline over the questions both scored. A
    mean that drops more than ``tolerance`` fails; a question whose recall drops more than
    ``question_tolerance`` is a warning (one agent run varies too much to fail on)."""
    shared = sorted(set(current["questions"]) & set(baseline["questions"]))
    failures, warnings = [], []
    if not shared:
        return ["no question in common with the baseline"], warnings
    now = summarize({q: current["questions"][q] for q in shared})
    then = summarize({q: baseline["questions"][q] for q in shared})
    for metric in ("recall", "precision"):
        if now[metric] is not None and then[metric] is not None \
                and now[metric] < then[metric] - tolerance:
            failures.append(f"mean {metric} {now[metric]:.1%} < baseline {then[metric]:.1%}"
                            f" - {tolerance:.0%} over {len(shared)} questions")
    for q in shared:
        a, b = current["questions"][q]["recall"], baseline["questions"][q]["recall"]
        if a is not None and b is not None and a < b - question_tolerance:
            warnings.append(f"{q}: recall {a:.0%} (baseline {b:.0%})")
    return failures, warnings


def tool_counts(log: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                tool = json.loads(line)["tool"]
                counts[tool] = counts.get(tool, 0) + 1
    return counts


def check_call(tool: str, arm: str, log: Path | None, max_searches: int) -> str | None:
    """Why the call is refused, or None."""
    if tool not in ARMS[arm]:
        return f"tool {tool!r} is not available in this setup; use: {', '.join(ARMS[arm])}"
    if tool == "search_docs" and log is not None \
            and tool_counts(log).get("search_docs", 0) >= max_searches:
        return "search budget used up: answer from what you have"
    return None


def parse_reply(status: int, body: str) -> tuple[int, str]:
    """(exit code, text to print) of an MCP tools/call reply, plain JSON or server-sent events.
    An empty or non-JSON reply (the server restarting, say) is an error that says so."""
    if "data:" in body:
        body = next(line[5:].strip() for line in body.splitlines() if line.startswith("data:"))
    try:
        msg = json.loads(body)
    except json.JSONDecodeError:
        return 1, (f"error: HTTP {status} with no usable reply ({body[:200]!r}); the server may be"
                   " restarting: try the call again")
    if "error" in msg:
        return 1, f"error: {msg['error']}"
    res = msg["result"]
    return 0, (("TOOL ERROR: " if res.get("isError") else "")
               + "\n".join(c.get("text", "") for c in res.get("content", [])))


def _call(args: argparse.Namespace) -> int:
    import httpx

    arm = os.environ.get("ARM", "full")
    log = Path(os.environ["CALL_LOG"]) if os.environ.get("CALL_LOG") else None
    arguments = json.loads(args.arguments) if args.arguments else {}
    refused = check_call(args.tool, arm, log, int(os.environ.get("MAX_SEARCHES", "10")))
    if refused:
        print(refused, file=sys.stderr)
        return 2
    if log is not None:
        with log.open("a") as f:
            f.write(json.dumps({"tool": args.tool, "arguments": arguments}) + "\n")
    token = os.environ.get("VENDORKB_MCP_TOKEN") or Path(
        os.path.expanduser(TOKEN_FILE)).read_text().strip()
    r = httpx.post(os.environ.get("VENDORKB_MCP_URL", DEFAULT_URL), timeout=300, json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": args.tool, "arguments": arguments}},
        headers={"authorization": f"Bearer {token}", "content-type": "application/json",
                 "accept": "application/json, text/event-stream"})
    code, text = parse_reply(r.status_code, r.text)
    print(text, file=sys.stderr if code else sys.stdout)
    return code


def _prepare(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    for sub in ("briefs", "results", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.questions, args.ids)
    for q in questions:
        name = f"{args.arm}-{q['id']}"
        command = (f"cd {REPO} && CALL_LOG={out / 'logs' / name}.jsonl ARM={args.arm}"
                   f" MAX_SEARCHES={args.max_searches} {sys.executable} eval/agent_eval.py call")
        (out / "briefs" / f"{name}.md").write_text(BRIEF.format(
            command=command, tools=", ".join(ARMS[args.arm]), max_searches=args.max_searches,
            output=out / "results" / f"{name}.json", qid=q["id"],
            question=" ".join(q["question"].split())))
    manifest = {"arm": args.arm, "max_searches": args.max_searches,
                "questions": [q["id"] for q in questions], "label": args.label}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(questions)} briefs in {out / 'briefs'}")
    return 0


def score_run(run: Path, questions_path: str | Path) -> dict[str, Any]:
    """Every question of the run's manifest scored, with the tool calls its agent made.
    Raises FileNotFoundError / ValueError naming the missing or unreadable answers."""
    manifest = json.loads((run / "manifest.json").read_text())
    qs = {q["id"]: q for q in load_questions(questions_path, manifest["questions"])}
    rows, bad = {}, []
    for qid in manifest["questions"]:
        name = f"{manifest['arm']}-{qid}"
        try:
            answer = json.loads((run / "results" / f"{name}.json").read_text())
        except (OSError, json.JSONDecodeError) as e:
            bad.append(f"{name}: {e}")
            continue
        rows[qid] = {**score_answer(qs[qid], answer),
                     "tools": tool_counts(run / "logs" / f"{name}.jsonl")}
    if bad:
        raise ValueError("missing or unreadable answers:\n  " + "\n  ".join(bad))
    return {"arm": manifest["arm"], "max_searches": manifest["max_searches"],
            "label": manifest.get("label"), "summary": summarize(rows), "questions": rows}


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.0%}"


def _score(args: argparse.Namespace) -> int:
    try:
        result = score_run(Path(args.run), args.questions)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 2
    for qid, r in result["questions"].items():
        calls = sum(r["tools"].values())
        print(f"{qid:9s} recall {_pct(r['recall']):>4s} precision {_pct(r['precision']):>4s}"
              f" listed {r['listed']:2d} calls {calls:3d} (search {r['tools'].get('search_docs', 0)})"
              f" false {r['false']}")
    s = result["summary"]
    searches = sum(r["tools"].get("search_docs", 0) for r in result["questions"].values())
    print(f"MEAN {result['arm']}: recall {_pct(s['recall'])} precision {_pct(s['precision'])}"
          f" over {s['questions']} questions; {searches} search_docs calls")
    if args.write_baseline:
        Path(args.write_baseline).write_text(json.dumps(result, indent=1) + "\n")
        print(f"baseline written to {args.write_baseline}")
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        if baseline.get("arm") != result["arm"] or \
                baseline.get("max_searches") != result["max_searches"]:
            print(f"baseline is arm {baseline.get('arm')} with {baseline.get('max_searches')}"
                  f" searches; this run is arm {result['arm']} with {result['max_searches']}",
                  file=sys.stderr)
            return 2
        failures, warnings = gate(result, baseline, args.tolerance, args.question_tolerance)
        for w in warnings:
            print(f"WARNING {w}")
        for f in failures:
            print(f"FAIL {f}")
        if failures:
            return 1
        print("PASS against the baseline")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="write one agent brief per question")
    p.add_argument("--out", required=True)
    p.add_argument("--arm", choices=sorted(ARMS), default="full")
    p.add_argument("--max-searches", type=int, default=6)
    p.add_argument("--ids", nargs="+", default=None)
    p.add_argument("--questions", default=str(REPO / "eval" / "questions.yaml"))
    p.add_argument("--label", default=None, help="free text kept in the manifest")
    c = sub.add_parser("call", help="what the agents run: one MCP tool call")
    c.add_argument("tool")
    c.add_argument("arguments", nargs="?", default=None)
    s = sub.add_parser("score", help="score a run, optionally against a baseline")
    s.add_argument("run")
    s.add_argument("--questions", default=str(REPO / "eval" / "questions.yaml"))
    s.add_argument("--baseline", default=None)
    s.add_argument("--write-baseline", default=None)
    s.add_argument("--tolerance", type=float, default=0.05)
    s.add_argument("--question-tolerance", type=float, default=0.25)
    args = ap.parse_args(argv)
    return {"prepare": _prepare, "call": _call, "score": _score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
