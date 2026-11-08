# Agent-level eval gate (2026-11-08, BACKLOG 28)

`eval/agent_eval.py` turns the four-question agent eval of `full-corpus-2026-11-08.md` into a
repeatable regression gate over every labelled cross-vendor question (23).

## How it works

1. `prepare --out RUN` writes one brief per question: the question, the tools of the arm, the
   answer format, and the shell command that calls the MCP server. Briefs carry no labels.
2. One fresh Claude Code subagent (Sonnet) per brief answers it through the deployed MCP server,
   nothing else in its prompt. `call` refuses tools outside the arm and more than
   `--max-searches` (default 6) `search_docs` calls, and logs every call.
3. `score RUN --baseline eval/agent_baseline.json` scores vendor recall and precision against
   `expected_vendors` (unclear vendors count on neither side) and exits 1 when a mean drops more
   than 5 points below the baseline. A question whose recall drops more than 25 points is a
   warning: one agent run varies too much to fail on.

Run it after a taxonomy, mapper or MCP tool change. A run costs 23 subagents and up to 138
searches (about 20k rerank tokens each); the baseline used 127.

## Baseline (taxonomy `2-535fdb7b`, image `sha-0b58ab4`)

| | recall | precision |
|---|---|---|
| mean over 23 questions | **96.0%** | **94.9%** |
| lowest question | 85% (cv-04, WAN seeding) | 67% (cv43-10, legal hold) |

## Run-to-run noise

Six questions (the largest, the weakest and four between) were run again on the same deployment:
the mean absolute difference per question was 1.5 points of recall (largest 9) and 3.2 of
precision (largest 20, cv43-06). The noise in a 23-question mean is about a point, so the
5-point tolerance does not fire on noise. The repeat passes the gate.

## What the runs showed

- **Agents lean on the capability table:** 40 `query_capabilities` calls on one question were
  common, and 6 searches were enough for every question.
- **Wasted calls:** 42 `query_capabilities` calls passed a list as `vendor` (it takes one name)
  and 13 used `target="product-wide"`, a scope label of the output, as a target. Both fail
  validation or resolve nothing, and one agent made 91 calls. Fixed in the MCP tools next.
- **Possible label gaps:** cv43-10 (legal hold) listed the same five unlabelled vendors in both
  runs (Arctera, Dell, Eon, Flosum, IBM). Consistent picks like these are candidates for a label
  review rather than agent errors; a review would raise precision, not change the gate.

## Spend

Claude Code subagents (29 Sonnet runs); about 162 searches, roughly 3.2M rerank tokens
(estimated at 20k each).
