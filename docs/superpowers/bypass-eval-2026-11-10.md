# Does vendorkb need its answer model? Three-arm pilot (2026-11-10, BACKLOG 42)

**Result: keep the answer model.** An agent that writes from `gather_evidence` is as accurate as
one that relays `answer_question`, but it cites less completely, says less, and is no faster.
`gather_evidence` stays as an optional tool. Raw tools are clearly worse on vendor recall.

## What ran

`eval/bypass_eval.py`, image `sha-4dd25f5`. Twelve questions: six labelled "which vendors"
questions (ex-rt-10, cv-01, cv-07, cv-08, cv43-04, cv43-10), three comparisons (ex-rt-07, cv-12,
cv-14) and three ransomware topic questions (the user's Veeam–Commvault comparison, tp-01,
tp-07). Each was answered three times, by a fresh Claude Haiku agent (a Claude Code subagent)
whose prompt was only its brief:

| arm | tools | rules |
|---|---|---|
| relay | `answer_question` | the Copilot agent's (docs/deploy/copilot-studio.md §5) |
| evidence | `gather_evidence`, `read_section` | the alternative rules of §5 |
| raw | `list_vendors`, `query_capabilities`, `search_docs`, `read_section`, `list_dimensions`, `list_targets` | the server's instructions; six searches at most |

36 replies. From each, up to eight cited statements (an even stride) went to two blind graders,
Claude Opus 5.5 and Claude Fable 5.2, with the whole text of the chunks the statement cites:
278 checks, 7 batches a grader. The graders agreed on 275 of 278.

## Results

| | relay | evidence | raw |
|---|---|---|---|
| vendor recall (6 questions) | 85.5% | 82.8% | 64.9% |
| vendor precision | 88.8% | 94.7% | 92.0% |
| checks unsupported (Opus / Fable) | 0% / 0% | 0% / 0% | 4.3% / 4.3% |
| checks partly supported (Opus / Fable) | 3.3% / 4.3% | 0% / 1.1% | 3.3% / 2.2% |
| claims without a marker (Opus / Fable) | 0% / 0% | 3.1% / 5.7% | 0.9% / 0.6% |
| sentences carrying a marker (code) | 93% | 69% | 67% |
| markers that cite a passage a tool returned | 99.7% | 99.7% | 99.3% |
| claims in the 12 replies (Opus / Fable count) | 691 / 807 | 484 / 546 | 345 / 354 |
| the question passed word for word | 12 of 12 | 12 of 12 | n/a |
| tool calls a question | 1.0 | 2.9 | 7.6 |

The rule set before the run: drop the answer model if the evidence arm is within 5 points of
the relay arm on vendor recall and precision, no more than 2 points worse on unsupported claims,
and keeps at least as many citations.

- **Vendor recall and precision: met.** 2.7 points less recall, 5.9 more precision. With six
  questions the pilot cannot tell differences under about 5 points.
- **Unsupported claims: met.** Neither arm had an unsupported check; the relay arm had three
  partly supported ones both graders flagged (statements of vendorkb's answer model that go a
  little past the passage), the evidence arm none.
- **Citations: not met.** The relay arm left no claim without a marker. The evidence arm left
  3–6% (15 and 31 claims), most of them in one reply (cv-01: paragraphs with a marker on some
  sentences only).

Two more things count against the bypass:

- **It says less.** The evidence replies hold about 30% fewer claims. The tool cuts each passage
  at 1,200 characters, and the agent is more cautious than the answer model.
- **It is not faster.** End to end (agent start to reply written), relay and evidence took about
  the same on every question: 20–30 s for the comparisons, 55–95 s for the "which vendors"
  questions, 165–330 s for the topic questions with other agents running. The retrieval is
  the time, and the agent writing the answer takes as long as the answer model did. The bypass
  saves the answer model's cost (about half a cent an answer) and spends more agent tokens.

The raw arm confirms the 2026-11-08 finding on a weaker model: Haiku with the raw tools finds
far fewer vendors (64.9% recall; 13% on cv-07) and is the only arm with unsupported statements.

## Side findings

- **Markers survive Haiku (BACKLOG 41).** The relay agents kept every marker of every answer
  (320 of 320). The loss seen in Copilot Studio came with its GPT model or its own formatting,
  not from Claude Haiku relaying; re-test in Copilot now that the agent is on Haiku.
- **The question arrives unchanged (BACKLOG 39).** With "word for word" in the tool and the
  rules, all 24 relay and evidence agents passed the question as the user wrote it.
- **Topic questions under load (BACKLOG 35).** With four to six agents at once, a topic
  `answer_question` or `gather_evidence` took 120–210 s (45–60 s alone). The database was busy
  with the vendor-filtered vector searches of the per-aspect retrieval, at about 1.2 cores. The
  public route cuts a request at 100 s.
- **Dropped connections.** Four tool calls (three logged, one before the log recorded failures) ended with "peer closed connection" mid-reply while
  the server was loaded; the pod did not restart and logged nothing. `agent_eval.py call` now
  reports it and the agent retries.
- **The eval's token file** (`~/.config/vendorkb/mcp-token`) was stale after the 2026-11-09
  rotation; it has to be refreshed after every rotation.

## Limits

- Twelve questions, one run an arm: differences under about 5 points are noise.
- Haiku in Claude Code is not Haiku in Copilot Studio, which adds its own orchestration and
  formatting.
- Times were taken with several agents running, so they are slower than production.
- A check is a sampled cited statement; a claim without a marker is counted but never checked
  against a source.

## Spend

36 Haiku agent runs (plus 6 that stopped on the stale token and one rerun), 14 grader runs.
Server side: 24 full retrievals, 12 of them with the answer model (about $0.06), and 39
`search_docs` calls.
