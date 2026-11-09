# Claim support check (BACKLOG 36)

Approved in chat on 2026-11-09.

## Why

The evidence gate only checks that a claim's quote is verbatim. The quote does not always show
that the vendor's product has the capability. On 1,078 graded security facts, 9.8% of the
distilled 9B's facts were unsupported, against 0 of 57 of the Opus teacher's.

The gold labels came from two blind graders per fact, Opus and Fable, who agreed on 96.1%.
Their 42 disputes were adjudicated. Under the brief, a requirement, limit, configuration step or
list of what the capability supports counts as support. The target is not graded.

Retraining the 9B has a failed precedent. In graph-rag's dpo4 round, preference pairs made the
student terse rather than selective. A fine-tuned verifier (verifier2) reached the same ceiling
as the API checkers: about half the unsupported facts caught at about 4% wrongly dropped.

So the facts are checked after extraction.

## What the user asked for

- A vendor must not turn from "yes" to "no" because good facts were dropped by mistake. A vendor
  whose facts really are all unsupported should turn to "no".
- The second opinion uses Opus through the API (Anthropic or OpenRouter), not subagents, so
  production can reproduce it.
- The incremental runs (`vendorkb refresh`) check new claims too.

## Calibration (gold labels, 1,078 facts)

| checker, prompt v2 | catches unsupported | wrongly drops supported | drops that are right |
|---|---|---|---|
| Haiku 5.5 effort low | 64% | 8.4% | 43% |
| Haiku 5.5 effort medium (the default) | 47% | 4.3% | 52% |
| **Haiku 5.5 effort high** | **41%** | **2.6%** | **62%** |
| Haiku 5.5 effort max | 38% | 2.0% | 66% |
| gpt-6-luna (Azure), effort default | 65% | 9.8% | – |

The dev and test halves agree. Combinations add nothing.

**Reviewer.** The reviewer was run on the 67 gold facts that Haiku at high effort flags: 41
unsupported, 26 supported.

| reviewer | confirms unsupported | keeps the wrongly flagged | cost per fact |
|---|---|---|---|
| **Opus 5.5 effort low** | **34/41** | **26/26** | $0.0037 |
| Opus 5.5 effort medium | 33/41 | 26/26 | $0.0053 |

OpenRouter's `:batch` model ids cannot be used through chat completions (404), so the plain ids
are used.

**Production path (Anthropic API, `support.CHECK_SYSTEM` / `REVIEW_SYSTEM` through
`llm.build_request` and `llm.call`).** Run on the same 1,078 gold facts on 2026-11-09:

- **Checker, Haiku at effort high:** it flags 41/100 unsupported facts and 32/978 supported ones
  (3.3%). Cost: $0.128.
- **Reviewer, Opus at effort low,** on the 73 flagged facts: it keeps 32/32 of the wrongly flagged
  facts and confirms 32/41 of the unsupported ones. Cost: $0.0037 per fact. On this path Opus
  answers without a thinking block at effort low.
- **End to end:** 32% of the unsupported facts are removed, and 0 of the 978 supported ones.

## Design

The first version was approved in chat. It sent only the flagged claims a vendor's "yes" rested
on to the reviewer, and dropped the other flagged claims unreviewed. The pre-merge review showed
that those drops go stale. A topic edit, a deleted page, or a lookup filtered by target or
offering could make a vendor lose its "yes" without any reviewer verdict.

The user chose instead to have the reviewer check every flagged claim, on the Anthropic API.

**Table.** `claim_support` holds one row per checked claim:

- the checker's verdict and its id (model, effort and prompt version);
- the reviewer's verdict and id, which stay NULL until a review happens;
- the cost of both calls.

The rows cascade with their claim.

**Scope.** A claim is checked when all of these hold:

- it belongs to the claims-v3 family;
- it is mapped under the current taxonomy version;
- its support is `supported` or `partial`;
- its dimension belongs to an aspect of any loaded topic (today: ransomware protection);
- it has no verdict yet.

**Steps.**

1. The checker asks each due claim. A claim it calls unsupported is *flagged*.
2. The reviewer re-asks every flagged claim. A claim it calls unsupported is *rejected*. A claim
   it calls supported is kept.

**Build.** `capabilities build` leaves out the rejected claims only.

- A flagged claim stays until its review, so a wrong flag never costs a fact.
- A vendor whose facts the reviewer all rejects loses its "yes".
- Each verdict belongs to one claim, so the exclusion depends on nothing else: not on topics, not
  on other claims, not on pages deleted later.

**Commands.**

- `vendorkb claims support` checks, then reviews. It takes a budget for each step, and `--limit`
  for a smoke test.
- `vendorkb refresh` runs the same two steps after mapping, with their own per-run budgets. It
  rebuilds capabilities when anything changed. A failed support step is reported in
  `support.error`, the rebuild still runs, and the Job fails.

**Models.**

- `SUPPORT_BACKEND` is `anthropic` by default, which uses `ANTHROPIC_API_KEY`. It can also be
  `openai`, which uses `SUPPORT_BASE_URL` / `SUPPORT_API_KEY`, falling back to the `MAP_` ones.
- The models and efforts are settings:

| setting | default |
|---|---|
| `SUPPORT_MODEL` | `claude-haiku-5-5` |
| `SUPPORT_EFFORT` | `high` |
| `SUPPORT_REVIEW_MODEL` | `claude-opus-5-5` |
| `SUPPORT_REVIEW_EFFORT` | `low` |
| `SUPPORT_CONCURRENCY` | 8 |

**Spend and failures.**

- Each call is estimated before it is sent, at `MAX_TOKENS` (4,000) of output, so a budget bounds
  the worst case. A step stops when spent plus in flight plus the next estimate would pass its
  budget.
- A failed or unusable call writes no verdict, and the claim is asked again by the next run.
- A claim deleted while it is being asked is skipped (`vanished`).
- These stop the step, because nothing else would succeed either:
  - a non-rate-limit 400;
  - a 401, 402, 403 or 404 (a bad key, no credit, an unknown model).

**Expected effect on the 31,476 ransomware-dimension facts.** From the production-path
calibration:

- about 1,000 of the roughly 3,100 unsupported facts removed;
- about 0 good facts lost.

The cost is about $3.75 for the checker and about $7.90 for the reviewer, on about 2,100 flagged
facts.

## Not in scope

- Claims outside topic dimensions.
- `not_supported` claims.
- Target verification.
- Re-checking when the checker changes.
- The Message Batches API (half price): sync calls for now.
