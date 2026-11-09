# Message Batches for every Claude call (study)

Status: proposed on 2026-11-09, awaiting the user's review.

## Why

The user, on 2026-11-09: none of vendorkb's Claude work is urgent. That covers the incremental
refresh, re-maps and a full re-extraction. All of it "can always wait until the following day".

The Message Batches API halves the price of every token, cache reads and writes included.
- A batch holds up to 100,000 requests or 256 MB.
- Most batches end within an hour, all within 24 hours.
- Results stay available for 29 days.
- Every Messages API feature works in a batch, including the effort setting and the strict JSON
  format all our calls use.

## Where Claude is called today

| call site | model | today | spend so far | batch fit |
|---|---|---|---|---|
| `claims extract` | Haiku 5.5 or Opus 5.5 | Anthropic API, **sync or batch** (batch mode exists, with a ledger); the refresh extracts with the distilled 9B on its own GPU endpoint (no Claude) | Opus teacher set; the Haiku pilot | already batch-capable |
| `taxonomy map` | Haiku 5.5 | **sync**; production and the refresh use OpenRouter | corpus map $28.50; re-maps $4–8 each; refresh ≤ $1 per run | yes |
| `claims support`, check | Haiku 5.5, effort high | **sync**, Anthropic API | $3.64 (first run) | yes |
| `claims support`, review | Opus 5.5, effort low | **sync**, Anthropic API | $6.41 (first run) | yes |
| `taxonomy propose` | the extract model | **sync** | ≤ $20 cap over all runs; rare | yes, low value |

Out of scope:
- The answer model (`JUDGE_*`, GLM on OpenRouter): a person or agent is waiting for it.
- The evaluation graders: Claude subagents, not API calls.

Prompt caching does not help the map and support calls. Their system prompts are 212–329
tokens, below the 512-token minimum for Haiku 5.5 and Opus 5.5. Extraction already caches its
system prompt, and in a batch the two discounts stack.

## Expected savings

The batch price is half of each spend above. Examples:
- A full corpus re-map: ~$28.50 → ~$14.
- The support check of every topic fact: ~$10 → ~$5.
- Extending the support check to every fact (BACKLOG 36): ~$15–20 plus the review → about half.
- A full re-extraction with Haiku: already estimated on batch prices in BACKLOG 36.

The incremental refresh spends cents to a dollar per run, so its saving is small in money. Its
value is that the whole pipeline runs one way.

## Design

### 1. A shared batch runner for the `taxonomy/llm.py` callers

**New module** `vendorkb/llm_batch.py`. It follows the pattern `claims/extract.py` already uses
in production:

- **Ledger first.** A row in a new `llm_batches` table is written under a `pending-<uuid>` id
  before `messages.batches.create`, then renamed to the provider's id. A crash between the two
  keeps the spend counted, and holds the items back until a person checks the provider's batch
  list.
- **Size.** Requests are split under the count and size caps. `extract._groups` already does
  this.
- **Budget.** Each request is reserved at its full `max_tokens` of output, at the batch price.
- **Collecting.** A run first collects every ended batch of its kind and hands each result to
  the caller's handler. Its cost is priced from the result's usage, at the batch price. Then the
  ledger row is marked processed.
- **Failures.** An `errored` or `expired` result writes nothing, so the item is due again next
  run. Repeated failures are capped, as extract caps `MAX_ATTEMPTS`. At the cap an item is
  *failed* and no longer asked:
  - a map adjudication: its claims stay unmapped;
  - a check: the claim stays unchecked, and so stays in capabilities;
  - a review: the claim stays flagged, and so stays in capabilities.

  Each command's `--retry-failed` resets those items.
- **No double asks.** An item whose batch is still in flight is never submitted again.
- **Per-item rows.** A second table, `llm_batch_items` (`batch_id`, `custom_id`, `kind`,
  `payload` jsonb, `attempts`, `status`), keeps what decoding a result needs. For the map that is
  the phrase, class and candidates; it is not in `map_decisions` until the decision exists. The
  table also holds the attempt counts that map and support have no other place for.

**Ledger row:** `batch_id` (PK), `kind` (`map`, `support-check`, `support-review`, `propose`),
`model`, `custom_ids`, `est_cost_usd`, `actual_cost_usd`, `status` (`submitted` or `processed`),
`submitted_at`, `ended_at`.

**Callers.** Each caller already has its three pieces:
- a request builder (`llm.build_request`);
- a reply parser;
- a writer: the map's `map_decisions` insert, or support's `record`.

Batch mode reuses them unchanged. Only the transport differs.

### 2. Two ways to run

- **`--wait`** (the CLI default in batch mode): submit, poll until the batches end, apply. This
  is how `extract --mode batch` works today. Use it for one-off runs: re-maps after a taxonomy
  load, extending the support check. The command's advisory lock is held only while it collects
  and submits, never while it waits. The ledger already prevents double asks, so a waiting CLI
  never blocks the refresh.
- **Submit and return** (the refresh): collect what has ended, apply it, submit the next work,
  exit. The next run carries on. The CronJob never sits in a poll loop for up to a day.

**Defaults.**
- `taxonomy map` and `claims support` default to batch mode on the anthropic backend, as
  `extract` already does. `--mode sync` stays for the rare case where waiting is not acceptable.
- Batch mode with an openai backend is refused, with the same message as `extract`.
- The refresh's mode is a Helm value (`claimsRefresh.batch`, default true), so a rollout can be
  rolled back without a new image.

**Budget.** `--max-usd` caps what one run submits. Each request is reserved at its full
`max_tokens` of output, at the batch price, so a cap bounds the worst case. Batches submitted by
earlier runs do not count against it.

For big jobs that reservation is several times the real cost: about 5x for the map, whose
`max_tokens` is 1024. The command therefore prints both the reservation and the expected cost
(from the calibrated reply sizes) before it submits.

**Locks.** `taxonomy map` takes no lock today; it gets one (`0x766B66`). Support's lock covers
check and review. A refresh that finds a lock held skips that stage and exits cleanly.

### 3. Taxonomy map in batch mode

The mapper already caches every adjudication in `map_decisions`. Batch mode fills that cache and
then maps from it.

A claim's dimension candidates depend on its target's class (`applicable(class_id)`). So
dimension adjudications can only be listed once the claim's target is resolved, and the cache
fills in **rounds**:

1. Target adjudications the cache lacks go into a batch.
2. Once they land, the dimension adjudications they unlock go into the next batch.
3. A normal map pass then resolves every claim from the cache, with no model call.

A claim with an adjudication in flight stays unmapped and is skipped by the map pass until the
batch lands. `taxonomy map` already allows claims left over to wait for the next run.

The logic between parsing a reply and storing it, `_Adjudicator._keep` and `_carry`, moves into
the collect handler, so a batch result and a sync reply are stored the same way.

### 4. Support check in batch mode

The semantics do not change. A check batch writes verdicts. Every flagged claim then goes into a
review batch. A flagged claim stays in capabilities until its review lands.

### 5. The refresh becomes a short pipeline over several runs

The hourly refresh is due today only when new chunks exist. It also becomes due when an
`llm_batches` row is `submitted`: one cheap query. A due run does, in order:

1. extract new chunks (GPU endpoint, sync, as today);
2. collect ended map batches, write their decisions, and map the claims the cache now resolves;
   submit the adjudications still missing (targets, then the dimensions they unlock);
3. collect ended check batches; submit checks for claims with no verdict;
4. collect ended review batches; submit reviews for flagged claims;
5. rebuild capabilities once support work is quiescent and something was written. Quiescent
   means:
   - no in-scope claim is waiting for a check;
   - no flagged claim is waiting for a review;
   - no batch is in flight.

   Items at their attempt cap count as done.

**Why step 5 waits.** Today's sync refresh checks and reviews before it rebuilds, so an
unsupported new fact is never published. Rebuilding after each stage would publish new facts
unchecked for hours (my call: no). Waiting keeps that guarantee, and `capabilities build` keeps
its rule unchanged: it leaves out rejected claims only. New facts outside the topic dimensions
wait for the same build; they are not urgent either.

A new page's facts reach capabilities after four batch stages: map targets, map dimensions,
check, review. That is usually a few hours. If every stage took the 24-hour maximum, it would be
about four days.

**The user's decision (2026-11-09): batch every stage.** The alternative was to batch only a
stage with more than a threshold of requests and run small incremental stages sync. It was
declined: one way of running, at half price, beats same-day freshness.

### 6. Extraction

Batch mode already exists. A full re-extraction with Haiku or Opus would use it as it is:
`extract --mode batch`, three batches for 217k chunks. Nothing to build. Worth a smoke on a few
hundred chunks first, since that mode has not run at corpus scale.

### 7. Production changes

- `map.backend` moves from OpenRouter to the Anthropic API: same Haiku price, half with batches.
- `ANTHROPIC_API_KEY` goes into `vendorkb-secret`. The support check already needs it once the
  refresh is enabled.
- OpenRouter stays for the answer model only.

## Calls left to the builder

- Whether `llm_batches` and `extract_batches` merge. My call: no. Extract's table works and is
  tested, so `llm_batches` is a sibling built on the same rules.
- The poll interval, and the attempt cap for errored and expired items: copy extract's.
- `taxonomy propose` batch mode: last, and optional. Its replies are applied to the YAML in
  cluster order after the batch ends.

## Risks

- **Latency.** Each of the refresh's four batch stages can take up to 24 hours, so new facts
  usually lag by a few hours and at worst by about four days. Accepted by the user.
- **Batch queue limits** on the Anthropic organisation (requests in queue per tier): the corpus
  re-map, ~281k adjudications, needs three batches and may queue. The ledger holds the rest.
- **Two runs collecting the same batch.** The refresh and a manual command are serialised by
  advisory locks: support's existing one and the map's new one (section 2).
- **Poison items.** An item that always errors stops at its attempt cap and is reported, so it
  never stalls the refresh's quiescence (section 5).
- **Proving the variable is live** (CLAUDE cost rule): every batch-mode run first smokes a few
  items and asserts the ledger row, the result count and the batch-priced cost before the full
  submission.

## Order of work

1. `llm_batch` runner and ledger, with fake-client tests (extract's fake batch client is the
   model).
2. Support check in batch mode, since its spend is the largest of the incremental work.
3. Taxonomy map in batch mode.
4. Refresh pipeline (section 5); Helm `map.backend: anthropic`; the secret key.
5. Optional: `taxonomy propose` in batch mode.
