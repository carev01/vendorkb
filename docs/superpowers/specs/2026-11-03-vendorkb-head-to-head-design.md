# Head-to-head: chunk retrieval (vendorkb M1/M2) vs. the Graphiti answer-api

> Status: draft for review, 2026-11-03. Source design: the "Backup Vendor Knowledge
> System" DESIGN.md (v1 draft, Nov 2026; the user's copy, not in the repo), milestones
> M1 and M2 only.

## 1. Intent

The project has spent months on a Graphiti knowledge graph without ever checking whether
the graph gives better answers than plain retrieval over the document text. A second
opinion recommends replacing it with chunk retrieval plus, later, a capability table.
Before anything is migrated or the graph is re-extracted, this project **measures both
on the same questions** and lets the result decide the direction.

The user's bar is **perfect answers**: whichever system gets closer to that wins.

## 2. Goals, non-goals, anti-goals

**Goals**
- Build vendorkb M1 and M2: section chunking, hybrid keyword+vector retrieval with
  reranking, and `search_docs` / `read_section`, plus a thin answer step so that it
  produces answers comparable with answer-api's.
- Build an eval set of 136 questions (100 deciding, 36 reported) with gold facts and gold sources (§5).
- Run both systems on that set, judge blind, and apply the pre-stated decision rule (§7).
- Report the result per question type, with enough detail to show *why* each system
  wins or loses.

**Non-goals (this project)**
- Claim extraction, the taxonomy, and the capability table (DESIGN M3–M5).
- An agent that calls `search_docs` / `read_section` itself. That would add a second
  variable, so it comes after this project, and only if chunk retrieval wins.
- MCP exposure of the new tools (M2's MCP server). The tools are Python functions and a
  CLI here. MCP wrapping is trivial later and does not change what is measured.
- Production deployment. Production stays parked (all workers at 0, no GPU).
- Changing the current answer-api or the graph in any way.

**Anti-goals: things that would make the result worthless even if everything runs**
- An unfair comparison:
  - questions phrased in the source article's own words (this favours keyword search);
  - the new side indexing documents the graph never saw, or newer versions of them;
  - different answer-writing models on the two sides.
- The question writer also acting as the judge.
- A decision rule chosen after seeing the numbers.
- Judge noise read as a result (lesson from the 2026-10-29 verifier trial: judge noise
  can exceed real differences).

## 3. Constraints

- **Repo and layout (user-approved).** Everything lives in the graph-rag repo, on a branch,
  as a new top-level package `src/vendorkb/`:
  - `vendorkb` never imports `graphiti_core`, `neo4j`, `graph_extract` or `answer_api`.
  - The eval runner (§6) is the only code that touches both systems.
  - If the new design wins, `vendorkb` must be movable to its own repo unchanged.
  - The runner lives outside `vendorkb`, in `scripts/headtohead/`.
- **Database.** pgvector is required, and production's Postgres (`postgres:16-alpine` in k3s)
  has no pgvector and must not be touched. Use a local Docker Compose container
  `pgvector/pgvector:pg16` with its own `vendorkb` database. A compose file under
  `deploy/vendorkb/`, with credentials from `.env`.
- **Corpus.** Index exactly the articles the graph has extracted: the 20,899 `:Article`
  nodes with a `HAS_EPISODE` edge:

  | Vendor | Articles |
  |---|---|
  | Veeam | 20,192 |
  | Microsoft | 468 |
  | AWS | 147 |
  | Cohesity | 92 |

  - **Article list:** a one-off export script writes the list (article id, vendor,
    product, source, title, url, `content_hash`) to a JSON file. That script may use
    neo4j; `vendorkb` reads only the file.
  - **Markdown:** fetched from DocExtractor through the existing read-only API, with the
    key from `.env`.
- **Version drift.** The graph was extracted between 2026-09-25 and 2026-10-11, and some
  articles may have changed upstream since.
  - Compare DocExtractor's current `content_hash` with the one stored on each `:Article`.
  - Changed articles are still indexed, since the graph has some version of them.
  - Changed articles are excluded from question writing, so no question's gold answer
    exists only in a newer version. The number excluded is reported.
- **The current side is what the deployed answer-api pod serves.** Where the pod and
  `.env` differ, the pod wins. The runner records the pod's image tag and model settings
  at run time.
  - **The graph is frozen** (checked 2026-11-03): the last episode was written on
    2026-10-11 at 09:11 UTC, and nothing after. The 883 community reports (2 marked stale)
    were generated between 12:52 and 18:47 the same day, after the last extraction. No
    rebuild is needed; the runner re-checks both timestamps before running.
- **Models (same as production unless stated):**
  - Embeddings: `jinaai/jina-embeddings-v5-text-nano-retrieval`, 768 dimensions, at the
    homelab endpoint (`EMBED_BASE_URL`, default `http://srv-llm.home.lan:8082/v1`). Batch
    cap 32, as in `ExtractSettings`.
  - Reranker: Voyage `rerank-3` (`RERANK_*` in `.env`), the one answer-api uses.
  - Answer writer: `z-ai/glm-5.3-flash` on OpenRouter, the pod's `JUDGE_MODEL` (answer-api's
    synthesis uses the `JUDGE_*` tier; config comments saying GLM-5.2 are stale). The pod
    sets no `SYNTHESIS_REASONING_EFFORT`, so it runs at the code default, `low`. vendorkb
    uses the same value.
- **Money.** Indexing is free (homelab embedder). The paid parts are:
  - reranking: Voyage, one call per question;
  - answering: GLM, 136 questions × 2 systems;
  - answer-api's own internal calls.

  Before the paid run: smoke-test on 2 questions, show the measured per-question cost,
  and get the user's go-ahead (CLAUDE.md cost rule). Question writing and judging use
  Claude subagents (Claude usage, not API spend).
- **Secrets and data.** Keys only in `.env`, never printed or committed; gitleaks before
  every push. Questions and gold facts quote public vendor docs and may live in the repo
  (`eval/headtohead/`). Raw answers and judge transcripts also go there.

## 4. The new side (vendorkb M1/M2)

### 4.1 Ingestion and chunking
- **Input:** article markdown plus metadata (vendor, product, source, title, url,
  content_hash, article id).
- **Chunk by heading hierarchy**, keeping `section_path` (the heading trail):
  - Target 300–1,200 tokens.
  - Split an oversized section on sub-headings, then on paragraphs.
  - Merge a tiny section into its parent.
  - Never split a table, even if that makes a chunk too large.
  - Prefix each chunk's text with its title and section path, so it is self-describing.
    That prefix is embedded and indexed with the chunk.
- **Tokens** are counted with the embedder's tokenizer, or a stated approximation if the
  tokenizer is unavailable (builder's call; record which).
- **Schema** (DESIGN §4, reduced to what M1/M2 needs):
  - `documents(doc_id = DocExtractor article id, vendor, product, source, title, url,
    content_hash unique, fetched_at)`
  - `chunks(chunk_id, doc_id, section_path text[], ordinal, text, token_count,
    tsv tsvector, embedding vector(768))`

  Re-ingest is idempotent on `content_hash`.

### 4.2 Retrieval: `search_docs(query, vendor?, product?, k=10)`
1. Run two searches, each returning its top 50: Postgres full-text search
   (`websearch_to_tsquery`, English config) and pgvector cosine search.
2. Merge them with Reciprocal Rank Fusion (k=60).
3. Rerank the top 40 with `rerank-3` and return the top k. Each result carries its chunk
   id, text, section path, vendor/product/title/url, and its scores.
4. Apply the vendor/product filter when the caller passes one.

**Vendor scope (my call).** The answer step detects named vendors and products in the
question:
- It matches case-insensitively against the names in `documents`, plus the repo's
  `scope_aliases.json`, which is read as a data file, not imported.
- When it finds a match, it passes that filter to the search.
- If the filtered search returns nothing, it reruns the search unfiltered.

This mirrors what answer-api does, so scope detection isn't a difference between the
two sides.

**No per-vendor fan-out for cross-vendor questions (my call).** M2 is plain retrieval;
answer-api has an enumeration mode that checks each vendor. This is a known handicap for
the new side on cross-vendor questions, reported per type rather than corrected. If the
new side loses only there, that points at a fan-out or the capability table, not at
chunk retrieval.

### 4.3 `read_section(chunk_id, expand=0|1|full)`
- `expand=0`: returns the chunk.
- `expand=1`: returns the chunk with its neighbours.
- `expand=full`: returns every chunk sharing its section path.

It is not used by the answer step (single-shot); it exists for the later agent variant
and for manual inspection.

### 4.4 Answer step (what the eval calls)
- `answer(question)` calls `search_docs` (k=10). It sends the chunks, each labelled with
  a short id, to the answer writer, with instructions to:
  - answer only from the chunks;
  - cite chunk ids, never URLs;
  - say it does not know when the chunks do not support an answer.
- Code then resolves the cited ids to URLs and section paths deterministically. Unknown
  ids are dropped and counted.
- The returned answer is plain text with numbered citations, the same shape the judge
  sees from answer-api.
- The prompt is written fresh for chunks. The *rules* (ids only, abstain when
  unsupported) match answer-api's.

### 4.5 CLI
`python -m vendorkb.cli ingest|search|answer|stats`.
- `ingest` resumes after interruption (skips docs already stored with the same hash).
- `stats` prints document and chunk counts and chunk-size percentiles.

## 5. The eval set

About 136 questions in `eval/headtohead/questions.yaml`: 100 new questions, which
decide the result, plus 36 existing ones, which are reported separately. Each has:
- an id and a type;
- the question text;
- gold facts: short, individually checkable statements;
- gold sources: article ids, with the section where known;
- the vendor(s) involved;
- the article(s) it was written from.

**New questions (100), written by Opus subagents from the articles themselves**,
sampled across many sources, not concentrated in a few. Single-vendor questions are
stratified by vendor, and the mix is reported. The total was raised from the 80
discussed, because the 36 existing questions no longer count towards the decision (my
call, see below) and the sign test needs the numbers.

| Type | Count | Notes |
|---|---|---|
| single-vendor detail | 50 | biased towards limits, conditions, versions, prerequisites |
| multi-article | 15 | answering needs two articles, e.g. a prerequisite on one page, a limit on another |
| cross-vendor | 15 | only on topics at least two of the four vendors' extracted articles cover |
| unanswerable | 20 | plausible questions the 20,899 articles do not answer, with the search used to check that recorded |

**Gold-fact rules (my calls):**
- At most 8 gold facts per question. No "list everything" questions whose complete
  answer would be longer than that.
- Gold facts are the *minimum* a perfect answer must contain. Extra items that are correct
  and supported by their citations never count against an answer.
- **Cross-vendor gold** is built by checking every one of the four vendors' extracted
  articles for the topic. The gold says, per vendor, "documented: …" or "no evidence in the
  corpus", so finding every vendor is rewarded.

**Writing rules:**
- Phrase the question the way a backup admin would ask it, **not in the article's words**.
  Distinctive terms that a user would genuinely type (product names, feature names) are
  allowed.
- Write gold facts so that a judge can check them in an answer.
- Each subagent gets the article text and the rules, and writes a fixed number of
  questions.

**Existing questions (36), reported but not deciding (my call):** answer-api's
retrieval and router were tuned against these, and they predate the rewording rule.
Letting them count would tilt the decision towards answer-api.
- `golden_questions.json` (15) and `router_golden_tier1.json` (21) have expected article
  ids but no gold facts.
- An Opus subagent adds gold facts from the expected articles.
- A question whose expected articles are outside the 20,899, or have changed (§3), is
  dropped and reported.

**User review:** before any system is run, the user reviews a sample of about 20
questions across all types for realism. Their edits apply to the set, and the set is
frozen and committed before the run.

## 6. The run

- **Runner:** `scripts/headtohead/run.py`.
- **Current side:** answer-api `GET /answer?q=…` (default mode and scope, as production
  serves it). It runs against the deployed pod through `kubectl port-forward`, or a local
  `uvicorn … --loop asyncio` on the same graph; the builder picks one and records it. The
  envelope's answer text and resolved sources are captured.
- **New side:** `vendorkb` `answer(question)`.
- **Order:** questions run sequentially or with modest concurrency.
- **Saved per question and side:** the answer, the citations (url, title, section), the
  latency, and the token usage where available.
- **Errors:** a failure is retried once, then recorded as a failed answer, which counts
  as not perfect. Errors are never silently skipped.

## 7. Judging and the decision rule (fixed before any run)

**Judge setup:**
- Opus subagents judge blind. For each question the judge gets:
  - the question, its type and gold facts;
  - the two answers under shuffled anonymous labels;
  - for each answer, the **source document text behind every citation**.

**Source text, the same rule for both sides (my call, following the DESIGN's "every
statement traces to a stored source span").** A claim is supported only if the original
document text supports it. An extracted fact or a generated report is the system's own
restatement, never evidence.
- **New side:** the cited chunk's text.
- **answer-api, fact citation:** the text of the episode(s) the fact came from (each
  episode's content is the chunk it was extracted from), labelled with their articles.
  The fact text is shown too, marked as the system's paraphrase.
- **answer-api, community-report citation:** the report is resolved to the facts it
  cites, then to their episodes' text.
- The source text per answer is capped at a size the builder sets. Any truncation is
  recorded per question.

**Fixed system lines (my calls):**
- `(Vendor · Product)` labels are attribution metadata, not claims.
- An item answer-api flags "needs validation" is a declared hedge. It is neither counted
  as an unsupported claim nor credited as presenting a gold fact.
- In the enumeration line "Checked / Documented for …", each "documented for X" is a claim
  that needs support from the citations. "Checked" describes the process and is not
  judged.
- The judge decides whether an answer abstained; no refusal-string matching.

**Per answer the judge records:**
- each gold fact: present / absent;
- each claim in the answer: supported by its citations / not supported;
- each citation: supports what it is attached to / does not;
- for unanswerable questions: abstained / answered.

**Perfect answer:**
- every gold fact present;
- no unsupported claim;
- every citation supporting;
- for unanswerable questions: abstained, with no invented answer.

**Noise control:**
- Every question is judged in two independent passes. A question where the two passes
  disagree on either answer's "perfect" verdict is judged a third time, and the majority
  stands.
- The user hand-checks 10 questions chosen at random.

**Decision rule:**
- Applies to the 100 new questions only.
- Compare the two systems question by question. A question is a win for the side whose
  answer is perfect when the other's is not.
- The side with more wins wins, provided a two-sided sign test on the questions with a
  winner gives p < 0.05. For the expected numbers that is a margin of roughly 10
  questions.
- Otherwise it is a **tie**, and a tie goes to the new design on cost and simplicity
  (my call, accepted by the user).

**Reported alongside, not deciding:**
- fact recall;
- claim and citation precision;
- abstention correctness;
- retrieval hit rate (gold article among the retrieved sources);
- latency and cost per answer.

Everything is also broken down per question type, and repeated on the 36 existing
questions as a separate table.

## 8. Deliverables

1. `src/vendorkb/` with unit tests (chunker, RRF, scope detection, citation resolution),
   and the local compose file.
2. The full corpus ingested, with `stats` output recorded.
3. `eval/headtohead/questions.yaml`, user-reviewed and frozen.
4. The runner and judge outputs in `eval/headtohead/runs/<date>/`.
5. A short report, `docs/superpowers/head-to-head-report-<date>.md`: the decision-rule
   outcome, the per-type tables, five illustrative wins and losses for each side, and a
   recommendation.

## 9. Left to the builder

- Library choices within the stack: Python 3.12, `uv`, psycopg 3, pgvector, `httpx`/
  `openai` clients.
- The schema details beyond §4.1, and the migrations tooling.
- How the export script and the ingest batch work, and their concurrency.
- The exact answer and judge prompts, within §4.4 and §7.
- How the question-writing and judging subagents are batched.
- Whether answer-api runs via port-forward or locally (§6).
