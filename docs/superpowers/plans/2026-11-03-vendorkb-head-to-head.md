# vendorkb Head-to-Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build vendorkb M1/M2 (chunk retrieval + a single-shot answer step) and the harness that answers the same questions with it and with the deployed Graphiti answer-api, judges both blind, and applies the pre-stated decision rule.

**Architecture:**
- `src/vendorkb/` is a standalone package: Postgres + pgvector in a local container, section chunks, hybrid FTS+vector retrieval with RRF and Voyage rerank, and an answer step on the same writer model as answer-api.
- `scripts/headtohead/` is the only code that touches both systems. It holds the corpus export, the runner, the judge-context builder and the scorer.
- The eval set lives in `eval/headtohead/`.

**Tech Stack:** Python 3.12, uv, psycopg 3 + pgvector, httpx, openai SDK (OpenAI-compatible endpoints), pyyaml, testcontainers (`pgvector/pgvector:pg16`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-03-vendorkb-head-to-head-design.md` (approved 2026-11-03).

## Global Constraints

- **Branch:** `vendorkb-head-to-head`. Never commit to main from this plan. Pushing and merging need the user.
- **Imports:** `src/vendorkb/` never imports `graphiti_core`, `neo4j`, `graph_extract`, `answer_api`, `graph_sync` or `theme_builder`. Task 1 adds a test that enforces it.
- **Database:** local Docker Compose `pgvector/pgvector:pg16`, database `vendorkb`. Production Postgres is never touched.
- **Corpus:** exactly the `:Article` nodes with a `HAS_EPISODE` edge (expected 20,899).
- **Embeddings:** `EMBED_BASE_URL` (default `http://srv-llm.home.lan:8082/v1`), model
  `jinaai/jina-embeddings-v5-text-nano-retrieval`, dimension 768, batch cap 32. Raw text,
  no prefix, matching production.
- **Rerank:** Voyage `rerank-3`. POST `{RERANK_BASE_URL}/rerank` with
  `{"model","query","documents","top_k"}` and a Bearer key. The reply's `data[]` items
  carry `index` and `relevance_score`.
- **Answer writer:** `JUDGE_BASE_URL` / `JUDGE_MODEL` (`z-ai/glm-5.3-flash`) / `JUDGE_API_KEY`
  from `.env`, reasoning effort `low`.
- **Retrieval constants:** FTS top 50 (`websearch_to_tsquery('english', …)`), vector top 50
  (cosine), RRF k=60, rerank the top 40, return k=10.
- **Chunking:** 300–1,200 tokens; never split a table; prefix `"{title}\n{' > '.join(section_path)}\n\n"`.
- **Tokens:** `len(text) // 4`. Recorded in `stats` output as "approximate tokens (chars/4)" (the builder's call, per spec §4.1).
- **Refusal string** (vendorkb answer step, exact): `"I don't have enough information to answer that from the available sources."` This is the same text as answer-api's.
- **Never coerce an empty LLM reply to a value** (`content or ""` is forbidden): an empty or None reply is a failure, not an answer.
- **Secrets:** keys come only from `.env` and are never printed or committed; run gitleaks before every push.
- **Lint:** `uv run ruff check src tests scripts/headtohead` (E702: no semicolons in test fakes); `uv run mypy src`.
- **Money:** nothing that calls GLM or Voyage runs outside the operational tasks O4 and O5. O4 needs the user's go-ahead.

## Review Focus

1. **An article whose markdown has no headings, or only a single H1.** It must still become chunks (section path `[title]`), not zero chunks. Test in Task 2.
2. **A reranker outage (non-200, transport error, unreadable payload) during search.** The result must fall back to RRF order with `rerank_ok=False`, not raise and not silently drop results. The runner records the flag. Test in Task 4.
3. **The answer model citing a marker that wasn't in its context, or citing none.** Unknown markers are dropped and counted; an answer with zero valid citations that isn't the refusal is kept but flagged `uncited=True`. Test in Task 6.
4. **A question that names a vendor whose filtered search returns nothing.** The search reruns unfiltered, and the result records `scope_relaxed=True`. Test in Task 6.
5. **Judge output with the wrong number of labels, or a pass missing for a question.** The scorer rejects it with the question id rather than scoring a partial answer. Test in Task 11.

---

### Task 1: Package scaffold, config, schema, local database

**Files:**
- Modify: `pyproject.toml`: add `"psycopg[binary]>=3.2"`, `"pgvector>=0.3"` and `"pyyaml>=6"` to dependencies, and `"src/vendorkb"` to the hatch packages.
- Create: `deploy/vendorkb/docker-compose.yml` (service `vendorkb-db`, image `pgvector/pgvector:pg16`, port `127.0.0.1:55432:5432`, named volume, `POSTGRES_PASSWORD` from env `VENDORKB_DB_PASSWORD`).
- Create: `src/vendorkb/__init__.py`, `src/vendorkb/config.py`, `src/vendorkb/schema.sql`, `src/vendorkb/db.py`
- Test: `tests/unit/test_vendorkb_config.py`, `tests/integration/test_vendorkb_db.py`, `tests/unit/test_vendorkb_isolation.py`

**Interfaces:**
- Produces:
  - `VendorkbSettings` (pydantic-settings, reads `.env`, `extra="ignore"`). Fields:
    `vendorkb_dsn` (default `postgresql://postgres:@127.0.0.1:55432/vendorkb`),
    `embed_base_url`, `embed_model`, `embed_dim=768`, `embed_max_batch=32`,
    `rerank_base_url`, `rerank_model`, `rerank_api_key`, `judge_base_url`, `judge_model`,
    `judge_api_key`, `answer_reasoning_effort="low"`, `docext_base_url`,
    `docext_read_key`, `docext_verify_tls=True`.
  - `get_settings() -> VendorkbSettings`.
  - `async def connect(dsn: str) -> psycopg.AsyncConnection` (registers pgvector).
  - `async def init_schema(conn) -> None`: idempotent; applies `schema.sql`.
- Schema (`schema.sql`):
  - `CREATE EXTENSION IF NOT EXISTS vector`.
  - `documents(doc_id uuid PRIMARY KEY, vendor text NOT NULL, product text, source text,
    title text, url text, content_hash text NOT NULL, graph_content_hash text,
    fetched_at timestamptz NOT NULL)`.
  - `chunks(chunk_id uuid PRIMARY KEY, doc_id uuid REFERENCES documents ON DELETE CASCADE,
    section_path text[] NOT NULL, ordinal int NOT NULL, text text NOT NULL,
    token_count int NOT NULL, tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding vector(768) NOT NULL, UNIQUE(doc_id, ordinal))`.
  - Indexes: GIN on `tsv`, HNSW on `embedding vector_cosine_ops`, btree on
    `documents(vendor)` and `documents(product)`.

- [ ] **Step 1: Write the failing tests**
  - `test_settings_defaults`: `VendorkbSettings(_env_file=None)` gives `embed_dim == 768`,
    `embed_max_batch == 32` and `answer_reasoning_effort == "low"`.
  - `test_vendorkb_imports_nothing_from_the_graph_stack`: walk every `.py` file under
    `src/vendorkb/`, parse it with `ast`, and assert that no `Import`/`ImportFrom` names a
    module whose first segment is in
    `{"graphiti_core","neo4j","graph_extract","answer_api","graph_sync","theme_builder"}`.
  - Integration `test_init_schema_is_idempotent_and_creates_tables` (testcontainers
    `PostgresContainer("pgvector/pgvector:pg16")`): run `init_schema` twice. `documents`
    and `chunks` exist, and an insert with a 768-dimension vector succeeds.
- [ ] **Step 2: Run them, expect failure:** `uv run --extra dev pytest tests/unit/test_vendorkb_config.py tests/unit/test_vendorkb_isolation.py tests/integration/test_vendorkb_db.py -v`. Expected: ImportError / module not found.
- [ ] **Step 3: Implement** the files above, then `uv lock` and `uv sync --extra dev`.
- [ ] **Step 4: Run them, expect pass;** also `docker compose -f deploy/vendorkb/docker-compose.yml up -d` and `uv run python -c "import asyncio; from vendorkb.db import connect, init_schema; from vendorkb.config import get_settings; ..."` (one-liner that connects and runs `init_schema`). Expected: exit 0.
- [ ] **Step 5: Commit** `feat(vendorkb): package scaffold, settings, pgvector schema, local compose`.

### Task 2: Section chunker

**Files:**
- Create: `src/vendorkb/chunker.py`
- Test: `tests/unit/test_vendorkb_chunker.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Chunk: section_path: tuple[str, ...]; ordinal: int; text: str; token_count: int`.
  - `def chunk_markdown(title: str, markdown: str, *, min_tokens: int = 300, max_tokens: int = 1200) -> list[Chunk]`.
  - `def approx_tokens(text: str) -> int` (`len(text) // 4`).
- Behaviour:
  - ATX headings (`#`…`######`) build `section_path`. A path always starts with `title`.
    A leading H1 equal to the title is not repeated.
  - Headings inside fenced code blocks are ignored.
  - A section over `max_tokens` splits first at its sub-headings, then at blank-line
    paragraph boundaries. A markdown table (consecutive lines starting with `|`) is one
    indivisible block.
  - A section under `min_tokens` merges into the preceding chunk of the same parent path,
    or the following one if it is first.
  - Ordinals count up from 0. Each `text` starts with the prefix from Global Constraints,
    and `token_count` counts the whole text.

- [ ] **Step 1: Write the failing tests:**
  - `test_heading_trail_becomes_section_path`: `## A` / `### B` content gives a chunk whose `section_path == ("T", "A", "B")`.
  - `test_text_is_prefixed_with_title_and_path`: `chunk.text.startswith("T\nA > B\n\n")`.
  - `test_no_headings_gives_one_path_title_chunk` (Review Focus 1): plain paragraphs give at least 1 chunk, each with `section_path == ("T",)`.
  - `test_tiny_sections_merge_with_their_parent`: three 20-token sections under one parent give 1 chunk.
  - `test_oversized_section_splits_on_paragraphs`: one ~3,000-token section gives at least 3 chunks, each `<= 1200` tokens.
  - `test_table_is_never_split`: a 2,000-token table appears whole in exactly one chunk (all its rows are in that chunk), even though it exceeds `max_tokens`.
  - `test_headings_inside_code_fences_are_ignored`.
  - `test_ordinals_are_dense_from_zero`.
- [ ] **Step 2: Run them, expect failure:** `uv run --extra dev pytest tests/unit/test_vendorkb_chunker.py -v`.
- [ ] **Step 3: Implement** `chunk_markdown`. Approach: one pass that parses lines into a tree of sections (heading level → node), with fence state tracked; then a per-section split pass and a merge pass.
- [ ] **Step 4: Run them, expect pass.**
- [ ] **Step 5: Commit** `feat(vendorkb): heading-aware section chunker`.

### Task 3: Clients and ingest

**Files:**
- Create: `src/vendorkb/clients.py`, `src/vendorkb/ingest.py`
- Test: `tests/unit/test_vendorkb_clients.py`, `tests/integration/test_vendorkb_ingest.py`

**Interfaces:**
- Consumes: `Chunk`, `chunk_markdown` (Task 2); `connect`, `init_schema`, `VendorkbSettings` (Task 1).
- Produces:
  - `class Embedder: def __init__(self, settings, *, transport: httpx.AsyncBaseTransport | None = None); async def embed(self, texts: list[str]) -> list[list[float]]`. It sends batches of at most `embed_max_batch` to the OpenAI-compatible `/embeddings` endpoint and raises `RuntimeError` if any vector's length is not `embed_dim`.
  - `class Reranker: async def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]] | None`. It returns `None` on a non-200, a transport error or an unreadable payload (the same contract as answer-api's `rerank`, reimplemented without importing it), and drops out-of-range indices.
  - `class DocExtractor: async def article(self, article_id: str) -> dict` (`GET /api/articles/{id}`, header `X-API-Key`). It returns the JSON; the caller reads `content_markdown`.
  - `@dataclass class CorpusEntry: article_id: str; vendor: str; product: str | None; source: str | None; title: str; url: str | None; graph_content_hash: str | None`.
  - `def load_corpus(path: Path) -> list[CorpusEntry]` (reads Task 8's JSON).
  - `@dataclass class IngestReport: fetched: int; skipped_unchanged: int; failed: list[str]; changed_since_graph: list[str]; chunks_written: int`.
  - `async def ingest(conn, corpus: list[CorpusEntry], *, docext: DocExtractor, embedder: Embedder, concurrency: int = 8) -> IngestReport`.
- Behaviour:
  - `content_hash = sha256(content_markdown.encode()).hexdigest()`.
  - An article whose `documents.content_hash` already equals that hash is skipped (resume).
  - Otherwise the document and its chunks are replaced in one transaction.
  - An article whose hash differs from `graph_content_hash` is listed in `changed_since_graph` (and still ingested).
  - A fetch or embed failure is listed in `failed` with its id and never aborts the run.

- [ ] **Step 1: Write the failing tests** (fake transports via `httpx.MockTransport`):
  - `test_embedder_batches_at_the_cap`: 70 texts make 3 requests of sizes 32, 32 and 6.
  - `test_embedder_rejects_wrong_dimension`.
  - `test_reranker_returns_none_on_http_500_transport_error_and_bad_payload` (three cases).
  - `test_reranker_drops_out_of_range_indices`.
  - Integration `test_ingest_writes_chunks_and_resumes`: 2 articles; a second `ingest` call reports `skipped_unchanged == 2` and the chunk count is unchanged.
  - Integration `test_ingest_flags_changed_since_graph_and_records_failures`: one article with a mismatched `graph_content_hash`, and one whose fetch returns 500. The report lists them in `changed_since_graph` and `failed` respectively.
- [ ] **Step 2: Run them, expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, expect pass.**
- [ ] **Step 5: Commit** `feat(vendorkb): embedder/reranker/DocExtractor clients and resumable ingest`.

### Task 4: Hybrid search and read_section

**Files:**
- Create: `src/vendorkb/search.py`
- Test: `tests/unit/test_vendorkb_rrf.py`, `tests/integration/test_vendorkb_search.py`

**Interfaces:**
- Consumes: `Embedder`, `Reranker` (Task 3); `connect` (Task 1).
- Produces:
  - `def rrf(rankings: list[list[str]], *, k: int = 60) -> list[str]`: ids, best first; ties broken by first appearance.
  - `@dataclass class Hit: chunk_id: str; doc_id: str; text: str; section_path: list[str]; vendor: str; product: str | None; title: str; url: str | None; rrf_rank: int; rerank_score: float | None`.
  - `@dataclass class SearchResult: hits: list[Hit]; rerank_ok: bool`.
  - `async def search_docs(conn, query: str, *, embedder: Embedder, reranker: Reranker, vendor: list[str] | None = None, product: list[str] | None = None, k: int = 10, use_rerank: bool = True) -> SearchResult`, using the Global Constraints values (50/50, k=60, rerank top 40). `use_rerank=False` returns RRF order with `rerank_ok=False` and makes no reranker call. It is for free sanity checks only; the answer step always reranks.
  - `async def read_section(conn, chunk_id: str, expand: Literal["0", "1", "full"] = "0") -> list[Hit]` (ordered by ordinal).

- [ ] **Step 1: Write the failing tests:**
  - `test_rrf_scores_by_reciprocal_rank`: `rrf([["a","b"],["b","c"]])` returns `["b","a","c"]`.
  - `test_rrf_tie_breaks_by_first_appearance`.
  - Integration (fake embedder returning fixed vectors, fake reranker):
    - `test_search_merges_fts_and_vector_and_reranks`: a chunk found only by FTS and one found only by vector both appear, ordered by the fake rerank scores.
    - `test_vendor_filter_restricts_hits`.
    - `test_use_rerank_false_makes_no_reranker_call`.
    - `test_rerank_outage_falls_back_to_rrf_order` (Review Focus 2): the reranker returns `None`, so `rerank_ok is False`, hits are in RRF order, and `len(hits) == min(k, candidates)`.
    - `test_read_section_expand_modes`: `"0"` gives 1 hit, `"1"` gives the neighbours, `"full"` gives every chunk with the same `section_path`.
- [ ] **Step 2: Run them, expect failure.**
- [ ] **Step 3: Implement.** Two SQL queries, each `LIMIT 50`, with an optional `WHERE d.vendor = ANY(%s)` / `d.product = ANY(%s)`, then `rrf`, then rerank the first 40 texts.
- [ ] **Step 4: Run them, expect pass.**
- [ ] **Step 5: Commit** `feat(vendorkb): hybrid FTS+vector search with RRF and rerank; read_section`.

### Task 5: Scope detection

**Files:**
- Create: `src/vendorkb/scope.py`
- Test: `tests/unit/test_vendorkb_scope.py`

**Interfaces:**
- Produces:
  - `@dataclass class ScopeIndex: vendors: list[str]; products: list[str]; aliases: dict[str, list[str]]`.
  - `def load_scope_index(vendors: list[str], products: list[str], aliases_path: Path) -> ScopeIndex`.
  - `async def scope_index_from_db(conn, aliases_path: Path) -> ScopeIndex`: `SELECT DISTINCT vendor` / `product` from `documents`, then `load_scope_index`. `aliases_path` defaults to `src/answer_api/scope_aliases.json`, **read as JSON data, never imported**. A string value is normalised to a one-element list.
  - `def detect_scope(question: str, index: ScopeIndex) -> tuple[list[str], list[str]]`: (vendors, products).
- Behaviour:
  - Case-insensitive, whole-word matching; longest match first, and a matched span is not reused by a shorter name.
  - An alias maps to its target(s). A target that is a vendor name goes to the vendors list, otherwise to products.
  - Cross-vendor wording (`vendors`, `across`, `compare`, `which vendor`) returns `([], [])`.

- [ ] **Step 1: Write the failing tests:**
  - `test_alias_vbr_maps_to_product`.
  - `test_multi_target_alias_veeam_agent_returns_all_targets`.
  - `test_longest_match_wins` ("Veeam Backup for Microsoft 365" is not also counted as "Microsoft").
  - `test_case_insensitive_whole_word` ("aws" matches; "laws" does not).
  - `test_cross_vendor_wording_is_unscoped`.
  - `test_aliases_file_is_read_as_data`: the file loads, and `sys.modules` gains no `answer_api` entry.
- [ ] **Step 2: Run them, expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, expect pass.**
- [ ] **Step 5: Commit** `feat(vendorkb): question scope detection from document names and scope_aliases.json`.

### Task 6: Answer step

**Files:**
- Create: `src/vendorkb/answer.py`
- Test: `tests/unit/test_vendorkb_answer.py`

**Interfaces:**
- Consumes: `search_docs`, `SearchResult`, `Hit` (Task 4); `detect_scope`, `ScopeIndex` (Task 5); `VendorkbSettings` (Task 1).
- Produces:
  - `REFUSAL: str` (the exact string in Global Constraints).
  - `def build_prompt(question: str, hits: list[Hit]) -> str`: each hit is labelled `[n]` (1-based) with its title and section path. The rules text says: answer only from the sources; cite with `[n]` markers after each claim; never write URLs; if the sources do not answer the question, reply with exactly the refusal string.
  - `def resolve_markers(raw: str, hits: list[Hit]) -> tuple[str, list[dict], int]`: (answer text, citations, unknown marker count). A citation is `{"marker": n, "chunk_id", "doc_id", "url", "title", "section_path"}`, ordered by first appearance. Unknown markers are removed from the text.
  - `async def answer(conn, question: str, *, settings, embedder, reranker, scope_index, client: AsyncOpenAI) -> dict` returning
    `{"query", "answer", "citations", "unknown_markers", "uncited", "scope": {"vendors", "products", "relaxed"}, "rerank_ok", "retrieved", "usage": {"prompt_tokens","completion_tokens"} | None, "error": str | None}`.
- Behaviour:
  - Detected scope is passed to `search_docs`. If the filtered search returns 0 hits, it reruns unfiltered with `relaxed=True`.
  - 0 hits after that gives `REFUSAL` with no model call.
  - The model call uses `temperature=0` and `max_tokens=4000`. `reasoning_effort` is passed through the OpenRouter `extra_body={"reasoning": {"effort": settings.answer_reasoning_effort}}`.
  - An empty or None reply is retried once; if still empty, `error="empty reply"` and `answer=None`. Never `""` and never the refusal.
  - `uncited=True` when the answer is not the refusal and has 0 valid citations.

- [ ] **Step 1: Write the failing tests** (fake `AsyncOpenAI`-shaped client, monkeypatched `search_docs`):
  - `test_markers_resolve_to_chunks_in_order_of_appearance`.
  - `test_unknown_markers_are_dropped_and_counted` (Review Focus 3).
  - `test_uncited_non_refusal_is_flagged` (Review Focus 3).
  - `test_refusal_is_not_flagged_uncited`.
  - `test_filtered_empty_reruns_unfiltered_and_marks_relaxed` (Review Focus 4).
  - `test_no_hits_refuses_without_calling_the_model` (the fake client records 0 calls).
  - `test_empty_reply_is_an_error_not_an_answer`: two empty replies give `answer is None` and `error == "empty reply"`.
  - `test_prompt_contains_no_urls`: the hits carry URLs, and `build_prompt` output contains no `http`.
- [ ] **Step 2: Run them, expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, expect pass.**
- [ ] **Step 5: Commit** `feat(vendorkb): single-shot answer step with id-only citations and refusal`.

### Task 7: CLI

**Files:**
- Create: `src/vendorkb/cli.py` (argparse, `python -m vendorkb.cli`)
- Test: `tests/integration/test_vendorkb_cli.py`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces the subcommands:
  - `ingest --corpus PATH [--concurrency 8] [--report PATH]`: writes the `IngestReport` as JSON.
  - `search QUERY [--vendor V]... [--k 10] [--rerank]`: prints hits. Reranking is **off** unless `--rerank` is given, so manual checks cost nothing.
  - `answer QUESTION`: prints the answer dict as JSON.
  - `stats`: prints JSON `{"documents", "chunks", "chunks_per_vendor", "token_percentiles": {"p50","p90","p99","max"}, "token_counting": "approximate tokens (chars/4)", "over_max_tokens": n}`.

- [ ] **Step 1: Write the failing test** `test_stats_reports_counts_and_percentiles` (testcontainer DB seeded with 3 chunks): the JSON has `documents == 1`, `chunks == 3`, and every percentile key present.
- [ ] **Step 2: Run it, expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run it, expect pass;** `uv run ruff check src tests && uv run mypy src`. Expected: clean.
- [ ] **Step 5: Commit** `feat(vendorkb): CLI (ingest/search/answer/stats)`.

### Task 8: Corpus export (graph side)

**Files:**
- Create: `scripts/headtohead/__init__.py`, `scripts/headtohead/export_corpus.py`
- Test: `tests/unit/test_headtohead_export.py`

**Interfaces:**
- Produces:
  - `CORPUS_QUERY: str`: Cypher returning, for each `(a:Article)` with `EXISTS {(a)-[:HAS_EPISODE]->()}`: `article_id`, `vendor`, `product`, `source`, `title`, `url`, `graph_content_hash` (from `a.content_hash`). The structural chain is `(v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(s:Source)` then 1–2 hops to `a`. Bound the variable-length part to `[*1..2]` from `s`; an unbounded or `Vendor`-rooted `[*1..4]` path timed out on 2026-11-03.
  - `def rows_to_corpus(rows: list[dict]) -> list[dict]`: one entry per article (deduplicated by id), sorted by id, every key from `CorpusEntry` present.
  - CLI: `uv run --extra dev python -m scripts.headtohead.export_corpus --out eval/headtohead/corpus.json`. It prints counts per vendor, and also writes the graph-frozen check (`max(Episodic.created_at)`, `min/max(Community.generated_at)`, the stale community count) to `eval/headtohead/graph_state.json`.
- Note: before writing `CORPUS_QUERY`, confirm the property names (`Article.title`, `url` vs `source_url`) with one `MATCH (a:Article) RETURN keys(a) LIMIT 1`.

- [ ] **Step 1: Write the failing tests:**
  - `test_rows_to_corpus_dedupes_and_sorts`.
  - `test_rows_to_corpus_keeps_missing_optional_fields_as_none`.
  - `test_corpus_query_is_bounded`: `"[*1..2]"` is in `CORPUS_QUERY` and `"HAS_EPISODE"` is in it.
- [ ] **Step 2: Run them, expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, expect pass;** then run the CLI for real (read-only on Neo4j, free). Expected: total ≈ 20,899; per vendor Veeam ≈ 20,192, Microsoft ≈ 468, AWS ≈ 147, Cohesity ≈ 92; `graph_state.json` max episode time is 2026-10-11T09:11Z.
- [ ] **Step 5: Commit** the code (the corpus JSON is committed in O1).

### Task 9: Questions file schema and validator

**Files:**
- Create: `scripts/headtohead/questions.py`, `eval/headtohead/README.md` (format and writing rules, copied from spec §5)
- Test: `tests/unit/test_headtohead_questions.py`

**Interfaces:**
- Produces:
  - `TYPES = ("single_vendor", "multi_article", "cross_vendor", "unanswerable")`.
  - `TARGET_COUNTS = {"single_vendor": 50, "multi_article": 15, "cross_vendor": 15, "unanswerable": 20}`.
  - Question record (YAML list):
    `{id, type, question, gold_facts: [str], gold_sources: [{article_id, section?}], vendors: [str], written_from: [article_id], decides: bool, origin: "new"|"golden_questions"|"router_golden_tier1", search_check?: str}`.
  - `def validate(questions: list[dict], corpus_ids: set[str], changed_ids: set[str]) -> list[str]` returns the problems; empty means valid. Rules:
    - ids unique; type in `TYPES`;
    - `1 <= len(gold_facts) <= 8`, except unanswerable, which has `gold_facts == []` and a non-empty `search_check`;
    - every gold source and `written_from` id is in `corpus_ids` and not in `changed_ids`;
    - `decides is True` exactly when `origin == "new"`;
    - the new-question counts per type equal `TARGET_COUNTS`;
    - a cross-vendor question has at least 2 `vendors`, and its gold facts name every vendor in `vendors`.
  - CLI: `python -m scripts.headtohead.questions --check eval/headtohead/questions.yaml` prints the problems and exits 1 if there are any.

- [ ] **Step 1: Write the failing tests:** one passing fixture set, then one test per rule that breaks it (e.g. `test_more_than_eight_gold_facts_is_rejected`, `test_existing_questions_cannot_decide`, `test_changed_article_cannot_be_a_gold_source`, `test_wrong_type_counts_are_rejected`, `test_cross_vendor_gold_must_cover_each_vendor`).
- [ ] **Step 2: Run them, expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, expect pass.**
- [ ] **Step 5: Commit** `feat(headtohead): question schema and validator`.

### Task 10: Runner

**Files:**
- Create: `scripts/headtohead/run.py`
- Test: `tests/unit/test_headtohead_run.py`

**Interfaces:**
- Consumes: `vendorkb.answer.answer` and the Task 3 clients; the questions validated by Task 9.
- Produces:
  - `async def ask_answer_api(client: httpx.AsyncClient, question: str) -> dict`: `GET /answer?q=…` with no mode/scope (production defaults). It returns `{"answer", "citations", "mode"?, "raw": <envelope>, "latency_s", "error"}`.
  - `async def run(questions, *, answer_api: httpx.AsyncClient, vendorkb_answer: Callable[[str], Awaitable[dict]], out_dir: Path, limit: int | None, concurrency: int = 4) -> dict`. It writes `out_dir/answers.jsonl`, one line per question and side: `{"qid", "side": "graph"|"vendorkb", "answer", "citations", "latency_s", "usage", "error", "flags": {…}}`. It returns the run summary.
- Behaviour:
  - Each side retries a failure once. A second failure is recorded with `answer=None` and `error`.
  - Before running, it re-checks the graph-frozen state against `eval/headtohead/graph_state.json`, and refuses if `max(Episodic.created_at)` changed.
  - It records `out_dir/run_meta.json`: the answer-api pod image tag and its `JUDGE_MODEL`/`RERANK_MODEL`/`SYNTHESIS_REASONING_EFFORT`, read via `kubectl exec` env and printed without secrets, plus the vendorkb settings minus keys and the git commit.
  - `--limit 2` is the smoke mode. It also prints the GLM token usage per question for vendorkb, and the per-question OpenRouter cost when the response's `usage.cost` is present.
  - CLI: `python -m scripts.headtohead.run --questions eval/headtohead/questions.yaml --answer-api http://127.0.0.1:18100 --out eval/headtohead/runs/<date> [--limit 2]`.

- [ ] **Step 1: Write the failing tests** (MockTransport for answer-api, a fake `vendorkb_answer`):
  - `test_each_question_yields_one_line_per_side`.
  - `test_failure_is_retried_once_then_recorded_not_skipped`.
  - `test_limit_runs_only_n_questions`.
  - `test_refuses_when_graph_changed` (the frozen-state check fails, so it raises before any request; the transport records 0 calls).
- [ ] **Step 2: Run them, expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, expect pass.**
- [ ] **Step 5: Commit** `feat(headtohead): runner for both systems with smoke mode and frozen-graph check`.

### Task 11: Judge context and scoring

**Files:**
- Create: `scripts/headtohead/judge_context.py`, `scripts/headtohead/score.py`, `eval/headtohead/JUDGE_BRIEF.md`
- Test: `tests/unit/test_headtohead_judge_context.py`, `tests/unit/test_headtohead_score.py`

**Interfaces:**
- Consumes: `answers.jsonl` (Task 10), the questions (Task 9), and the vendorkb DB for chunk text.
- Judge context:
  - `EPISODES_FOR_FACTS: str`: Cypher `MATCH ()-[r:RELATES_TO {uuid: $u}]->() UNWIND r.episodes AS eid MATCH (e:Episodic {uuid: eid}) OPTIONAL MATCH (a:Article)-[:HAS_EPISODE]->(e) RETURN r.fact AS fact, e.content AS text, a.id AS article_id, a.title AS title`.
    - Every answer-api mode's citations carry `fact_uuid` (`synthesize._build_citations`; list mode and enumeration build the same shape), so community-report answers resolve the same way. The spec's "report → facts → episodes" path is already what their citations are.
    - A citation without `fact_uuid` gets `{"unresolvable": true}` in the context and is counted.
  - `def build_tasks(questions, answers, *, graph_sources: dict[str, list[dict]], chunk_texts: dict[str, str], seed: int, max_source_chars: int = 60000) -> list[dict]`. Each task is `{"key", "qid", "type", "question", "gold_facts", "labels": {"graph": "A"|"B", "vendorkb": "B"|"A"}, "answers": {"A": {...}, "B": {...}}, "truncated": {"A": bool, "B": bool}}`, where each answer carries its text and a numbered source list with the source text.
    - Graph sources show the fact text marked `system paraphrase (not evidence)` and then the episode texts.
    - The labels are shuffled per question from `seed`; pass 2 uses a different seed.
    - It writes `batches/batch-NNN.jsonl` of 4 tasks each.
- `JUDGE_BRIEF.md`: the instructions for the Opus subagents, with the answer format, written from spec §7. It contains:
  - the per-answer record format;
  - the three fixed-line rules: labels are not claims; a "needs validation" item is a hedge that is neither unsupported nor gold credit; each "documented for X" is a claim and "Checked" is not judged;
  - the source-text rule;
  - judge-decided abstention.
  - Answer format per task:
    `{"key", "A": {"gold": [bool…per gold fact], "claims": [{"text", "supported": bool}], "citations_ok": bool, "abstained": bool}, "B": {…}}`.
- Scoring (`score.py`):
  - `def perfect(rec: dict, qtype: str) -> bool`. Unanswerable: `abstained and not any claim`. Otherwise: `all(gold) and all(c["supported"] for c in claims) and citations_ok and not abstained`. A side with `answer is None` (error) is never perfect.
  - `def validate_judgement(j: dict, task: dict) -> str | None`: `len(gold)` must equal `len(task["gold_facts"])` for both labels, and both labels must be present.
  - `def combine_passes(passes: list[dict | None]) -> bool | None`: the majority of available verdicts; `None` if fewer than 2.
  - `def needs_third_pass(p1: bool, p2: bool) -> bool`.
  - `def sign_test_p(wins_a: int, wins_b: int) -> float`: exact two-sided binomial at p=0.5.
  - `def decide(per_question: list[dict]) -> dict` returns `{"wins_graph", "wins_vendorkb", "p", "winner": "graph"|"vendorkb"|"tie", "decision": "graph"|"vendorkb"}`. A tie decides `"vendorkb"` (spec §7). It only counts questions with `decides=True`.
  - `def report_tables(per_question, judgements) -> dict`: per type, fact recall, claim precision, citation precision, abstention correctness, the retrieval hit rate (a gold article among the cited sources' article ids), and latency. Built separately for `decides=True` and the 36 reported questions.
  - CLI: `python -m scripts.headtohead.score --run eval/headtohead/runs/<date>` prints problems (missing or invalid judgements, by key, Review Focus 5), the decision and the tables, and writes `result.json`.

- [ ] **Step 1: Write the failing tests:**
  - `test_graph_sources_mark_fact_as_paraphrase_and_include_episode_text`.
  - `test_labels_are_shuffled_and_recorded`: across 20 tasks with a fixed seed, both orders occur, and `labels` maps back correctly.
  - `test_source_text_is_capped_and_truncation_flagged`.
  - `test_citation_without_fact_uuid_is_marked_unresolvable`.
  - `test_perfect_requires_all_gold_supported_claims_and_citations`.
  - `test_unanswerable_perfect_only_when_abstaining_without_claims`.
  - `test_errored_answer_is_never_perfect`.
  - `test_wrong_gold_count_is_rejected_with_key` (Review Focus 5).
  - `test_majority_of_three_passes`.
  - `test_sign_test_values`: `sign_test_p(15, 5)` ≈ 0.0414 and `sign_test_p(12, 8)` ≈ 0.503 (exact binomial, 4 decimal places).
  - `test_tie_decides_vendorkb`.
  - `test_reported_questions_do_not_count_toward_decision`.
- [ ] **Step 2: Run them, expect failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, expect pass;** full suite `uv run --extra dev pytest -m "not live" -q`, `uv run ruff check src tests scripts/headtohead`, `uv run mypy src`. Expected: all green.
- [ ] **Step 5: Commit** `feat(headtohead): blind judge context and pre-stated scoring rule`.

---

## Operational tasks (after the code; the controller runs them, not subagents)

### O1: Corpus and index (free)
- [ ] Run `export_corpus` (Task 8) and commit `eval/headtohead/corpus.json` and `graph_state.json`.
- [ ] `docker compose -f deploy/vendorkb/docker-compose.yml up -d`, then `python -m vendorkb.cli ingest --corpus eval/headtohead/corpus.json --report eval/headtohead/ingest_report.json`. Re-run until `failed` is empty or only holds articles that are permanently gone (report those).
- [ ] `python -m vendorkb.cli stats > eval/headtohead/stats.json`. Check that `documents` equals the corpus count minus permanent failures, and record `changed_since_graph` (the count goes in the report).
- [ ] Sanity: 5 `search` queries by hand (e.g. "hardened repository immutability"), without `--rerank`, so they're free. The top hits must be on topic.

### O2: Question writing (Claude usage only)
- [ ] Sample source articles stratified by source (excluding `changed_since_graph`), then dispatch Opus subagents in batches with the writing rules from `eval/headtohead/README.md`:
  - 50 single-vendor;
  - 15 multi-article (pairs from the same source);
  - 15 cross-vendor (topics found in at least 2 vendors' articles, with per-vendor gold checked against every vendor's articles);
  - 20 unanswerable (each with its `search_check`).
- [ ] Gold facts for the 36 existing questions, from their expected articles. Drop and report any that fail the corpus or changed check.
- [ ] `python -m scripts.headtohead.questions --check eval/headtohead/questions.yaml` must come back clean.

### O3: User review and freeze
- [ ] Show the user 20 questions: 8 single-vendor, 4 multi-article, 4 cross-vendor and 4 unanswerable, plus 2 of the 36. Apply their edits, re-validate, then commit "freeze questions". From here on, `questions.yaml` is not edited.

### O4: Smoke run and cost gate (paid; needs the user)
- [ ] Port-forward answer-api (`kubectl -n graph-rag port-forward deploy/graph-rag-answer 18100:8000`), then run `run --limit 2`.
- [ ] Assert that both sides produced answers with citations, `run_meta.json` has the pod's models, and vendorkb `rerank_ok` is true.
- [ ] Show the user the measured cost per question and the projection for 136 questions. **Stop until they say go.**

### O5: Full run (paid, after go)
- [ ] `run` on all questions into `eval/headtohead/runs/<date>/`. Check that every question has two lines and list any errors.

### O6: Judging (Claude usage)
- [ ] Pass 1 and pass 2: `judge_context` with seeds 1 and 2, and Opus subagents on each batch following `JUDGE_BRIEF.md`. Validate with `score.py`, and re-ask any invalid or missing ones.
- [ ] Pass 3 for every question where `needs_third_pass` is true.
- [ ] Give the user 10 random questions (fixed seed) to hand-check against the judge's verdicts, and record agreement.

### O7: Result and report
- [ ] `score.py`, then write `docs/superpowers/head-to-head-report-<date>.md` containing:
  - the decision-rule outcome (wins, p, winner, decision);
  - the per-type tables for the deciding and reported sets;
  - `changed_since_graph` and the truncation counts;
  - five wins and five losses per side, with question, both answers and judge notes;
  - the hand-check agreement;
  - a recommendation.
- [ ] Commit, run gitleaks, then ask the user before pushing or merging.
