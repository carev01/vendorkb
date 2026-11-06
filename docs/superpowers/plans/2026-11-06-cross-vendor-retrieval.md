# Cross-vendor Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cross-vendor questions over the 43-vendor corpus are measured by an eval that
knows every vendor. Enumeration questions retrieve per vendor and answer with one cited line
per vendor, plus a deterministic "nothing documented" line. Comparison questions give each
named facet a fair share.

**Architecture:**
- A deterministic shape classifier (`shape.py`) splits questions into single, comparison and
  enumeration.
- `search.py` gains a reusable candidate step (one query embedding, many scoped searches).
  On top of it come two retrieval modes: per-vendor fan-out with a rerank floor, and
  per-facet quotas.
- `answer.py` dispatches on shape. The enumeration answer is post-checked in code: each line
  must cite a chunk from its own vendor.
- An eval script (`eval/cross_vendor.py`) dumps labelling candidates and scores vendor and
  facet recall.

**Tech Stack:** Python 3.12, psycopg 3 async, pgvector, Voyage voyage-4 + rerank-3, pytest +
testcontainers, uv.

**Spec:** `docs/superpowers/specs/2026-11-06-cross-vendor-retrieval-design.md`. §9 holds
binding calls that refine §3–§4.

## Global Constraints

- **Cite or abstain.** Every vendor named in an enumeration answer is backed by a cited chunk
  whose `vendor` is that vendor. This is checked in code after the model replies. The LLM
  never writes URLs.
- **The closing line says "nothing documented", never "not supported".** Its exact prefix:
  `Checked with nothing documented: `.
- **One query embedding per question.** Fan-out reuses it: never embed the question once per
  vendor.
- **No paid run without the user's go-ahead.** Rerank calls in the eval (floor calibration)
  and answer runs are paid. Retrieval with `use_rerank=False` is free apart from query
  embeddings.
- **Tests never read `.env`.** Integration tests use the `vkb_dsn` testcontainer fixture and
  fake embedder/reranker/LLM clients.
- **CI gate:** `uv run ruff check src tests`, `uv run mypy src`, `uv run pytest`.
- **Defaults pinned by the spec:**
  - `per_vendor=3`, multi-product vendors up to 5 (their best chunk per product);
    enumeration answer chunks capped at 60;
  - comparison: 5 hits per facet after rerank;
  - targets: enumeration vendor recall ≥ 0.8, comparison facet recall ≥ 0.9, single-vendor
    and multi-article pool recall within 0.02 of baseline;
  - latency under 10 s per enumeration question before the answer model; rerank tokens per
    question under 100K.

## Review Focus

1. **A vendor with no chunks at all.** For example, Red Hat has 7 documents. It must appear
   in the closing line, and must not crash the fan-out or the rerank index mapping. Test in
   Task 2.
2. **Rerank failure during enumeration** (`rerank()` returns None). Every vendor with
   candidates is kept, `rerank_ok=False`, and no vendor is dropped by the floor. Test in
   Task 2.
3. **A model line whose name is a product or alias, not the exact vendor name** ("Azure
   Backup: …"). It is dropped, never mapped fuzzily. Test in Task 3.
4. **A comparison where one facet has zero matching documents.** That facet is in
   `facets_empty` and in the closing line; the other facet still gets its hits. Test in
   Task 2 (retrieval) and Task 3 (closing line).
5. **A question with cross-vendor wording plus one named vendor** ("other vendors besides
   Veeam"). It is classified as enumeration over all vendors. Test in Task 1.

---

### Task 1: Question shape

**Files:**
- Create: `src/vendorkb/shape.py`
- Modify: `src/vendorkb/scope.py` (split facet finding out of `detect_scope`)
- Test: `tests/unit/test_vendorkb_shape.py`, `tests/unit/test_vendorkb_scope.py` (unchanged
  behaviour)

**Interfaces:**
- Produces:
  - `scope.find_facets(question: str, index: ScopeIndex) -> tuple[list[str], list[str]]`:
    `(vendors, products)` named in the question, ignoring `_CROSS_VENDOR`.
    `detect_scope` becomes `([], [])` if cross-vendor wording, else `find_facets(...)`.
  - `scope.is_cross_vendor(question: str) -> bool`.
  - In `shape.py`:
    - `Facet = tuple[Literal["vendor", "product"], str]`;
    - `@dataclass(frozen=True) class Shape: kind: Literal["single", "comparison",
      "enumeration"]; facets: list[Facet]`;
    - `detect_shape(question: str, index: ScopeIndex) -> Shape`.
  - Rules (spec §9.3):
    - two or more distinct facets make a comparison; a product counts as one facet, even
      when its vendor is also named;
    - cross-vendor wording with 0 or 1 facets makes an enumeration, whose `facets` lists the
      named one, if any;
    - otherwise single.
  - Extend `_CROSS_VENDOR` so the eval's enumeration wordings match:
    - "which of these vendors/products", "which products";
    - "which vendors" (already covered);
    - "what products", "any products".

- [ ] **Step 1: Write failing tests.** Build a `ScopeIndex` with the 43 production vendor names (local fixture): Acronis, Afi.ai,
  Arcserve, Arctera, Atempo, AvePoint, AWS, Axcient, Bacula, Barracuda, CloudAlly, CloudCasa,
  Cohesity, Commvault, Datto, Dell, Druva, Eon, Flosum, Gearset, Google, GRAX,
  Hornetsecurity (Altaro), HYCU, IBM, Keepit, Microsoft, N-Able, Nakivo,
  NinjaOne (Dropsuite), Portworx, Red Hat, Rubrik, Salesforce, Slide, Spanning, Storware,
  Synology, Trilio, Unitrends, Veeam, Velero, Zerto and these products: `AWS Backup`, `Azure Backup`, `Microsoft 365 Backup`,
  `Entra ID Backup`, `FortKnox`, `Veeam Backup & Replication`,
  `Veeam Backup for Microsoft 365`, `Veeam Data Cloud`.
  - `test_eval_shapes`: load `eval/questions.yaml`. Every `cross_vendor` question in
    {ex-rt-10, ex-rt-12, cv-01..cv-09} is `enumeration`. Every one in
    {ex-rt-07, ex-rt-08, ex-rt-09, cv-10..cv-15} is `comparison`. At least 95% of
    `single_vendor` questions are `single`; assert the misclassified ids are printed in the
    failure message.
  - `test_other_vendors_besides_one_is_enumeration`: "Do other vendors besides Veeam support
    tape?" gives `Shape("enumeration", [("vendor", "Veeam")])`.
  - `test_product_and_its_vendor_is_one_comparison_side`: "Compare AWS Backup and Azure
    Backup" gives a comparison with facets `[("product", "AWS Backup"),
    ("product", "Azure Backup")]`.
  - `test_detect_scope_unchanged`: the existing scope tests pass untouched.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_vendorkb_shape.py -q` and expect a FAIL
  (no module).
- [ ] **Step 3: Implement** `find_facets`, `is_cross_vendor` and `detect_shape`. Vendors and
  products are deduplicated in question order. When a product's vendor is also named, the
  vendor facet is dropped (the product is more specific).
- [ ] **Step 4: Run** the unit suite and expect a PASS. Then run the gate.
- [ ] **Step 5: Commit** `feat(shape): classify questions as single, comparison or
  enumeration`.

### Task 2: Retrieval modes

**Files:**
- Modify: `src/vendorkb/search.py`
- Test: `tests/integration/test_vendorkb_search.py`, `tests/unit/test_vendorkb_rrf.py`

**Interfaces:**
- Consumes: `Facet` from Task 1.
- Produces:
  - `rrf(rankings, *, k=RRF_K, weights: list[float] | None = None) -> list[str]`. Each
    ranking's contribution is multiplied by its weight; the default is all 1.0, so
    behaviour is unchanged.
  - `FTS_WEIGHT = 1.0`: a module constant, passed as the full-text weight by
    `hybrid_candidates`.
  - `async hybrid_candidates(conn, query: str, qvec: list[float], *, vendor=None,
    product=None, limit=RERANK_CANDIDATES, vector_column="embedding") -> list[Hit]`: the
    current FTS + vector + RRF step, in RRF order. `search_docs` is rewritten on top of it
    with identical results (the existing tests stay green).
  - `@dataclass class EnumerationResult: hits: list[Hit]; kept: list[str]; dropped:
    list[str]; empty: list[str]; rerank_ok: bool`.
    - `kept` holds vendors whose evidence goes to the answer.
    - `dropped` holds vendors whose best score fell below the floor.
    - `empty` holds vendors with no candidates.
    - `kept`, `dropped` and `empty` together are exactly the vendors passed in.
  - `async enumerate_vendors(conn, query, *, embedder, reranker, vendors: list[str],
    per_vendor=3, per_vendor_multi=5, min_score: float = ENUMERATION_MIN_SCORE,
    max_hits=60, use_rerank=True) -> EnumerationResult`:
    - One embed call.
    - For each vendor: `hybrid_candidates(..., vendor=[v], limit=per_vendor_multi * 2)`.
      Keep the best chunk per product first, up to `per_vendor_multi` for a vendor with
      more than one product among its candidates, else `per_vendor`.
    - Then one `reranker.rerank` over all kept chunks.
    - A vendor survives if its max score ≥ `min_score`.
    - Hits are the survivors' chunks: vendors ordered by best score, chunks by score,
      truncated to `max_hits` by dropping each vendor's lowest-scored chunks first (never a
      whole vendor while another still has more than one chunk).
    - Rerank None or unusable: keep every vendor with candidates, use RRF order, and set
      `rerank_ok=False`.
    - `use_rerank=False`: same path as a failure, but this is not an error (free eval mode).
  - `ENUMERATION_MIN_SCORE = 0.0`: a placeholder until Task 6 calibrates it.
  - `@dataclass class ComparisonResult: hits: list[Hit]; facets_empty: list[str];
    rerank_ok: bool`.
  - `async compare_facets(conn, query, *, embedder, reranker, facets: list[Facet],
    per_facet=5, use_rerank=True) -> ComparisonResult`:
    - One embed call.
    - For each facet, `hybrid_candidates` scoped to it, with
      `limit = RERANK_CANDIDATES // len(facets)`.
    - Rerank each facet's candidates separately and keep its top `per_facet`. Per-facet
      reranking makes the quota survive the rerank.
    - Hits are interleaved facet by facet.
    - A facet with no candidates goes in `facets_empty`.
- `search.py` may grow past ~300 lines. If it does, move the two modes to
  `src/vendorkb/modes.py` and re-export nothing (callers import from `modes`). This is the
  implementer's call; record it in the report.

- [ ] **Step 1: Write failing tests.** Use the existing search test fixtures (fake embedder,
  fake reranker that scores by a keyword or returns None).
  - `test_rrf_weights`: `rrf([["a","b"],["b","a"]], weights=[1.0, 0.0]) == ["a","b"]`, and
    `weights=None` equals the unweighted result.
  - `test_search_docs_unchanged`: the existing search tests pass.
  - `test_enumeration_covers_every_vendor`: seed 4 vendors, one of them with 0 documents.
    `kept + dropped + empty` is the set of all 4 vendors, the 0-document vendor is in
    `empty`, and the embedder is called exactly once.
  - `test_enumeration_per_vendor_cap`: a single-product vendor with 10 matching chunks
    contributes at most 3 hits; a vendor with 3 products contributes at most 5, at least one
    per product.
  - `test_enumeration_floor_drops_low_vendor`: the fake reranker scores vendor C's chunks
    at 0.1 with `min_score=0.3`, so C is in `dropped` and none of its chunks are in `hits`.
  - `test_enumeration_rerank_failure_keeps_all`: the reranker returns None, so every vendor
    with candidates is in `kept`, `rerank_ok is False` and `dropped == []`.
  - `test_enumeration_max_hits`: 30 vendors × 3 chunks with `max_hits=60` gives
    `len(hits) == 60` and every kept vendor still has at least 1 hit.
  - `test_comparison_quota_survives_rerank`: facet A has 30 chunks scoring 0.9 and facet B
    has 5 scoring 0.5, so the hits are 5 from A and 5 from B, interleaved.
  - `test_comparison_empty_facet`: facet B has no documents, so `facets_empty == ["B"]` and
    A still has 5 hits.
- [ ] **Step 2: Run** them and expect a FAIL.
- [ ] **Step 3: Implement** to the interfaces above.
- [ ] **Step 4: Run** the gate and expect a PASS.
- [ ] **Step 5: Commit** `feat(search): per-vendor enumeration and per-facet comparison
  retrieval`.

### Task 3: Shape-aware answers

**Files:**
- Modify: `src/vendorkb/answer.py`, `src/vendorkb/cli.py` (the `answer` output now includes
  the new fields; no new flags)
- Test: `tests/unit/test_vendorkb_answer.py`

**Interfaces:**
- Consumes: `detect_shape` (Task 1); `enumerate_vendors` and `compare_facets` (Task 2).
- Produces:
  - `answer(...)` dispatches on `detect_shape`:
    - single: today's path, unchanged;
    - comparison: `compare_facets`, then the normal prompt;
    - enumeration: `enumerate_vendors` over every corpus vendor (`scope_index.vendors`),
      then the enumeration prompt.
  - The result dict gains:
    - `"shape"`: the kind;
    - `"facets_empty"`: a list (comparison);
    - `"vendors_checked"` and `"vendors_listed"`: lists (enumeration);
    - `"lines_dropped"`: an int (enumeration).
  - `ENUMERATION_RULES`: appended to the system rules for enumeration. It tells the model:
    - answer with one line per vendor, exactly `Vendor (Product, ...): how it is done [n]`;
    - begin each line with one of the exact vendor names listed in the prompt;
    - write no line for a vendor whose sources do not state the capability;
    - add nothing after the lines.
  - The prompt lists the kept vendor names.
  - `check_enumeration_lines(text: str, hits: list[Hit], vendors: list[str]) -> tuple[str,
    list[str], int]`: returns `(kept_text, vendors_listed, dropped_count)`. A line is kept
    only when its leading name, up to the first ` (` or `:`, is exactly one of `vendors`
    AND at least one of its `[n]` markers resolves to a hit whose `vendor` equals it.
  - `closing_line(names: list[str]) -> str`: returns
    `"Checked with nothing documented: " + ", ".join(sorted(names)) + "."`, or `""` when
    `names` is empty.
    - Enumeration: names = every corpus vendor minus `vendors_listed`.
    - Comparison: names = `facets_empty`.
  - Every enumeration line dropped means the answer is `REFUSAL`, with the closing line
    still appended.
  - Marker resolution and citations reuse `resolve_markers` on the kept text.

- [ ] **Step 1: Write failing tests** with fake model clients returning canned text and
  monkeypatched retrieval.
  - `test_enumeration_drops_wrong_vendor_line`: "Veeam: does X [1]" where hit 1 is a
    Commvault chunk is dropped, so `lines_dropped == 1` and Veeam is not in
    `vendors_listed`.
  - `test_enumeration_drops_alias_line`: "Azure Backup: does X [2]" where hit 2 is a
    Microsoft chunk is dropped (an exact vendor name is required).
  - `test_enumeration_closing_line_lists_unlisted_vendors`: corpus vendors are {A, B, C, D},
    A is listed and C is dropped by the floor. The answer ends with
    `"Checked with nothing documented: B, C, D."`.
  - `test_enumeration_all_lines_dropped_refuses`.
  - `test_comparison_closing_line_for_empty_facet`: `facets_empty == ["AWS Backup"]` gives
    `"Checked with nothing documented: AWS Backup."` at the end.
  - `test_single_shape_unchanged`: today's answer tests pass untouched.
- [ ] **Step 2: Run** them and expect a FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the gate and expect a PASS.
- [ ] **Step 5: Commit** `feat(answer): enumeration lines checked against cited vendors;
  closing line`.

### Task 4: Cross-vendor eval tooling

**Files:**
- Create: `eval/cross_vendor.py`
- Modify: `eval/README.md` (new fields and commands)
- Test: `tests/unit/test_cross_vendor_eval.py` (the scoring functions only)

**Interfaces:**
- Consumes: `hybrid_candidates`, `enumerate_vendors` and `compare_facets` (Task 2);
  `detect_shape` (Task 1).
- New question fields (cross-vendor questions):
  - `shape: enumeration | comparison`;
  - `expected_vendors: [{vendor, product, article_ids: [..]}]`;
  - `unclear_vendors: [vendor, ...]`;
  - `holdout: true` (new questions only);
  - `key_terms: [..]` (enumeration; for the labelling keyword search).
- Commands:
  - `uv run python eval/cross_vendor.py candidates --question-ids ... --out DIR`:
    - For each question and each corpus vendor, write `DIR/<qid>/<vendor>.json` with the
      top 10 `hybrid_candidates` scoped to that vendor, plus FTS hits for each of
      `key_terms` (top 5 per term).
    - Each record holds chunk_id, doc_id, title, section_path, text and url.
    - Free: one query embedding per question.
  - `uv run python eval/cross_vendor.py score [--rerank] [--min-score X] [--fts-weight W]
    --out FILE`:
    - Runs every question through the system's retrieval: shape detection, then the
      matching mode, `use_rerank` only with `--rerank`.
    - Writes per-question results and the summary metrics.
    - `--legacy`: every question goes through today's `search_docs` with scope detection,
      i.e. the production baseline before the new modes.
- Scoring functions (pure, unit-tested):
  - `vendor_recall(expected: set[str], unclear: set[str], hit_vendors: set[str]) -> float`:
    unclear vendors are excluded from both sides.
  - `vendor_noise(expected, unclear, hit_vendors) -> list[str]`.
  - `facet_recall(gold_doc_ids_by_facet: dict[str, set[str]], hits) -> float`.
  - `pool_recall`: reuse `retrieval_compare._first_gold`.
- The summary reports, each split into calibration and holdout:
  - enumeration vendor recall (mean) and noise;
  - comparison facet recall;
  - single-vendor and multi-article pool recall;
  - rerank tokens per question (when `--rerank`);
  - p50/p90 latency per shape.

- [ ] **Step 1: Write failing unit tests** for `vendor_recall`, `vendor_noise` and
  `facet_recall`, including: unclear excluded from both sides; expected empty → recall
  reported as None, not 0; a facet missing from the hits → 0 for that facet.
- [ ] **Step 2: Run** them and expect a FAIL.
- [ ] **Step 3: Implement** the script and the README section.
- [ ] **Step 4: Run** the gate and expect a PASS. Then smoke `candidates` and `score` (free
  mode) against a testcontainer-seeded DB, or skip if impractical, and say so in the report.
- [ ] **Step 5: Commit** `feat(eval): cross-vendor candidates and vendor/facet recall
  scoring`.

### Task 5: Labels, new questions, baseline (controller-run, no product code)

Operational. The controller runs it with Opus subagents and the production port-forward
(`docs/deploy/k3s.md`, "Development access").

- [ ] **Step 1: Shapes.** Add `shape` to the 20 cross-vendor questions (11 enumeration,
  9 comparison per spec §3.1), and `key_terms` to the 11 enumeration questions.
- [ ] **Step 2: Write ~10 new enumeration questions** (`holdout: true`, `decides: false`,
  `origin: cross_vendor_43`) with Opus subagents. They are written from the 39 new vendors'
  documents, on capabilities at least 3 vendors document, following `eval/README.md`'s
  rules.
- [ ] **Step 3: Candidates.** Run `cross_vendor.py candidates` for all enumeration
  questions, against production.
- [ ] **Step 4: Label.** Run one Opus subagent per question, reading `DIR/<qid>/*.json`. It
  decides yes, no or unclear per vendor under spec §9.6, citing chunk ids for each yes. The
  result fills `expected_vendors` and `unclear_vendors`.
- [ ] **Step 5: Spot check.** A second Opus subagent re-labels a random 10% of the
  (question, vendor) pairs. Report the agreement rate. Disagreements become `unclear`
  unless one side cites a chunk that settles it.
- [ ] **Step 6: Baseline.** Run `cross_vendor.py score` (free mode) on production at the
  current main, i.e. before Task 3's dispatch is live. Use `--legacy` for the baseline. Then run it
  with the new modes in free mode.
- [ ] **Step 7: Commit** the `questions.yaml` changes and the baseline JSON
  (`eval/runs/cross-vendor-baseline-2026-11-06.json`).

### Task 6: Floor calibration and keyword weighting (paid rerank — needs the user's OK)

- [ ] **Step 1: Ask the user** for the rerank run: about 21 enumeration questions × ~60K
  tokens plus comparisons, about 1.5M rerank tokens in total. Smoke-test 2 questions first
  and confirm `rerank_ok` and nonzero rerank tokens (Global Constraints).
- [ ] **Step 2: Calibrate** `ENUMERATION_MIN_SCORE` on the calibration questions only: the
  highest floor that keeps enumeration vendor recall ≥ 0.8. Report holdout recall at that
  floor.
- [ ] **Step 3: Keyword weighting.** Score `--fts-weight` at 1.0, 0.5 and 0.25 (free mode is
  enough for pool recall). Adopt a lower `FTS_WEIGHT` only if all-question pool recall does
  not drop and single-vendor stays within 0.02.
- [ ] **Step 4: Set the constants**, add a short report
  `docs/superpowers/cross-vendor-retrieval-2026-11-06.md` (numbers against targets), and
  commit `feat(search): calibrated enumeration floor[, FTS weight]`.

The answer-level eval (paid LLM) is out of this plan. It is proposed to the user afterwards.
