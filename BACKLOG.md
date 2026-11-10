# Backlog

Open work as of 2026-11-06, after the cross-vendor follow-ups and the M3/M4 pilot. Each item says
why it matters and what "done" looks like. Sources:
`docs/superpowers/cross-vendor-retrieval-2026-11-06.md` and
`docs/superpowers/m3-m4-pilot-2026-11-06.md`.

Items marked **$** spend money (OpenRouter, Anthropic or Voyage) and need a go-ahead first.

## Status after the full-corpus run (2026-11-08)

Results: `docs/superpowers/full-corpus-2026-11-08.md`.
- **Done:**
  - Item 3: `extract --unextracted`.
  - Item 4: the whole corpus is extracted by the distilled 9B (217,037 chunks; 54 errored).
  - Item 12: the MCP server is deployed on the LAN.
  - Item 13: on the capabilities side.
  - Item 18: the extractor; the Azure deploy is pending.
  - Item 22: production rollout of v2.
  - Every claim is mapped (Haiku, $28.50), and capabilities hold 458,785 rows.
  - All 423 in-scope comparison-table rows have a fact.
- **Partly done:** item 20 (dimension precision 0.84 -> 0.886 with the evidence quote).
- **New:** items 24-28 below.

## Status after the taxonomy v2 / claims v3 branch (2026-11-06)

Results: `docs/superpowers/taxonomy-v2-claims-v3-2026-11-06.md`.
- **Done:**
  - Item 1 became claims-v3/v3.1, at 96/100 precision with Opus.
  - Item 2 was superseded: the teacher set re-ran its own cut-off chunks at 8,000 tokens.
  - Item 5: the Anthropic key now works.
  - Item 6 was superseded by taxonomy v2 (PR #1 closes).
  - Item 7: the mapper adjudicates with an LLM.
  - Item 15, M6: prices are per model, and an unpriced model is refused.
  - Item 16, M9 (partly): embeddings are float32 and sliced, but keys are still re-embedded on
    every run.
- **Opus teacher set for item 18 is captured:** 10,000 chunks (9,500 train and 500 held out),
  27,126 claims, $65.09.

## Claims (M3)

1. **Closed 2026-11-09 (stale):** the claims-v2 pilot prompt was superseded by claims-v3 and the
   distilled 9B; claim quality is now tracked by item 36.
2. **Closed 2026-11-08 (superseded):** the 335 pilot chunks that hit the output limit were
   re-extracted with the rest of the corpus under claims-v3 (item 18).
   Fold into item 1.
3. **Done 2026-11-08: `vendorkb extract --unextracted [--vendor]`** (and `vendorkb refresh`). Claims cascade-delete with their chunks, and
   the pilot id files name chunk ids that a re-sync replaces. Nothing finds "chunks with no `done`
   extraction for the current model and prompt", so coverage decays with every sync (review M4).
4. **Done 2026-11-08: extraction beyond the pilot.** About 227k chunks remain. The plan is the Opus-distilled
   local model on Vast (item 18), not an API model. Do it after item 1, so the corpus is
   extracted with the good prompt.
5. **Closed 2026-11-08:** the Anthropic API key works since 2026-11-06 and the Opus teacher set
   was captured (item 18). Was: **A valid Anthropic API key.** The `sk-ant-usr` token in `.env` returns 401, so the pilot
   ran through OpenRouter. The Opus teacher capture (item 18) needs it for Batches pricing. The Batches path (`--mode batch --backend anthropic`) is tested only
   with fakes and has never run live.

## Taxonomy and capabilities (M4)

6. **Closed 2026-11-08 (superseded by v2, item 22; PR #1 closed, branch kept):** Was: **Review taxonomy v1 on PR #1** (https://github.com/carev01/vendorkb/pull/1, branch
   `taxonomy-v1`). It has 2,071 features and 969 targets. Known issues:
   - duplicate-looking targets under different parents (`platforms.kubernetes` vs
     `kubernetes.cluster`);
   - generic targets (`backup.data`);
   - an overfull Management category (729 features);
   - wrong aliases.

   The budget ran out before proposals for the smaller clusters.
7. **Done (2026-11-07, items 20 and 23):** Was: **$ LLM adjudication in the mapper** (DESIGN.md §5.4 step 3). Nearest-neighbour mapping is
   15–60% precise at every cosine band measured, so until this exists only alias mappings are
   trustworthy.
8. **Closed 2026-11-08 (not adopted):** adding capability-table vendors to `answer()`'s
   enumeration candidates traded precision for recall on the 23 labelled questions (recall
   0.952 -> 0.984, precision 0.877 -> 0.842, partly circular), and `answer()` would first have to
   map the question to taxonomy filters, which nearest-neighbour mapping does poorly. Agents
   already combine the table and search through the MCP tools (agent gate: recall 96%,
   precision 95%); `answer()` stays retrieval-only, now with the item 11 second pass.
9. **Calibrate the remaining thresholds.** The clustering (0.12), mapper (0.80) and resolution
   (0.85) thresholds are all uncalibrated.

## Retrieval and answers

10. **Done 2026-11-08 (late).** The 0.833 was mostly the metric: a blind side-support grade
    (Opus and Fable: does each side's passages answer that side?) found all 3 "missed" Azure
    sides supported by other pages, now added to `gold_sources` (marked `graded`). It also found
    two real faults the article metric could not see: cv-12 compared Storware's "Backup and
    Recovery" instead of Microsoft's Entra product (a generic product name matched as a facet;
    such names now need their vendor's name, and "Entra Backup and Recovery" is an alias), and
    duplicated pages (one page in two guides) took two of a side's five slots (now skipped).
    Result on the 9 questions: facet recall 1.0, and 16 of 18 sides supported by both graders,
    1 split (cv-12 Entra: on-demand backup not stated) and 1 partial (ex-rt-08 Veeam: the
    immutability pages never reach the candidates, since the question does not name immutability;
    query expansion would be the lever).
11. **Done 2026-11-08.** Diagnosed on the 23 labelled questions (419 expected vendors): 362
    listed; 32 had passages the model wrote no line for (a 60-passage prompt makes it skip
    vendors, e.g. Acronis pre/post scripts at rerank 0.885); 24 fell below the relevance floor; 1
    line was dropped by the citation check; one answer was cut off (reasoning used the 8,000
    tokens). Fix: a second call over the kept vendors the first answer left out, with only their
    passages (listed 362 -> 386, unlabelled vendors 16 -> 23, about $0.005 an answer), a
    16,000-token budget, and a closing line that tells "Passages found but not judged to show
    it" apart from "No supporting passage found for" (expected vendors in the latter 57 -> 24,
    all below the floor).
12. **MCP server** (DESIGN.md M2/§6). *Built 2026-11-08: `vendorkb mcp`, seven read-only tools,
    bearer token, Helm Deployment/Service/LAN ingress. Open: a public HTTPS route with OAuth
    for Copilot Studio, and an agent-level eval (cross-vendor answers with query_capabilities
    vs search only, the M4 done-when).*

## Final-review minors (M3/M4 branch)

Left open at merge. I1, I2, M1, M8, M10 and M11 were fixed.

13. **Done 2026-11-08 (capabilities side): `capabilities.build` takes vendor and product from
    `documents`.** `raw_claims` still keeps the extraction-time values, and an offering mapped
    under the old vendor is not re-mapped. **M2: stale vendor/product on re-tag.** `store.update_metadata` re-tags a document without
    touching chunks, so `raw_claims` and `capabilities` keep the old vendor and product. Fix:
    take both from `documents` through `chunks` in `capabilities.build`.
14. **Done 2026-11-08 (documented as reserved):** DocExtractor's delta records carry no version
    (only its products endpoint has one, per product), so `documents.version` and every
    claim's `version` stay NULL and answers come from the latest documents; `schema.sql` says
    so where the column is declared. A product-level version would label every document of a
    product with the current release, which is wrong for older pages.
15. **Done 2026-11-08:**
    - **M5:** the batch ledger row is written under a `pending-` id before `batches.create`
      and renamed to the provider's id; a row left by a crash keeps its estimate in the budget
      and holds its chunks back (`unconfirmed_batches`, `held_chunks`) until a person
      reconciles it. A 4xx refusal deletes it.
    - **M6:** unknown-priced models were already refused (`price_per_mtok`); now a paid reply
      that fails to parse is recorded errored (`parse:<Error>`) with its usage and cost, and
      still stops the run.
    - **M7:** `taxonomy propose` takes an advisory lock and ledgers every run in `propose_runs`
      (spend updated per call), with a cumulative cap `propose_budget_usd` (default $20).
16. **Done 2026-11-08 (item 27):** keys are cached in `embedding_cache` and held as float32.
    Was: **M9: `taxonomy map` memory and cost.** It re-embeds about 24k keys on every run and holds
    them as Python float lists, more than 2 GB. Fix: convert to float32 per batch, and cache key
    vectors by (text, model).
17. **Done 2026-11-08 (the lock half):** `init_schema` skips its DDL when the stored schema
    marker matches and every table exists, and otherwise waits at most 10 s per lock (an ad-hoc
    read held the sync job's DDL for 24 minutes, queueing readers behind it).
    **M12 done 2026-11-08:** a test re-runs the whole DDL over a database with rows in every
    table the services write and checks every count. The locks: `ADD COLUMN IF NOT EXISTS`
    takes ACCESS EXCLUSIVE even when the column exists, but the DDL now runs only when the
    schema marker changes (a deploy), bounded by the 10 s lock timeout, so every other
    `init_schema` call takes none; no change.

## Local extractor (DESIGN.md M5: trained on Vast, served on Azure)

18. **Fine-tune a local corpus extractor, distilled from Opus** (user decision,
    2026-11-06). The project aims for near-perfect claims, so the teacher is Claude Opus 5.5,
    not the Haiku pilot output. Training and the full-corpus run both happen on Vast.
    - **Why distil:** running Opus over the whole corpus directly would cost an estimated
      $1.5–3k. A 9B student on one rented GPU with vLLM should cost tens of dollars per full
      run, so re-extractions after prompt or schema changes stay cheap.
    - **$ Teacher capture:** build a stratified set of a few thousand chunks, covering all 43
      vendors, empty chunks, tables and procedure-heavy pages. Extract it with the v2 prompt (item
      1) on Opus through the Anthropic Batches API (needs item 5). Keep only claims that pass the
      evidence gate. Pricing is $4/$20 per MTok and Opus thinking cannot be disabled, so smoke a
      few chunks and measure real cost per chunk before sizing the set.
    - **Teacher quality bar:** a ≥95% spot-check before any training. Opus must not grade its
      own output alone: use a human-checked sample, or a grader that is not the teacher.
    - **Student candidates:** Qwen3.5-9B and Qwen3.8-27B. Train both LoRAs on the same Opus
      data, which is the expensive part and is shared, and pick by held-out agreement. 27B in
      bf16 needs about 56 GB, so it fits one 80 GB A100 (Azure Container Apps, validated
      2026-11-06) but not the P6000.
    - *2026-11-08: the corpus is extracted (217,037 chunks by the 9B, 54 errored), and
      `deploy/azure` serves the adapter on a serverless A100 for `vendorkb refresh` (deploy
      with the user's Azure login; LoRA serving validated against the merged model).*
    - **Serving:** vLLM's OpenAI-compatible API on Azure Container Apps (A100, scale to zero) for
      incremental extraction, through the existing `openai` extract backend. Batch the deltas
      to amortise cold starts.
    - **Student bar:** ≥90% agreement with Opus on a held-out set, and precision within a few
      points of it. Then run the full corpus on Vast, replacing item 4's API route.

## Taxonomy v2 follow-ups

19. **Done: taxonomy v2 reviewed by Claude** at the user's request. The decision log is
    `taxonomy/review-2026-11-06.tsv`, and the user may spot-check any row.
20. **Measured 2026-11-08 (late), nothing adopted:** on a fresh labelled sample (200 model-decided
    claims, two blind graders, `eval/labels/dimension-sample-2026-11-08.json`) dimension
    precision is 0.904 and recall 0.851. 25 of 31 errors are retrieval: the right dimension
    ranks about 25th by phrase embedding, so the model never sees it. Replayed on the sample
    (`eval/map_replay.py`, about $0.03 a variant): 8 or 10 candidates add distractors (the right
    dimension enters the candidates for 2-3 more items); a keyword second pass for none answers
    gains as many wrong picks as right ones; keyword or fused retrieval finds the right
    dimension less often (110 and 140 of 174 in the top 5, against 149). Left: retrieval that
    reads the evidence quote, which changes every question's candidates (a full re-ask, about
    $28, plus Voyage tokens). The user prefers no stronger map model.
21. **Done 2026-11-08:** curated the 936 unmapped offering texts with 5+ claims (125 new
    offerings, 270 aliases; blind review 425/431, precision 80/80). Claims naming an offering that
    resolve: 39% -> 70%; offering-scoped capability rows 12,154 -> 21,909. Left: 15,173 claims,
    mostly texts judged not an offering and the long tail under 5 claims. (Was: recall 0.59.)
22. **Done 2026-11-07: production rollout of v2.** The order matters, because the branch's `init_schema` drops
    the v1 tables:
    1. Take a database backup first. The upgrade is one-way: the old image's schema fails on a
       v2 database, so an image rollback breaks every sync run (see `docs/deploy/k3s.md`,
       "Upgrading to taxonomy v2").
    2. Check `pg_depend` for objects depending on the v1 tables.
    3. Merge and deploy.
    4. Load the reviewed taxonomy, then map and build capabilities.
    5. Re-measure findability.

    The teacher and precision claims are already in production under claims-v3 and claims-v3.1.
23. **Done 2026-11-08: adjudication decisions persist** in `map_decisions` (keyed on prompt
    rules, model family, kind, phrase, class and candidates), so a re-run or a re-map after a
    taxonomy change pays only for new questions; `taxonomy map --seed-decisions` records the
    decisions the stored mappings imply. **Persist mapper adjudications** so re-runs and a
    lower `--max-usd` do not redo paid calls.
    Also keep a persistent list of members propose rejected.

## After the full-corpus run (2026-11-08)

24. **Done 2026-11-08:** a claim whose target text matched no taxonomy target gets scope
    `target_unmapped` (65k of them had counted as product-wide), and a conflict is now the same
    statement (normalized capability phrase, target scope text, conditions) both supported or
    partial and not_supported, or with differing values: measured 0.5% of rows, down from 65%.
    **Conflict groups are too coarse.** 65% of capability rows sit in a conflict group. They
    concentrate in a few large keys, such as a product-wide dimension with thousands of claims.
    A positive-versus-not-supported-only rule still flags 59%. Add the conditions (or the
    target scope) to the key, or report conflicts only within one document set. Done when
    conflict groups are rare enough to read.
25. **Deploy the Azure extractor and enable `claimsRefresh`** (`deploy/azure`, docs/deploy/k3s.md
    §7a). This needs the user's Azure login and serverless A100 quota. Weekly is enough (user,
    2026-11-09): the CronJob now runs Sundays 03:45. With item 37's batch pipeline, only the
    extraction should stay weekly; the batch stages need a frequent, cheap collect run, or a
    new page would take four weeks to reach capabilities.
26. **Done 2026-11-09: the MCP server in Copilot Studio** (spec
    `specs/2026-11-09-copilot-mcp-design.md`, setup docs/deploy/copilot-studio.md):
    - the public route `https://vendorkb.area51.pp.ua/mcp`, only `/mcp`, through Cloudflare;
    - one shared secret, rotated, accepted as `X-API-Key`; access is controlled by sharing the
      agent;
    - `read_document` for a cited page's full text;
    - agent instructions that relay vendorkb's cited answers.

    Veeam's tenant accepted the connection. Open:
    - the Competitive Intelligence team's feedback;
    - ~~an app-level rate limit~~ done 2026-11-09: Traefik RateLimit middlewares, per client
      (`CF-Connecting-IP`) and global;
    - Authenticated Origin Pulls, so only Cloudflare can reach the origin;
    - `PART_CHARS` against Copilot's real limit;
    - hiding `search_docs` and `query_capabilities` if Copilot composes its own answers from
      them.
27. **Done 2026-11-08:** `embedding_cache` keeps each (model@dimension, input type, text)
    vector as raw float32 bytes, read back bit for bit, so a re-map embeds only strings it has
    never seen (`embeddings_cached` / `embeddings_new` in the map report). Strings over 1,000
    characters are never cached.
29. **Done 2026-11-08 (round 2):** Opus curators decided the 1,456 unmapped target strings named
    by two or more vendors or backing 15+ claims, and a blind Fable review accepted their
    proposals: 177 new targets, 621 target aliases, 72 variant aliases, and four Windows aliases
    moved from `windows_server` to `microsoft_windows`. Claims with an unmapped target fell from
    61,118 to 48,006, and 461 targets now have facts (286 before). Left: 48k claims over 7.3k
    strings (about half on single-vendor strings, 11.6k on strings the curators judged generic
    or not a workload), and 127k claims with no dimension. Round 2's 1,456 strings gave 13k
    claims; the next slice gives less.
30. **Closed 2026-11-08:** the "ten" gaps were ten phrases that collapsed into six dimensions,
    all added. The round-2 dimension curators found 95 gap phrases in about 15 themes: online or
    offline database backup (~140 claims), console password policy (~100), database-native
    backup interfaces such as RMAN or Backint (~60), NORECOVERY or STANDBY restores (~50), UEFI
    and Secure Boot (~45), off-host backup (~40), and smaller ones (direct-to-cloud backup,
    white-label branding, cross-version restore, interface languages). None of them appears in
    the comparison tables. Four were added later the same day as buyer-relevant
    (`protection.online_database_backup`, `protection.native_database_interface`,
    `recovery.uefi_secure_boot`, `protection.off_host_backup`; 3,264 capability rows). Seven more
    followed that night (`security.password_policy`, `recovery.database_recovery_state`,
    `protection.replica_backup_source`, `operations.interface_languages`,
    `recovery.cross_version`, `operations.white_label_branding`, and `protection.direct_to_cloud`
    as a lookup-only dimension; 2,582 rows); hot/cold backup became aliases. Product lifecycle, certificate lifecycle,
    accessibility and the 1-3-vendor tail were judged not worth a dimension (report section
    "Seven more gap themes").
31. **Done 2026-11-08:** dimensions and targets take `lookup_aliases`, matched like aliases but
    never candidate-search keys (`aliases.search` false). The 194 held round-2 dimension aliases
    went in as lookup aliases with 0 questions asked again (about 90k as search keys). A blind
    check of 50 found 44 correct; the wrong patterns went, leaving 173 (188 lookup aliases with
    the new dimensions', mapping 2,228 claims). A new entity still re-asks the phrases near it, but an entity chosen before for
    the same question now stays among its candidates. Adding four dimensions had displaced the
    earlier answer of 438 questions; with the fix, 1,089 questions went back to an earlier pick.
28. **Done 2026-11-08:** `eval/agent_eval.py prepare|call|score` with `eval/agent_baseline.json`
    (23 questions, recall 96.0%, precision 94.9%; a mean dropping more than 5 points fails).
    Report: `docs/superpowers/agent-eval-gate-2026-11-08.md`.

## Topic expansion (2026-11-09)

32. **Done 2026-11-09:** facts survive page updates (PR #32: superseded chunks, chunk reuse; 46%
    of updated chunk tokens need no re-extraction) and the ransomware protection topic (PR #33:
    eight aspects, per-aspect retrieval and answers, MCP `query_capabilities(topic=)`). Blind
    eval PASS: aspect coverage 67.0% -> 98.9%, unsupported-claim rate 2.39% -> 0.72%; agent gate
    PASS (recall 97%, precision 96%). Report: `docs/superpowers/topic-expansion-2026-11-09.md`.
33. **Done 2026-11-09: the "which vendors" grid is on (`TOPIC_GRID=true`, helm rev 28)** after the
    mapping fix (BACKLOG 34) brought it to 95% adjudicated precision (pooled unsupported rate +0.46
    pts, bar +2). Before the fix: Its precision is the capability facts'
    (blind-graded 87-95% of sampled cells supported with cue gating) against 98-100% for today's
    prose enumeration. Path to enable: a cheap model check of each cell ("does this passage state
    that <vendor> has <aspect>?"; ~200 short calls per grid question), then re-run the topic eval
    with `--expect-topic` and the grid sampling already in `eval/topic_eval.py`.
34. **Done 2026-11-09 (PR #35, taxonomy 2-48ffb013):** sharper definitions and alias removals
    for 16 security/cyber dimensions, `cyber.threat_detection`; re-asked dimensions 74% -> 79% on
    the right dimension, grid cells 87% -> 95%. RBAC was not re-asked. Originally: The grid grading found facts
    mapped to the wrong dimension ("Inactive files" -> anomaly detection, "Dedicated backup
    networks" -> air gap, AI-agent session details -> deletion safeguards); they also reach
    `query_capabilities`. A blind sample per security/cyber dimension would size it.
35. **Topic retrieval time: cause found and fixed 2026-11-10 (branch
    `topic-speed-and-full-passages`).** `chunks.superseded_at` (added 2026-11-09) had no planner
    statistics: adding a column changes no row, so autovacuum never analysed the table (last
    analysed 2026-11-06). The planner took `superseded_at IS NULL`, which every search carries,
    for 0.5% of the chunks, and instead of the HNSW index it computed the distance to all
    227,613 chunks and sorted them (on disk, 108 MB): 2.2 s a vector search. A two-vendor
    topic question makes 18 of them.

    `schema.sql` now runs `ANALYZE chunks (superseded_at)` (a test checks the statistics exist
    after an upgrade). Measured on production with the statistics inside a rolled-back
    transaction, retrieval only, first call then warm:

    | question | before | with statistics |
    |---|---|---|
    | Veeam vs Commvault ransomware comparison | 44-46 s | 9-14 s, then 4 s |
    | one vendor's ransomware protection | 25-27 s | 5-6 s, then 3 s |
    | a single-vendor question | 5.5 s | 2.4-3.7 s |
    | "which vendors" question | 8 s | 8 s |

    The "which vendors" pool query (2,000 neighbours over all vendors) is now an exact sort on
    purpose (`search.ENUM_VEC_SQL`): it was one by accident, the HNSW index reaches 2,000 rows
    only by iterating (5-11 s, and other passages), and item 43's floor and depth were
    calibrated on the exact order. `answer_question` and `gather_evidence` log their seconds.

    **Deployed 2026-11-10 (image `sha-ba3d50e`, helm rev 35), measured on the live server:**
    the Veeam-Commvault topic question's retrieval 4.8 s alone (46 s before); four topic
    questions at once 6-9 s each (120-210 s in the pilot); two "which vendors" and two topic
    questions at once 9-14 s.

    **The full-text statement, profiled 2026-11-10.** It is now the largest part of every
    retrieval: 2.9 s for an unscoped question, 1.1 s scoped to one vendor, 3.9 s for the
    "which vendors" statement. The question's terms are ORed, so "backup" (45% of chunks) and
    "support" (23%) make it match 60% of the corpus, and the time is `ts_rank_cd` over every
    match (134k rows), after a sequential read of every chunk's `tsv`. Leaving out the lexemes
    in more than a share of the chunks when rarer ones remain (read from `pg_stats`, no scan)
    lets the GIN index serve it: 1.1 s, 0.2 s and 0.4 s for the three statements. Prototype on
    branch `fts-common-words` (`search.fts_query`, `FTS_COMMON_FREQ`). The free eval on the
    labelled questions (no rerank), rule off / over 20% / over 10%:

    | | off | 20% | 10% |
    |---|---|---|---|
    | single-vendor pool recall (60) | 0.950 | 0.950 | 0.950 |
    | single-vendor recall@10 before the rerank | 0.917 | 0.900 | 0.900 |
    | multi-article pool recall / recall@10 (19) | 1.000 / 0.895 | 1.000 / 0.895 | 1.000 / 0.947 |
    | comparison facet recall (9) | 0.833 | 0.889 | 0.889 |
    | single question, p50 / p90 seconds | 3.0 / 8.9 | 1.6 / 4.7 | 1.2 / 2.5 |
    | "which vendors", p50 seconds (calibration / holdout) | 9.5 / 6.3 | 7.0 / 4.0 | 4.7 / 3.5 |

    Not adopted yet: the free mode keeps every vendor, so it does not show what the change does
    to the reranked "which vendors" result that item 43's floor was calibrated on. That needs
    the 23 labelled questions with the rerank, before and after (about 7M rerank tokens).

    Left:
    - the dropped connections of the pilot: cause not found (not seen since the fix).

    *2026-11-10, from the item 42 pilot:* with four to six agents at once, a topic
    `answer_question` or `gather_evidence` took 120-210 s (45-60 s alone), past the public
    route's 100 s. The database was busy with the vendor-filtered vector searches of the
    per-aspect retrieval (about 1.2 cores); four calls also lost their connection mid-reply
    with nothing in the server log. Topic retrieval time is the next thing to fix here.

    **Topic residuals (parked in review):** offerings are not masked from triggers (products and
    their aliases are); single-vendor questions narrowed by a word ("…for NAS backups?") still get
    the full kit (Ruling 12: a missed cue costs length, not a wrong claim); topic answers take
    27-80 s and 109-148k rerank tokens (vs 8-23 s, 12-23k); `vendorkb stats` counts superseded
    chunks; MCP does not surface `catch_up_ok`/`summary_error`; the grid's catch-up has no
    per-vendor cap.
36. **Claims whose quote does not establish them for the vendor.** On 1,078 security facts
    with consensus gold labels (Opus + Fable, 96% agreement), 9.8% of the distilled 9B's facts
    are unsupported, vs 0/57 of Opus's (the earlier 15.6% used a brief that counted
    requirements and limits as unsupported). **Done 2026-11-09 (`claims/support.py`, spec
    `specs/2026-11-09-claim-support-check-design.md`):**
    - The checker is Haiku 5.5 at effort high: it flags 41% of the unsupported facts and 3.3%
      of the supported ones.
    - Opus 5.5 at effort low reviews every flagged claim. On the Anthropic API it kept 32/32
      wrongly flagged facts and confirmed 32/41.
    - End to end: 32% of the unsupported facts are removed, and none of the 978 supported
      ones.
    - Only rejected claims leave capabilities.
    - Scope: the topic dimensions, also in `vendorkb refresh`, on the Anthropic API.

    Retraining the 9B was not tried: graph-rag's DPO round (dpo4) made the student terse, not
    selective. Open:
    - other dimensions (~$15-20 with Haiku for every fact, plus the review);
    - target verification;
    - the Message Batches API for half price (item 37);
    - the mapper's per-claim handling of 401/402/404 (it spends its budget on a misconfigured
      endpoint; `support` stops instead).
37. **Every Claude call on the Message Batches API** (study approved 2026-11-09, build week of
    2026-11-16, lower priority; spec `specs/2026-11-09-message-batches-design.md`).
    - **Scope:** taxonomy map, support check and review, propose. Extraction already has a batch
      mode.
    - **Why:** half price, and nothing is urgent.
    - **The refresh:** every stage runs as a batch (user decision). New facts typically appear
      within a few hours, at worst after about 4 days.
    - **Order:** the runner and ledger, support, map (two rounds), the refresh pipeline, then
      propose (optional).
    - **Production:** the map moves to the Anthropic API, and `ANTHROPIC_API_KEY` goes into the
      secret.
38. **Closed 2026-11-09: Copilot refused to relay full documents.** The refusal came from the
    agent's default GPT model, a rule built into the model. With the agent's model switched to
    Claude Haiku in Copilot Studio, it relays `read_document`'s full page (the user).
39. **A sister brand's product credited to the wrong vendor** (user, 2026-11-09). Asked "Which
    vendors support backing up Entra ID?", the answer said "Spanning: provides a dedicated
    product, Datto Backup for Microsoft Entra ID", and listed Datto under "No supporting passage
    found for". It should be the other way round: Datto has the product, Spanning does not.

    Not a data gap (corrected 2026-11-09). Datto BCDR documents it: a "DATTO BACKUP FOR
    MICROSOFT ENTRA ID" page plus "What is backed up" and "Restoring". Capabilities hold 23
    `coverage.backup` / `entra_id` facts for Datto, plus 11 export and 8 restore facts.

    So the "which vendors" answer missed Datto's own facts. It then credited a Spanning page's
    mention of the Kaseya sister product to Spanning.

    **Diagnosed 2026-11-09 (Copilot transcript replayed):** the agent (GPT then) did not send
    the user's question. It sent "Which vendors support backing up Microsoft Entra ID? List all
    vendors documented, distinguishing supported versus partial support and specifying any
    conditions or product names, with sources." The appended instructions are part of the
    rerank query and lower every score: Datto's best passage 0.624 -> 0.515, under the floor
    (0.54), so Datto went to "No supporting passage found for" (Eon and Synology too).
    Spanning's "IT Complete modules" page, a list of Kaseya help links naming "Datto Backup for
    Microsoft Entra ID", scored 0.586 and was kept; the model wrote a Spanning line from it,
    and the line check only counts a line that names another vendor (`_names_other`). The
    user's own wording lists Datto and not Spanning. `answer_question` now logs the question
    as sent.

    *2026-11-10:* with the rule in place, all 24 relay and evidence agents of the item 42 pilot
    (Haiku) passed the question word for word. Copilot's own compliance is still to be read
    from the question log.

    **Done 2026-11-09 (branch `enum-workload-product-and-verbatim`):**
    - "Pass the user's question word for word" is in the `question` parameter's description,
      the tool's description, the server instructions and rule 1 of the agent's instructions
      (docs/deploy/copilot-studio.md §5; paste the new rule 1 into Copilot Studio).
    - "Which vendors support Entra ID backup?" read as a single question on Microsoft's product
      "Entra ID Backup" and was answered from Microsoft's pages only. With "vendors" wording, a
      workload vendor's product now enumerates like the workload vendor (`shape.detect_shape`),
      unless the product is what is asked about ("which vendors does Azure Backup integrate
      with?"). No labelled question changes shape.

    Open:
    - whether the agent complies: read the question log after the next Copilot tests. If it
      still pads the question, retrieve on the text up to the last question mark (the padded
      question cut there scores Datto 0.659, Spanning 0.419; no labelled question has text
      after its last question mark, so measure on labelled questions with the agent's sentence
      appended);
    - the Spanning line. Not a blanket drop of lines naming another vendor: of 419 stored
      enumeration lines, 6 name a non-workload vendor and all 6 are true integrations (Velero,
      Storware, Dell storage). With the user's own wording the page is still kept (0.667) and
      the line is the model's choice; a prompt rule is the next step if it recurs
      (`lines_naming_other_vendors` counts such lines).

    Was, to investigate:
    - why the enumeration's per-vendor retrieval and relevance floor (`enumeration_min_score`)
      found no passage for Datto;
    - whether the answer should credit a product named in the text to its own vendor
      (offerings / scope aliases) rather than the page's vendor;
    - possibly the same root cause as item 40: facts not feeding the answer path.
40. **Topic comparisons ignore most of the facts** (user, 2026-11-09). "Compare ransomware
    protection capabilities between Veeam and Commvault" got a poor answer:
    - Only "threat/anomaly detection and malware scanning" was good.
    - It said "No documentation of MFA, encryption standards, or RBAC details appears in the
      provided Commvault sources". Yet capabilities hold, for Commvault on the ransomware
      dimensions, 971 encryption facts and 1,595 built-in security facts.
    - It said Veeam has no documented cyber recovery testing; SureBackup is in the corpus.
    - Two vendors × eight aspects got only 10 sources, the Veeam side mostly from one Vault
      Security page.

    **Fixed 2026-11-09 (branch `fix-topic-named-aspects`):** the question as asked takes the
    topic path and is answered well (37 passages, all cited, eight aspects for both vendors).
    The Copilot agent had sent a rewording that names the aspects ("…: immutability, air gap,
    anomaly detection, malware scanning, MFA, encryption, RBAC, cyber recovery testing"). A
    named aspect cue turned the topic off, so the plain comparison answered eight subjects from
    five passages a side (10 sources; reproduced, with the reported gaps). Now a single or
    comparison question that names two or more aspects of a triggered topic keeps the topic
    path for those aspects (`topics.detect_named_aspects`): the same wording gets 27 passages,
    all cited, with Commvault's MFA and encryption. No labelled eval question changes path.
    Left: a subject outside the kit ("cyber recovery testing") has no retrieval of its own, so
    Veeam's SureBackup is still not found for it.

    Was: Investigate the comparison path for a topic question (`shape` comparison with a topic):
    - per-vendor, per-aspect retrieval quotas;
    - whether the grid/fact chunks (`fact_chunks_many`) feed it;
    - the rerank budget;
    - the answer prompt's use of the citations.

    Done when the same question cites evidence for every aspect each vendor documents.
    Measure with `eval/topic_eval.py` plus the user's question, before and after.
41. **The Copilot agent drops the [N] markers** (user, 2026-11-09). `answer_question` returns
    the answer with numbered citation markers and a Sources list. The agent's reply loses the
    markers, so a statement no longer points to its source.

    To investigate:
    - which model and instructions are in use;
    - whether rule 1 ("keep every [N] marker") is followed under Claude Haiku;
    - whether Copilot Studio's generative-answers formatting strips bracketed numbers. If so,
      a marker form it keeps (e.g. "(source 3)" or a superscript), offered as a
      `citation_style` option of `answer_question`.

    Done when an agent reply keeps a marker on every cited statement.

    *2026-11-10:* in the item 42 pilot, Haiku agents relaying `answer_question` kept every
    marker (320 of 320 over 12 answers), and the transcript of item 39 shows GPT kept them too,
    adding its own footnote numbers. So the loss is not Claude Haiku's: re-test in Copilot
    Studio with the agent on Haiku before changing the marker form.
42. **Do we need vendorkb's answer model?** (user, 2026-11-09; take with items 40 and 41,
    which touch the same answer path).

    `answer_question` does two jobs:
    - *gathers the evidence* deterministically: every vendor checked for "which vendors"
      questions, per-vendor and per-aspect comparisons, the topic grid built in code,
      relevance floors, refusals;
    - *writes* a cited answer with the answer model (GLM, `JUDGE_*`), with the uncited check.

    The Copilot agent then rewrites that answer, so two model passes stand between the
    evidence and the reader. That second pass likely drops the markers (item 41).

    **Proposal.** Keep the evidence gathering and drop the second model pass. A new tool returns
    the evidence bundle with no prose: the cited facts and passages per vendor and aspect, the
    grid table where it applies, and evidence ids to cite. The agent (Claude Haiku in Copilot
    Studio) writes the answer from it.

    **Gains:** one model pass instead of two; no GLM cost; faster answers. Today's 25–80 s is
    mostly writing and checking.

    **Cost:** the citation check moves from code into the agent's instructions, and vendorkb can
    no longer verify the final text.

    **Broad questions.** For "which vendors do X?", the agent must know every vendor and product.
    `list_vendors` already returns them, with document and fact counts. Make sure the agent uses
    it: in its instructions and the tool description; possibly with per-vendor coverage of the
    question's dimensions or topic, so that "no evidence" and "not checked" stay distinct.

    **Decide by measurement.** Run `eval/agent_eval.py` with Haiku as the agent, on the labelled
    questions plus the user's "Compare ransomware protection capabilities between Veeam and
    Commvault", in three arms:
    1. relay `answer_question`, as today;
    2. write from the evidence bundle;
    3. raw tools only (`search_docs`, `query_capabilities`, `list_vendors`).

    Score each blind on unsupported claims, vendor recall and kept citations. Drop the answer
    model if arm 2 matches or beats arm 1. For reference, the 2026-11-08 raw-tool agent eval
    with Sonnet reached vendor recall 93% and precision 98%.

    **Built 2026-11-10 (branch `enum-depth-and-evidence`), not yet measured:** the MCP tool
    `gather_evidence(question, part)` is arm 2's tool. `answer.gather` is the deterministic
    half of `answer` (shape, topic, every vendor / each side / each aspect retrieved);
    `answer.evidence` returns it, numbered as the answer numbers its sources, and the tool
    shows it grouped by vendor or aspect, in parts of 20,000 characters, ending with what was
    found nothing for and the writing rules. A grid question returns the table and its sources.
    `answer_question` is unchanged. Passages are cut at 1,200 characters (`read_section` has
    the rest), which the answer model's whole passages are not.
    Measured on production: a two-vendor ransomware comparison is 37 passages in 3 parts (56k
    characters, 46 s); "which vendors support backing up Entra ID" 45 passages in 4 parts (66k,
    9 s). So for a topic the time is the retrieval (18 rerank lists), not the writing: the
    bypass saves little there.

    **Decided by the pilot, 2026-11-10: keep the answer model**
    (`docs/superpowers/bypass-eval-2026-11-10.md`; `eval/bypass_eval.py`). Twelve questions,
    three arms, Haiku agents, blind Opus 5.5 and Fable 5.2 graders (275 of 278 checks agreed):

    | | relay | evidence | raw |
    |---|---|---|---|
    | vendor recall / precision | 85.5% / 88.8% | 82.8% / 94.7% | 64.9% / 92.0% |
    | checks unsupported | 0% | 0% | 4.3% |
    | claims without a marker | 0% | 3-6% | under 1% |
    | claims in the 12 replies | 691-807 | 484-546 | 345-354 |

    The evidence arm matches on accuracy, cites less completely, says about 30% less and is no
    faster end to end. `gather_evidence` stays as an optional tool.

    *After the pilot (user, 2026-11-10):* `gather_evidence` returns each passage whole, as the
    answer model reads it (a chunk is at most about 1,200 tokens), not cut at 1,200 characters.
    The replies are about twice as long: the Veeam-Commvault comparison is 116k characters in 7
    parts (56k in 3 before), a "which vendors" question 132k in 8. Not re-measured with agents. Not run: the full 35
    questions (the pilot cannot tell differences under about 5 points; the citation and speed
    results do not depend on that), and a Copilot Studio agent on the alternative rules.
43. **Vendors listed for a "which vendors" question moved with its wording** (user,
    2026-11-10). "Which vendors support Entra ID backup?" left Rubrik and Commvault under "No
    supporting passage found for"; "…backing up Entra ID?" listed Rubrik.

    **Cause.** Only each vendor's first 2-3 candidates (hybrid search order, which follows the
    wording) reached the reranker, and the vendor's best passage was often further down:
    Rubrik's Entra ID page was fourth and scores 0.74, Datto's product page sixth for the
    Copilot agent's padded wording (0.72; item 39).

    **Fixed 2026-11-10 (branch `enum-depth-and-evidence`):** the reranker judges six candidates
    per vendor (`modes.ENUM_RERANK_DEPTH`, calls of 100 passages run together) and the answer
    gets each vendor's best 2-3 by score. The floor was recalibrated to 0.56 on the 23 labelled
    questions (one run, 12 judged per vendor, 6.95M rerank tokens; shallower depths read from
    the same scores):

    | judged per vendor | floor | recall | noise vendors a question |
    |---|---|---|---|
    | 2-3 (before) | 0.54 | 0.956 | 2.4 |
    | 6 | 0.56 | 0.971 | 2.5 |
    | 8 | 0.56 | 0.971 | 2.7 |
    | 12 | 0.58 | 0.959 | 2.3 |

    On four wordings of the Entra ID question (17 vendors with `coverage.backup` facts for
    `entra_id`): kept in all four 12 -> 14, Rubrik and Datto among them. About 170K rerank
    tokens a question (70K before), 8-13 s for the retrieval. The grid's catch-up keeps its own
    floor (`topics.CATCH_UP_MIN_SCORE`, 0.54).

    Left:
    - Commvault (best passage 0.50-0.58: its page says "Azure Active Directory") and Eon (in
      the padded wording) still move with the wording; Bacula (2 facts) is never kept. A score
      floor cannot make this exact. The capability facts would (a vendor with facts for the
      question's target is kept whatever its score), which item 8 tried and did not adopt for
      precision; with the target read from the question by exact alias it may be worth a
      second look.
    - The kept vendors without evidence grow on some wordings (Storware, Unitrends, Synology:
      1-3 -> 2-5 on the Entra wordings); the answer model declines most ("Passages found but
      not judged to show it").

    *2026-11-10, the remaining flicker:* three identical calls kept Commvault once and dropped
    it twice, also with an exact vector sort. The embedding provider returns a slightly
    different vector for the same text now and then (1 of 4 calls: largest component
    difference 7.7e-4, cosine 0.99997), which moves a candidate across the six-per-vendor cut.
    **Fixed 2026-11-10 (branch `query-embedding-cache`):** `search.embed_queries` reads a
    query's vector from `embedding_cache` (input type "query") and stores a new one as float32,
    returning the stored value on the first call too; `search_docs`, the enumeration,
    comparisons and the topic paths use it. On production, inside a rolled-back transaction,
    five identical calls kept the same 20 vendors and the same 45 passages with one embedding
    call. A question worded differently is another text, and still moves the borderline vendors.

    **Answer level (2026-11-10):** the 23 labelled questions through the full answer, the
    deployed code and the branch side by side, one run each: expected vendors listed 385 ->
    397 of 419 (mean recall 0.933 -> 0.955), mean precision 0.932 -> 0.931, vendors listed
    that are not expected 28 -> 29, 33 -> 34 s an answer, rerank tokens 65K -> 159K a question.
