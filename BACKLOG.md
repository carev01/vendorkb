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

1. **$ `claims-v2` prompt and re-measure precision.** The pilot spot-check found 86/100 claims
   correct, against a 95% target. The errors were 7 wrong_support (mostly "partial" used too
   loosely) and 5 not_a_capability (procedures and known issues extracted as capabilities). None
   was hallucinated.
   - Narrow "partial", and exclude procedures and known issues.
   - Re-extract the pilot under the new prompt version, then repeat the 100-claim Opus
     spot-check.
   - Done when precision is ≥95%. About $10 on OpenRouter.
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
    §7a). This needs the user's Azure login and serverless A100 quota.
26. **Copilot exposure of the MCP server:** a public HTTPS route with OAuth in front of `/mcp`,
    then Copilot Studio onboarding. DocExtractor's public route (traefik with a Cloudflare
    origin certificate) is a ready pattern. The LAN DNS record for `vendorkb-mcp.k3s.home.lan`
    exists since 2026-11-08.
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
33. **The "which vendors" grid is off (`TOPIC_GRID`).** Its precision is the capability facts'
    (blind-graded 87-95% of sampled cells supported with cue gating) against 98-100% for today's
    prose enumeration. Path to enable: a cheap model check of each cell ("does this passage state
    that <vendor> has <aspect>?"; ~200 short calls per grid question), then re-run the topic eval
    with `--expect-topic` and the grid sampling already in `eval/topic_eval.py`.
34. **Capability precision on the cyber/security dimensions.** The grid grading found facts
    mapped to the wrong dimension ("Inactive files" -> anomaly detection, "Dedicated backup
    networks" -> air gap, AI-agent session details -> deletion safeguards); they also reach
    `query_capabilities`. A blind sample per security/cyber dimension would size it.
35. **Topic residuals (parked in review):** offerings are not masked from triggers (products and
    their aliases are); single-vendor questions narrowed by a word ("…for NAS backups?") still get
    the full kit (Ruling 12: a missed cue costs length, not a wrong claim); topic answers take
    27-80 s and 109-148k rerank tokens (vs 8-23 s, 12-23k); `vendorkb stats` counts superseded
    chunks; MCP does not surface `catch_up_ok`/`summary_error`; the grid's catch-up has no
    per-vendor cap.
