# Backlog

Open work as of 2026-11-06, after the cross-vendor follow-ups and the M3/M4 pilot. Each item says
why it matters and what "done" looks like. Sources:
`docs/superpowers/cross-vendor-retrieval-2026-11-06.md` and
`docs/superpowers/m3-m4-pilot-2026-11-06.md`.

Items marked **$** spend money (OpenRouter, Anthropic or Voyage) and need a go-ahead first.

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
2. **$ Re-extract the 335 chunks that hit the output limit.** Use max_tokens 4000. About $0.40.
   Fold into item 1.
3. **`vendorkb extract --unextracted [--vendor]`.** Claims cascade-delete with their chunks, and
   the pilot id files name chunk ids that a re-sync replaces. Nothing finds "chunks with no `done`
   extraction for the current model and prompt", so coverage decays with every sync (review M4).
4. **Extraction beyond the pilot.** About 227k chunks remain. The plan is the Opus-distilled
   local model on Vast (item 18), not an API model. Do it after item 1, so the corpus is
   extracted with the good prompt.
5. **A valid Anthropic API key.** The `sk-ant-usr` token in `.env` returns 401, so the pilot
   ran through OpenRouter. The Opus teacher capture (item 18) needs it for Batches pricing. The Batches path (`--mode batch --backend anthropic`) is tested only
   with fakes and has never run live.

## Taxonomy and capabilities (M4)

6. **Review taxonomy v1 on PR #1** (https://github.com/carev01/vendorkb/pull/1, branch
   `taxonomy-v1`). It has 2,071 features and 969 targets. Known issues:
   - duplicate-looking targets under different parents (`platforms.kubernetes` vs
     `kubernetes.cluster`);
   - generic targets (`backup.data`);
   - an overfull Management category (729 features);
   - wrong aliases.

   The budget ran out before proposals for the smaller clusters.
7. **$ LLM adjudication in the mapper** (DESIGN.md §5.4 step 3). Nearest-neighbour mapping is
   15–60% precise at every cosine band measured, so until this exists only alias mappings are
   trustworthy.
8. **Capability table vendors in enumeration answers.** Add alias-only table vendors to the
   enumeration candidates in `answer()`, behind an eval. Measured on the 23 labelled questions:
   - retrieval alone: vendor recall 0.952, precision 0.877;
   - table ∪ retrieval: recall 0.984, precision 0.842.

   The per-vendor citation check should hold precision. The table recall is partly circular, so
   re-measure after item 4.
9. **Calibrate the remaining thresholds.** The clustering (0.12), mapper (0.80) and resolution
   (0.85) thresholds are all uncalibrated.

## Retrieval and answers

10. **Comparison facet recall is below target.** It is 0.833 with the rerank, against 0.9.
11. **Enumeration vendors that land in the closing line.** Of the vendors the labels expect,
    31 across 23 questions appear under "No supporting passage found for" instead of the list.
    Either retrieval found weak evidence, or the answer step dropped a line its citation check
    rejected. Diagnose which before changing anything.
12. **MCP server** (DESIGN.md M2/§6). Not built.

## Final-review minors (M3/M4 branch)

Left open at merge. I1, I2, M1, M8, M10 and M11 were fixed.

13. **Done 2026-11-08 (capabilities side): `capabilities.build` takes vendor and product from
    `documents`.** `raw_claims` still keeps the extraction-time values, and an offering mapped
    under the old vendor is not re-mapped. **M2: stale vendor/product on re-tag.** `store.update_metadata` re-tags a document without
    touching chunks, so `raw_claims` and `capabilities` keep the old vendor and product. Fix:
    take both from `documents` through `chunks` in `capabilities.build`.
14. **M3: `documents.version` is never written.** Every claim's `version` is NULL. Either
    populate it from the feed, or document it as reserved.
15. **Spend-safety gaps in `extract` and `taxonomy propose`:**
    - **M5:** a crash between `batches.create` and the ledger insert leaves a paid batch
      unledgered, so its chunks are paid for twice. Fix: insert the ledger row first.
    - **M6:** the sync-mode estimate uses Haiku prices for any model. An unexpected exception
      in `_call`/`_parse_openai` also skips recording the call's cost. Fix: refuse unknown-priced
      models without an override, and record errored outcomes.
    - **M7:** `taxonomy propose` caps each run, but has no cumulative ledger or lock.
16. **M9: `taxonomy map` memory and cost.** It re-embeds about 24k keys on every run and holds
    them as Python float lists, more than 2 GB. Fix: convert to float32 per batch, and cache key
    vectors by (text, model).
17. **M12: no test of `init_schema` on a populated database.** One more test gap remains. Also
    review whether every `init_schema` call needs the exclusive locks that
    `ADD COLUMN IF NOT EXISTS` takes.

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
    - **Serving:** vLLM's OpenAI-compatible API on Azure Container Apps (A100, scale to zero) for
      incremental extraction, through the existing `openai` extract backend. Batch the deltas
      to amortise cold starts.
    - **Student bar:** ≥90% agreement with Opus on a held-out set, and precision within a few
      points of it. Then run the full corpus on Vast, replacing item 4's API route.

## Taxonomy v2 follow-ups

19. **Done: taxonomy v2 reviewed by Claude** at the user's request. The decision log is
    `taxonomy/review-2026-11-06.tsv`, and the user may spot-check any row.
20. **$ Dimension adjudication precision is 0.84** on the reviewed taxonomy (the alias path is
    0.96). *2026-11-08: the evidence quote in each example claim raised it to 88.6% on the 150
    labelled non-alias claims (wrong 21 -> 14); still open: a stronger map_model, candidate
    pre-filtering, and the missed dimensions (20 of 27 "none" answers had a fitting dimension).*
    Try a stronger
    `map_model` on the 300 labelled claims, add the evidence quote to the prompt, or pre-filter
    candidates by family. The adjudicated labels from two graders are in the session scratchpad
    (`mapgrade2/`).
21. **Offering recall is 0.59.** Offering-name variants such as "Veeam Data Cloud for Azure" are
    not aliased. Add aliases, or adjudicate offerings like dimensions.
22. **Production rollout of v2.** The order matters, because the branch's `init_schema` drops
    the v1 tables:
    1. Take a database backup first. The upgrade is one-way: the old image's schema fails on a
       v2 database, so an image rollback breaks every sync run (see `docs/deploy/k3s.md`,
       "Upgrading to taxonomy v2").
    2. Check `pg_depend` for objects depending on the v1 tables.
    3. Merge and deploy.
    4. Load the reviewed taxonomy, then map and build capabilities.
    5. Re-measure findability.

    The teacher and precision claims are already in production under claims-v3 and claims-v3.1.
23. **Persist mapper adjudications** so re-runs and a lower `--max-usd` do not redo paid calls.
    Also keep a persistent list of members propose rejected.

