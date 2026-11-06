# Backlog

Open work as of 2026-11-06, after the cross-vendor follow-ups and the M3/M4 pilot. Each item says
why it matters and what "done" looks like. Sources:
`docs/superpowers/cross-vendor-retrieval-2026-11-06.md` and
`docs/superpowers/m3-m4-pilot-2026-11-06.md`.

Items marked **$** spend money (OpenRouter, Anthropic or Voyage) and need a go-ahead first.

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
4. **$ Extraction beyond the pilot.** About 227k chunks remain: about $100 on OpenRouter Haiku at
   full price, or about half with the Anthropic Batches API. Do it after item 1, so the corpus is
   extracted once with the good prompt. Item 5 or item 18 may replace the API route.
5. **A valid Anthropic API key.** The `sk-ant-usr` token in `.env` returns 401, so the pilot
   ran through OpenRouter. The Batches path (`--mode batch --backend anthropic`) is tested only
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

13. **M2: stale vendor/product on re-tag.** `store.update_metadata` re-tags a document without
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

## Local extractor (DESIGN.md M5)

18. **Fine-tune a local extractor (Qwen3.5) on the Haiku claims.** The goal is to make full
    re-extractions free, so the prompt and schema can keep changing. See the assessment in the
    2026-11-06 conversation, summarized here:
    - Do it after item 1. Distilling the current 86%-precise output would teach the errors too.
    - **Zero-cost first step:** run the base model with the v2 prompt on the spot-check chunks,
      through the gate and Opus grading, and measure tokens per second on the P6000. That gives
      the real full-corpus duration before any training.
    - Train on a rented GPU (Vast/Colab), not the P6000: it is Pascal, so it has no bf16 and no
      Unsloth support. Serve the result through llama.cpp.
    - Adopt it only if it reaches ≥90% agreement with the teacher on a held-out set, and its
      precision stays within a few points of the teacher's.
