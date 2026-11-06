# Taxonomy v2 and claims v3: results (2026-11-06)

Spec: `specs/2026-11-06-taxonomy-v2-claims-v3-design.md` (revision 2). Plan:
`plans/2026-11-06-taxonomy-v2-claims-v3.md`. All runs below were on the production corpus, but
taxonomy v2, mapping and capabilities ran on a **local copy**. The branch's schema drops the v1
tables, so it must not touch production before merge and deploy. The copy held the v3 claims,
their chunks and every document.

## Claims v3 (extraction)

- **Fields:** 14 per claim, including kind (capability, licensing or service), target_scope,
  offering, edition, availability, default_state, and value/unit/bound. No `requires`. Prices
  are never claims.
- **Prompt:** `claims-v3.1`. Each revision bumps the minor version. The system prompt is cached
  (Opus 5.5 caches prompts of 512 tokens or more).
- **Smoke:** Opus 5.5 at effort low on 31 chunks: 20 earlier spot-check chunks plus 11 chosen
  for qualifiers. All 14 earlier v1 errors were fixed. Three prompt defects were found and fixed
  before the precision check:
  - `offering` was used for features and components;
  - `target_scope` held operations;
  - install sizing requirements were extracted as claims.
- **Precision gate (≥95%, all fields correct, graded by Fable subagents):**

  | Round | Prompt | Chunks | Claims | Precision | Cost |
  |---|---|---|---|---|---|
  | 1 | claims-v3 | 300 | 735 | 94/100 | $7.13 |
  | 2 | claims-v3.1 | 300 fresh | 642 | **96/100** | $3.20 (with caching) |

  - Round 1's misses: 2 `default_state` values inferred from "must enable"; 1 changelog line;
    1 standard used as a target; 1 "partial" for a platform limit inherent to the target; 1
    configurable element put in conditions. Revision 1 fixed these.
  - Round 2's misses: 2 targets the chunk never named; 1 condition; 1 quote that did not state
    the fact.
  - Evidence gate pass rate: 99.9%.
- **Teacher set (Opus, for distilling Qwen3.5-9B):**
  - 10,000 chunks: 9,500 train and 500 held out, stratified. 5,000 come from the workloads the
    comparison tables cover, and the rest are spread across all 43 vendors.
  - Runs: Batches API in waves sized to the budget. 35 chunks were cut off at 3,000 tokens and
    re-run at 8,000.
  - 27,126 claims: about 25.4k capability, 0.7k licensing, 0.2k service. About 2,000 chunks
    gave no claims, which the student needs as negatives.
  - **Spend: $65.09** against a $100 cap, about $0.0065 per chunk.
- **Bugs found and fixed during runs:**
  - A NUL byte in a model reply crashed result storage. NULs are now stripped.
  - The runner kept advancing after a failed wave. It now aborts.

## Taxonomy v2

- **Seed** (written top-down by Claude, in our own words): 159 dimensions in 12 families, 34
  classes with object-type templates, 162 targets, 23 offerings.
- **Coverage of the comparison tables** (local map, never committed): 447 rows, 24 out of scope
  (pricing, competitor differentiators, catch-all rows), **423/423 in-scope rows map to ids**,
  with 0 unknown ids.
- **Alias proposals** from 28.5k v3 claims (Haiku, clustering threshold 0.12, $0.88 in total):

  | Kind | Aliased into existing entries | New entries proposed | None |
  |---|---|---|---|
  | dimension | 346 | 573 | 917 |
  | target | 68 | 115 | 211 |
  | object type | 116 | 169 | 204 |
  | offering | 5 | 50 | 167 |

  - A live run showed whole-cluster aliasing let wrong members in, so the model now picks which
    members of a cluster are aliases.
  - The result has 732 dimensions, 277 targets, 305 object types, 70 offerings and 4,782 aliases.
  - New entries carry `proposed: true` for review. They include real gaps (record-level restore,
    soft delete of backups), duplicates of seed entries (central management versus multi-tenant
    management, license model versus license metric), and over-specific limits.
- **Extrapolation:** `aspects(okta)` equals `aspects(entra_id)`: 199 aspects on the seed, 737 on
  the full taxonomy.

## Mapping and capabilities (local copy)

- **Mapper:** exact alias first, otherwise Haiku adjudication over 5 nearest candidates. 17,966
  adjudications, **$1.84**.
  - Dimension: 23,177 of 28,503 claims mapped (8,040 by alias, 15,137 by adjudication).
  - Target: 19,105 of 21,750 claims that name a target.
- **Bug:** the adjudication schema enumerated candidate ids, so each request compiled a new
  grammar and hit Anthropic's limit of 50 compilations per minute. It now uses one fixed schema
  (an integer choice).
- **Mapper precision** (300 random claims, graded by Opus subagents; labels in the scratchpad
  for the user's check):

  | Field | Precision of assigned ids | Recall |
  |---|---|---|
  | dimension | 0.852 (alias 0.975, LLM 0.788) | 0.727 |
  | target | 0.985 | 0.910 |
  | offering | 1.000 | 0.355 |

- **capabilities:** 23,177 rows over 43 vendors (16,048 product, 5,789 product-wide, 1,340
  offering). 943 conflict groups. 12,394 facts sit on seed dimensions and 10,783 on proposed
  ones; 477 of the 573 proposed dimensions hold 5 or more facts.

## Findability against the comparison tables

| Check | Result |
|---|---|
| Taxonomy coverage | 423/423 in-scope rows |
| Claim findability (capabilities, for the table's vendor) | 404/423: 355 with specific facts, 49 product-wide only |
| Chunk findability for the other 19 rows (hybrid search with rerank on production, top 10, relevance by an Opus subagent with anonymised rows) | 11/19 |
| **Total** | **415/423 (98.1%)** |

- The 8 rows left are all commercial or vendor-level aspects: hardware sourcing, certifications,
  exit strategy, durability, egress terms.
- **Caveat:** claims exist only for the roughly 10.6k chunks extracted (teacher set plus
  precision sets), not the full corpus. Claim findability will rise with full extraction.

## Spend (this work)

| Item | Cost |
|---|---|
| Opus smoke (direct) | $1.17 |
| Precision rounds (Opus) | $10.33 |
| Teacher set (Opus, Batches) | $65.09 |
| Taxonomy proposals (Haiku) | $0.89 |
| Mapping (Haiku, including the failed run) | about $1.92 |
| **Anthropic total** | **about $79.40** |
| Voyage | under 1M tokens: string embeddings, 19 queries, 19 reranks |

## Next

1. **Review the taxonomy** (PR): prune and merge the 857 proposed entries, and fix the
   duplicates.
2. **Dimension adjudication at 0.79** is the weakest link. Try a stronger map model on a sample,
   add the evidence quote to the prompt, or pre-filter candidates by family.
3. **Offering recall at 0.36:** add aliases for offering-name variants (for example "Veeam Data
   Cloud for Azure"), or adjudicate offerings like dimensions.
4. **Fine-tune Qwen3.5-9B** on the teacher set (backlog item 18), then run the full corpus on Vast.
5. **Production:** merge, then deploy (the first `init_schema` drops the v1 tables), then load the
   reviewed taxonomy, map and build capabilities.
