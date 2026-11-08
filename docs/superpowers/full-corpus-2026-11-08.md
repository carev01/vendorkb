# Full corpus, capabilities and MCP server (2026-11-07/08)

## Summary

- **Claims:**
  - Every one of the 227,615 chunks now has a claims-v3 extraction: 217,037 by the distilled Qwen3.5-9B, 10,524 by the Opus teacher set, and 54 errored (degenerate loops).
  - The 9B accepted 562,280 claims; the evidence gate rejected 3,951.
- **Taxonomy map:**
  - Every claim is mapped, by Claude Haiku 5.5 on the Anthropic API: 281,604 adjudications, $28.50.
  - The adjudication prompt now carries each example claim's evidence quote. Dimension precision on the labelled set went from 82.8% to 88.6%.
- **Capabilities:**
  - 458,785 rows (was 23,771): 43 vendors, 224 dimensions, 207 targets.
  - Every in-scope row of the internal comparison tables now has a fact: 423/423, up from 410.
- **MCP server:** deployed on the cluster (LAN) with seven read-only tools behind a bearer token. Smoke-tested live; agent eval below.
- **Freshness:**
  - `vendorkb refresh` (hourly CronJob, off until the endpoint exists) extracts and maps new chunks only when the sync brings some.
  - The extractor endpoint is packaged for Azure Container Apps (`deploy/azure`). Its deployment needs an Azure login.

## Corpus extraction

| | count |
|---|---|
| chunks | 227,615 |
| extracted by the 9B student (done) | 217,037 |
| extracted by the Opus teacher set | 10,524 |
| errored (cut off at 16k output tokens: degenerate loops) | 54 |
| chunks with no claims-v3 extraction | 0 |
| claims accepted (9B) / rejected by the gate | 562,280 / 3,951 |

The decode ran next to the GPU (Vast A100 SXM4 40 GB, about 6.3 chunks/s). It replayed into production through `extract.run`, so the lock, gate, records and ledger are the production path.

Two catches:
- **Duplicate content:** 9,921 chunks have the same text and metadata as another chunk. Their requests were identical, so the decode stored one reply per request, and the results named only one chunk of each pair. They were replayed by id.
- **Length cut-offs:** 122 chunks were cut off at 8,000 tokens and were re-decoded at 16,000. 72 of them completed; they were mostly long tables where the model writes one claim per row. The rest loop.

## Taxonomy map

### Choosing the adjudicator

There was no Anthropic API budget at first, so cheaper adjudicators were calibrated against Haiku's production decisions. The bar was set in advance: at least 85% agreement on Haiku-adjudicated dimensions, and the candidate at least as good in at least 50% of disagreements in a blind review. All rows below use the same 4,000-claim sample.

| adjudicator | agreement with Haiku | blind review (at least as good) |
|---|---|---|
| Qwen3.5-9B (local) | 65.7% | 43% |
| gpt-6-luna (Azure) | 61.6% | 42% |
| qwen3.8-27b (OpenRouter, bf16) | 77.5% | 50% |
| gemini-3.5-flash-lite | 81.0% | 47% |
| minimax-m3.1-flash | 80.0% | – |
| mimo-v2.6-flash | 78.6% | – |
| deepseek-v4.1-flash | 74.2% | – |
| Haiku 5.5 re-run (its own noise ceiling) | 93.1% | – |

None passed. Haiku costs $0.10 / $0.50 per Mtok on both Anthropic and OpenRouter. The user chose Haiku on the Anthropic API for the corpus.

### Evidence quote in the adjudication prompt

On the 150 labelled non-alias claims (picks not covered by the labels were graded blind):

| variant | precision | right overall | wrong | missed |
|---|---|---|---|---|
| prompt before | 82.8% | 72.0% | 21 | 21 |
| **+ evidence quote** (adopted) | **88.6%** | **77.3%** | **14** | 20 |
| + quote, softer "none" | 85.1% | 78.0% | 20 | 13 |

### The run

- **Mechanics:** `taxonomy map` now maps only claims not yet mapped under the current taxonomy version, so Haiku's earlier decisions were kept. It ran vendor by vendor (`--vendor`), one commit each.
- **Volume:** 281,604 adjudications in about 2 hours at concurrency 48, $28.50 in total. One failed call was redone by the last run.
- **Dimensions:** chosen by Haiku for 392,859 claims, by exact alias for 65,926, unmapped for 131,749.

## Capabilities

| | before (Opus claims) | after (full corpus) |
|---|---|---|
| rows | 23,771 | 458,785 |
| scope: product / product-wide / offering | – | 276,743 / 176,271 / 5,771 |
| comparison-table rows with a fact (of 423 in scope) | 410 | **423** |
| of those, a specific fact (not only product-wide) | 360 | **406** |

**Known issue:** 65% of rows fall in a conflict group. Most of them sit in a few very large, coarse keys, such as a product-wide dimension with thousands of claims. Even a positive-versus-not-supported-only rule still flags 59%. A finer conflict key is backlog work. The MCP output reports conflict groups with their size and lists at most 10 members.

## MCP server

`vendorkb mcp`: streamable HTTP at `/mcp` (stateless), bearer token required, Host allow-list, `/healthz`. It is deployed as `vendorkb-mcp` with a LAN ingress (`vendorkb-mcp.k3s.home.lan`, which still needs a DNS record).

| tool | what it does |
|---|---|
| `list_vendors` | vendors and products with documents, plus fact counts |
| `list_dimensions` / `list_targets` | browse the taxonomy by id, alias or substring |
| `query_capabilities` | the fact table: at most N per group, round-robin across vendors, each fact with quote and URL |
| `search_docs` / `read_section` | hybrid reranked search; a chunk in context |
| `answer_question` | vendorkb's cited answer in one call |

The full-corpus table OOM-killed the first deployment, because queries loaded every matching row before trimming. Queries are now bounded in SQL: 0.1–3.9 s and about 210 MB for the broadest filters.

### Agent eval (M4: does the capability table help an agent?)

Four held-out enumeration questions from `eval/questions.yaml` were each answered by a fresh
Sonnet agent through the deployed MCP server, with at most 10 `search_docs` calls each. The two
arms differ only in tools: search only (`list_vendors`, `search_docs`, `read_section`) versus all
tools. Answers are scored against the labelled `expected_vendors`, with unclear vendors excluded.

| question | search-only recall / precision | all tools recall / precision |
|---|---|---|
| Oracle RMAN integration (cv43-01) | 100% / 100% | 100% / 100% |
| Proxmox VE backup (cv43-03) | 100% / 100% | 100% / 100% |
| LTO tape library (cv43-04) | 87% / 93% | 93% / 93% |
| Nutanix AHV agentless (cv43-09) | 93% / 100% | 93% / 100% |
| **mean** | **95% / 98%** | **97% / 98%** |

With the capability table, agents matched or beat search alone on every question. They also
checked vendors through cheap table queries instead of reranked searches: the Proxmox agent made
36 capability queries and only 5 searches. Search alone is already strong on these questions, and
4 questions is too few to call the difference significant (backlog 28).

A first attempt at the all-tools arm OOM-killed the pod at 1 GiB. Its results were discarded, the
bounds above were added, and both arms were re-run on the same deployment. The pod had no restart
under 8 concurrent agents.

## Freshness

- **Trigger:** `vendorkb refresh` runs hourly and is two queries until the sync brings chunks that no extraction has touched.
- **Due run:** once the feed has been quiet for an hour, it wakes the extractor, extracts, maps (Haiku through OpenRouter, $1 cap per run) and rebuilds capabilities. Errored chunks never trigger a run.
- **Extractor:** the stock vLLM image on a serverless A100 that scales to zero, serving the base model plus the LoRA adapter (`deploy/azure`).
- **LoRA check:** LoRA serving was validated against the merged model: 49/59 identical replies, claim-set Jaccard 0.915.

## Spend (2026-11-07 evening to 2026-11-08 morning)

| | |
|---|---|
| Anthropic | $28.50 (corpus map) |
| OpenRouter | about $5.2 (adjudicator calibration and screening $5.11; experiments and smoke tests) |
| Vast | the A100 decode and checks; credit $12.49 to $8.73 |
| Voyage | about 7M tokens (calibration embeddings, map embeddings, agent-eval reranks) |

## What is left

- **DNS:** a LAN DNS record for `vendorkb-mcp.k3s.home.lan`.
- **Azure:** `az login`, then `ADAPTER_DIR=... bash deploy/azure/deploy.sh`. Put the key in the secret as `EXTRACT_API_KEY` and enable `claimsRefresh` (docs/deploy/k3s.md §7a).
- **Copilot:**
  - a public HTTPS route with OAuth in front of `/mcp`, then Copilot Studio onboarding;
  - DocExtractor's public route, traefik with a Cloudflare origin certificate, is a ready pattern.
- **Backlog:**
  - a finer conflict key;
  - offering recall (0.59);
  - the 20/27 missed dimensions;
  - persisting adjudications and caching key embeddings;
  - comparison facet recall (0.833);
  - enumeration closing-line vendors.

## Known-issues round (2026-11-08, later)

Fixed on production in the same day, and measured:

| | before | after |
|---|---|---|
| capability rows in a conflict group | 64.9% | **0.8%** (703 groups) |
| "product-wide" facts | 176,271 (65k of them an unmapped workload) | 113,852 (+ 41,941 `target_unmapped`) |
| claims with an unmapped target | 94,556 | **61,118** |
| offering mentions mapped | 21% | **39%** |
| dimension: alias / unmapped | 65,926 / 131,749 | 71,791 / 127,667 |
| comparison-table rows with a fact | 423 / 423 | 423 / 423 (405 specific) |
| a full re-map after a taxonomy change | ~$28 | **$0.22** |

### Changes

- **`init_schema`** stores a schema marker and runs no DDL when current; otherwise each lock
  waits at most 10 s. A long ad-hoc read had held the sync job's DDL for 24 minutes.
- **Capabilities:**
  - a claim naming a workload the taxonomy lacks gets scope `target_unmapped` instead of
    product-wide;
  - a conflict is now one statement (normalized phrase, target scope text, conditions) that is
    both supported/partial and not_supported, or that states different values.
- **Mapper:**
  - decisions persist in `map_decisions`, seeded from the corpus (268k decisions);
  - when a question's candidates change but offer nothing new, the earlier answer is carried;
  - alias lookups fold inflections;
  - a variant named as the target maps to its target;
  - the product itself named as the target means no target;
  - product names given as offerings are counted apart.
- **Taxonomy growth.** Opus curators proposed changes from the most frequent unmapped phrases,
  and a blind Fable review accepted 254 of 269:
  - targets: 79 new, 139 aliases, 26 variant aliases;
  - offerings: 27 new, 63 aliases;
  - dimensions: 29 aliases, plus 6 new for clear gaps.

  A blind check of 100 claims mapped through the new entries found 96 correct. The three
  wrong aliases it found were dropped.

### Measured and not adopted

- **A second pass over "none" answers** (a softer rule, 8 candidates, then a yes/no check) was
  tried on the labelled set. The verifier kept 4 right and 3 wrong picks, and rejected 4 right
  and 5 wrong. Adopting it would trade precision for coverage, so the "none" answers stay.
- **Re-asked questions flip.** Of 30,815 questions re-asked under changed candidates, 3,331
  flipped although the earlier answer was still offered: Haiku's run-to-run variance. That
  motivated the carry rule.

### Spend (this round)

| | |
|---|---|
| Anthropic | $4.48 |
| OpenRouter | about $4.2 |
| Voyage | about 4.5M tokens (seed and two re-maps) |
| Curation and review | Claude subagents |

## Taxonomy round 2 (2026-11-08, evening; BACKLOG 29 and 30)

The second curated round over the unmapped long tail, run like the first one (Opus curators, a
blind Fable review, a blind check of mapped claims). Taxonomy `2-31b1b705`.

| | before | after |
|---|---|---|
| claims with an unmapped target | 61,118 | **48,006** |
| claims with a target (alias or LLM) | 341,095 | 354,232 |
| targets with at least one fact | 286 | **461** |
| target-specific capability rows | 295,365 | 305,278 |
| `target_unmapped` capability rows | 41,941 | 32,069 |
| claims with no dimension | 127,667 | 127,276 |
| comparison-table rows with a fact | 423 / 423 (405 specific) | 423 / 423 (406 specific) |

### What was curated

- **Targets:** the 1,456 unmapped target strings named by two or more vendors or backing 15+
  claims (22.7k claims), round-1 strings excluded, in five alphabetical slices. Curators chose
  alias 331, variant alias 73, new target 191, generic 290, not a workload 571. Cross-slice
  duplicates were merged before the review (Centera, Lync/Skype for Business Server).
- **Dimensions:** the 1,354 unmapped capability phrases named by two or more vendors (7.4k
  claims): alias 195, none 1,064, gap 95.
- **Review:** 786 of 794 proposals accepted. Two rejections were aliases to ids merged away
  by the dedupe; they were re-pointed and accepted. Four class corrections from the reviewer
  were taken (Longhorn, Netgear ReadyDATA, Veritas Storage Foundation, Zadara).
- **Applied:** 177 new targets, 621 target aliases, 72 variant aliases. "microsoft windows",
  "windows file system(s)" and "ntfs" moved from `windows_server` to `microsoft_windows`.
  About 40 of the new targets are SaaS applications documented only by Securiti's
  data-subject-request connectors; they follow the braze/marketo/eloqua precedent.
- **Blind check:** 100 claims mapped through round-2 entries, one per phrase; 99 correct. The
  wrong one, "external usb drives" as a disk-storage variant (source data as often as a
  destination), was dropped.

### Dimension aliases held back

Every alias is a candidate-search key, so a new alias pulls its entity into the top-5
candidates of many other phrases, and each such question is asked again.

- A free count (`taxonomy map --max-usd 0`) found 108k questions to ask again, about $11. The 194
  dimension aliases drove ~90k of them while mapping 1,364 claims directly.
- **Pilot (Acronis, $0.22):** 79% of the re-asked dimension questions kept their answer. A
  blind grade of 60 changed answers (old versus new, order hidden) found 27 new better, 27 old
  better, 4 both acceptable and 2 both wrong: churn, not improvement.
- Keeping the 18 cheapest aliases still hurt. In the re-map, 246 re-asked dimension questions
  lost their earlier answer from the candidates (coverage.backup, given broad aliases such as
  "database backup and recovery", entered 109 of them), and one comparison-table row lost its
  only fact.
- All 194 are held in `taxonomy/held-dimension-aliases.json` until candidate search ignores
  alias-only changes (BACKLOG 31). Without them the table row is back.

### Re-asked targets that answered none

- When new targets entered a phrase's candidates, 551 re-asked target questions (5,975 claims)
  went from a pick to none. Only 639 (1,991 claims) went the other way.
- 464 of the 551 still offered the earlier answer, mostly obvious matches: "rhel" -> linux,
  "windows server 2003" -> windows_server, "vmware esx server" -> vmware_vsphere, "ibm iseries"
  -> ibm_i.
- **Fix (mapper):** a re-asked target answered none keeps an earlier answer still offered, and
  that is what gets stored. A none stored by an older run is read the same way. The re-map
  restored 499 answers. Dimensions are left alone: there the blind grade preferred the re-asked
  none as often as not.

### Dimension gaps (BACKLOG 30)

The 95 gap phrases fall into about 15 themes. None appears in the comparison tables, so none was
added:
- online or offline (hot or cold) database backup, ~140 claims;
- console password policy, ~100;
- database-native backup interfaces (RMAN, Backint, BR*Tools), ~60;
- database restore recovery state (NORECOVERY, STANDBY), ~50;
- UEFI and Secure Boot, ~45;
- off-host backup, ~40;
- direct-to-cloud backup, white-label branding, cross-version restore and interface languages,
  25-30 each;
- product lifecycle dates, backup source in replicated databases and certificate lifecycle,
  12-16 each.

### Measured and not adopted

- **Version folding:** stripping trailing versions before the alias lookup ("ubuntu 24.04" ->
  "ubuntu") would resolve 792 claims (1.3% of the tail). Not built. It surfaced the "microsoft
  windows" alias, now moved.
- **Paying for all re-asks:** about $11, see above.

### Spend (round 2)

| | |
|---|---|
| OpenRouter (Haiku 5.5) | $2.36 (pilot $0.22, re-map $2.12, two top-ups $0.01) |
| Anthropic | $0 |
| Voyage | 91 new strings (the embedding cache served the rest) |
| Curation, review and grading | Claude subagents (9 curators, 4 reviewers, 2 graders) |
