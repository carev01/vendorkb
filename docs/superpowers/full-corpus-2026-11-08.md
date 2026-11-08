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
