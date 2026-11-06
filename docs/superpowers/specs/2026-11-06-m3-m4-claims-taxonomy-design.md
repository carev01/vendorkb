# M3/M4: claim extraction, taxonomy v1 and the capability table

> Status: working spec, 2026-11-06 (overnight). It implements DESIGN.md §4, §5.3–§5.5 and the
> `query_capabilities` part of §6. User decisions (2026-11-06): go straight to M3/M4 after the
> retrieval follow-ups; extract with **Claude Haiku 5.5** on a **$25** budget; the pilot is a
> **stratified sample** over the pilot vendors plus the chunks behind the 23 labelled enumeration
> questions; taxonomy v1 goes up as a **draft GitHub PR** for the user's review. Choices marked
> *(my call)* are mine, made overnight.

## 1. Intent

vendorkb answers "which vendors do X" by retrieval. It finds 95% of the expected vendors, but in
6 of 8 answers the model left out a vendor whose evidence it had. DESIGN.md's answer is a
normalized capability table: claims extracted once per chunk, each with a verbatim evidence
quote, mapped to a reviewed taxonomy and queried with SQL. Every row links back to its chunk and
URL. Tonight builds the pipeline end to end on a pilot and measures it against the labelled
enumeration set.

## 2. Goals and anti-goals

**Goals**
- M3: `raw_claims` extracted by Haiku 5.5 from a stratified pilot of about 20,000 chunks, and an
  evidence gate (§4).
  - Report the gate pass rate, with a target of ≥ 90%.
  - Report claim precision on a 100-claim spot check by Opus subagents, with a target of ≥ 95%.
- M4:
  - cluster the claims' capability and target phrasings;
  - draft `taxonomy/taxonomy.yaml` v1;
  - load it, map claims to it, and build `capabilities`;
  - add `query_capabilities(feature, target?, vendor?, support?)` as a function and a CLI
    command;
  - measure vendor recall and precision on the 23 labelled enumeration questions against the
    retrieval baseline.

**Anti-goals**
- **A claim without a verbatim quote.** The gate rejects it into `rejected_claims`, and it never
  reaches `capabilities`.
- **A "not supported" claim the documentation doesn't state.** `not_supported` is only for
  explicit statements.
- **Overspending.** The extraction runner enforces a hard cap of $25. Every paid step starts with a
  2-item smoke test.
- **Merging the taxonomy without the user.** `taxonomy.yaml` lives on a draft PR. Code may merge;
  the taxonomy file waits for review.

## 3. Data model

Additions to `init_schema`, idempotent like the existing upgrades:
- `raw_claims` as in DESIGN.md §4, with two changes:
  - `vendor`, `product` and `version` are copied from the chunk's document, never extracted.
  - an `extraction_id` foreign key.
- `rejected_claims`: the same columns plus `reason`.
- `extractions(extraction_id, chunk_id, extractor_model, prompt_version, status, n_claims,
  input_tokens, output_tokens, created_at)` with a unique key on (chunk_id, extractor_model,
  prompt_version). This is the idempotency key from DESIGN.md: a re-run skips done chunks.
- `features`, `targets` and `aliases` as in DESIGN.md §4, loaded from YAML. `capabilities` as in
  DESIGN.md §4.

## 4. Extraction (M3)

- **Client.** The official Anthropic Python SDK (new dependency `anthropic`), model
  `claude-haiku-5-5`.
  - Output is constrained by `output_config.format` (`json_schema`). Effort is `low` *(my call:
    extraction is not reasoning-heavy, and low effort keeps output tokens down)*.
  - The key comes from env `ANTHROPIC_API_KEY`. Settings gain `extract_model`,
    `extract_budget_usd` and `extract_prompt_version`.
- **Input.** The chunk text, with its title and section path already prefixed by the chunker. The
  model also sees the document's vendor and product, so it can phrase claims about that product.
- **Output.** `{"claims": [{capability, target, support, conditions, evidence_quote}]}`.
  - `support` is one of supported / partial / not_supported / unclear.
  - `target` and `conditions` are nullable.
  - Return an empty list when the chunk has no capability content.
  - The prompt defines a capability as something the product can do (feature, workload or
    platform support, limit, security control). It asks for short, normalized phrasings and a
    verbatim `evidence_quote` of at most about 300 characters.
- **Evidence gate.** Normalize both strings: lowercase, collapse whitespace, unify quotes and
  dashes.
  - Accept if the quote is a substring of the chunk.
  - Otherwise accept if `rapidfuzz.fuzz.partial_ratio` ≥ 90 (new dependency `rapidfuzz`).
  - Otherwise reject with a reason.
  - Empty quotes are rejected.
- **Cost path.** The Message Batches API (50% off), polled until ended; results are matched by
  `custom_id` = chunk_id. A synchronous mode serves the smoke test.
  - The runner estimates cost from token counts at Haiku 5.5's prices ($0.10 in / $0.50 out per
    MTok; batch halves them).
  - It refuses to submit a batch whose estimated input cost, plus spend so far, would exceed the
    cap.
- **Pilot sample** *(my call on the numbers)*:
  - (a) 3,000 random chunks per pilot vendor (Commvault, Cohesity, Rubrik, Druva, AvePoint), a
    fixed seed, stratified by product, 15,000 in total;
  - (b) every chunk of every document cited in `expected_vendors` of the 23 enumeration
    questions;
  - (c) for each enumeration question and each of the 43 vendors, the top-5 hybrid candidates,
    so the table has a fair chance on every vendor the eval checks.
  - The union is about 20–25k chunks, an estimated $3–6 through batches.
- **Spot check.** Opus subagents judge 100 random accepted claims for support by their chunk,
  correct capability, and correct support value.

## 5. Taxonomy and capabilities (M4)

- **Clustering.**
  - Embed every distinct normalized `capability` string, and separately every distinct `target`
    string, with voyage-4 (input_type document; well under 1M tokens).
  - Cluster with agglomerative clustering (cosine, average linkage, distance threshold tuned so
    clusters read coherently; new dependency `scikit-learn`).
  - Clusters smaller than 2 claims across fewer than 1 vendor stay unclustered. *(My call:
    singletons go to a "long tail" list, not the taxonomy.)*
- **Proposal.** For each cluster, Haiku 5.5 (structured output) proposes `{id, name, category,
  definition, aliases[]}`.
  - The category is one of DESIGN.md's seeds: Workloads, Platforms, Storage targets, Data
    protection, Security, Recovery, Management.
  - Targets get hierarchical ids (`aws.rds`).
  - The proposals are written to `taxonomy/taxonomy.yaml` with `version: 1`. A short review guide
    goes in the PR description.
- **Loader.** `vendorkb taxonomy load` upserts `features`, `targets` and `aliases` from the YAML.
- **Mapper.**
  - Map in order: exact alias match (normalized), then the embedding nearest neighbour among
    feature names and aliases at cosine ≥ `map_threshold` *(calibrated on the spot-check
    sample)*.
  - Anything below the threshold stays unmapped.
  - LLM adjudication is deferred *(my call: cost and time; unmapped claims stay searchable)*.
- **Capabilities.** `vendorkb capabilities build` rebuilds the table from mapped `raw_claims` and
  the current taxonomy version. Conflicts between a vendor's claims for the same feature are kept
  as separate rows.
- **Query.** `query_capabilities(conn, feature, target=None, vendor=None, support=None)` returns
  rows of vendor, product, support, conditions, evidence quote, chunk_id and URL. A target query
  includes the target's descendants. CLI: `vendorkb capabilities query FEATURE [--target …]`.

## 6. Measurement (M4 "done when")

- For each of the 23 enumeration questions:
  - Resolve the question to feature (and target) ids with the mapper: alias, then nearest
    neighbour on the question's key phrase. Record the resolution.
  - Compute vendor recall and precision of `query_capabilities` against `expected_vendors`, with
    unclear vendors excluded.
- Compare with retrieval enumeration at FTS 0.25 and floor 0.54: recall 0.952, about 2.5 noise
  vendors per question.
- Also report the union of the two (table vendors ∪ retrieval-kept vendors), the obvious next
  integration for `answer()`.
- Caveat to state: part (c) of the pilot draws on retrieval candidates, so table recall on
  questions answered from those chunks partly reflects retrieval. Report recall split by whether
  the supporting doc came from (a)/(b) or only from (c).

## 7. Out of scope tonight

- The MCP server.
- 4B distillation (M5).
- Extraction beyond the pilot.
- Wiring `query_capabilities` into `answer()`. It is measured offline first.
