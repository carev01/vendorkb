# Topic expansion: facts survive page updates, and ransomware protection (2026-11-09)

Spec: `specs/2026-11-08-topic-expansion-design.md`. Plan: `plans/2026-11-08-topic-expansion.md`.
The work shipped as two PRs:
- PR #32, Part A: helm rev 24, image `sha-911da42`.
- PR #33, Part B: helm rev 27, image `sha-3392618`.

## Part A: facts survive page updates

- **Superseded chunks.** A re-synced page keeps the chunks that carry mapped claims, marked superseded (`chunks.superseded_at`), until its new version is extracted and mapped. `capabilities build` then retires them.
  - Facts on superseded chunks stay in every query, marked "(from an earlier version of the page)".
  - The mark is computed when the facts are read.
  - Removing a page still deletes everything.
- **Chunk reuse.** A chunk whose text matches the old version keeps its extraction, claims and facts. The comparison ignores whitespace, link and image targets, escapes, and narrow "Last Updated" lines (`chunker.reuse_key`). Every evidence quote must still appear in the new text.
  - Measured on 667 updated pages from DocExtractor's version history: **46% of updated chunk tokens and 44% of updated pages need no re-extraction.** Exact text alone would cover 20%.
- **No model check of changed chunks.** In a labelled sample of 80 changed chunks:

  | outcome | share |
  |---|---|
  | no fact changed | 46% |
  | facts added | 30% |
  | facts changed | 8% |
  | unclear | 16% |

  45% needed careful reading. A check that also catches additions costs about as much as extraction does.
- **Deploy notes.**
  - The schema went first, from a one-off Job running the new image; `init_schema` took 3.4 s.
  - A new `chunks_doc_idx` keeps the documents→chunks cascade indexed. The review measured 200 deletes taking 30 s without it.

## Part B: the ransomware protection topic

What changed:
- **Topic data.** `taxonomy/taxonomy.yaml` gains `topics:`, with eight aspects: built-in security, immutability, anomaly detection, malware detection, clean recovery, air gap, deletion safeguards and encryption. The topic is loaded into the `topics` / `topic_aspects` tables and is not part of the version hash.
- **When a question gets the topic.** A question takes the topic path when:
  - a trigger matches;
  - no aspect cue matches;
  - no product name is involved;
  - it is not narrowed to a workload;
  - it has at most 4 sides.
- **Retrieval.** One full-text statement serves the question's own passages. Aspect candidates are vector-only, plus capability-fact chunks. The rerank cap is 240 documents, with at most 4 reranks in flight.
- **Answer.** The answer is organised by aspect: "the retrieved sources do not cover this", never "lacks".
- **MCP.** `query_capabilities(topic=...)` groups facts by aspect. `list_dimensions` shows the topic when asked about it.

### Blind before/after eval (`eval/topic_questions.yaml`, tp-01..tp-10)

- **Runs.** The "before" run used main `911da42` (production code). The "after" run used branch code against production data.
- **Grading.** Opus and Fable graded blind; their agreement was 98.3%. An Opus adjudicator settled 5 disagreements.
- **Corpus check.** `corpus.json`, built from the capability table, decides which aspects a side documents. Every side documents all 8 except Veeam Backup & Replication, which documents 7 (no air-gap facts).

| | before | after |
|---|---|---|
| aspect coverage (adjudicated) | 67.0% | **98.9%** (+31.8 pts; bar +10) |
| unsupported-claim rate, pooled | 2.39% | **0.72%** (bar: at most +2 pts) |
| unsupported-claim rate, prose answers | 2.42% | 0.91% |

- **Per question after:** every question reached 100% coverage except tp-04 (7/8).
- **Motivating case:** tp-05 is ex-rt-08 (Veeam B&R vs Azure Backup). Its coverage went from 44% to 100%.
- **Verdict: PASS.**
- **Cost and latency of a broad topic answer:**
  - rerank tokens: 109–148k per question, against 12–23k before;
  - latency: 27–80 s, against 8–23 s;
  - answer model cost: about $0.002 per question.

### The grid ("which vendors"): built, measured, off by default

The code-built vendor × aspect grid works: it is cited, dated, uses a catch-up search, and degrades when a dependency fails. But its precision is capped by the capability facts it shows. Blind-graded share of sampled cells supported by their cited text:

| grid version | Opus | Fable |
|---|---|---|
| best fact per cell | 71.7% | 68.3% |
| cue-gated, top 5 candidates | 95.0% | 91.7% (but big vendors' cells went blank) |
| cue gate in SQL, all candidates | 86.7% | 86.7% |

- **Why the first version failed.** It failed on facts mapped to the wrong dimension. Examples: "Inactive files" counted as anomaly detection, "Dedicated backup networks" as air gap, and a page about an AI agent's session details as deletion safeguards.
- **Why gating does not fix it.** Gating on the claim's own text tops out at 92.5% on 180 graded cells: claims that name the aspect still lack support in their page.
- **Effect on the pass bar.** Today's prose enumeration scores 98–100%, so a ~200-cell grid at about 90% fails the bar.
- **Decision.** `TOPIC_GRID` is off by default, and "which vendors" questions keep today's enumeration path. Enabling the grid is a product decision; BACKLOG 33 describes the path to it.

### Agent eval gate (after the MCP change)

23 Sonnet agents ran against the deployed MCP (`sha-3392618`). They scored recall **97%** and precision **96%**, against a baseline of 96.0% and 94.9%: **PASS**.

## Lessons

- **Helm 4 and `kubectl patch`.** Helm 4 uses server-side apply. A `kubectl patch` on a field helm manages makes every later helm change to that field conflict (revs 23 and 25 failed). Suspend sync through `helm upgrade --set sync.suspend=true`. Clear a stale owner with `--force-conflicts`, as rev 26 did.
- **Full-text search over 227k chunks.** An OR'd full-text query costs about 2 s per text. Batching nine texts into one statement re-parses each tsquery per row unless the tsqueries are materialised, which took 69 s. Even materialised, nine texts took 10–14 s, so aspect retrieval became vector-only.
- **Ties in the grid's fact order.** Every page has the same fetch date, so "newest page first" ties and falls to chunk id. A small top-N per cell is then effectively random.
