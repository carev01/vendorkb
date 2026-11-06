# Cross-vendor questions over the 43-vendor corpus

> Status: draft for review, 2026-11-06. Follows the production retrieval check of 2026-11-05.
> To be done before the MCP server.

## 1. Intent

vendorkb now holds the whole DocExtractor corpus: 120,637 documents from 43 vendors. Before,
it held 20,899 documents from 4. The production retrieval check showed that single-vendor
questions still work: the gold article was in the rerank pool for 0.94 of them, against
0.96 before. Cross-vendor questions fell from 0.93 to 0.53. The causes, and the decisions
that follow from each:

- **One vendor crowds out the rest.** The 40-candidate rerank pool comes from one hybrid
  search. For "which products support pre- and post-backup scripts" all 40 candidates were
  Salesforce chunks.
- **Named comparisons are lopsided.** "AWS Backup vs Azure Backup on SAP HANA" is scoped
  correctly to both products (scope detection works; an earlier report that it missed them
  was wrong). But Azure Backup has 3x the documents and took 35 of the 40 slots.
- **The eval's gold is incomplete.** Cross-vendor gold was written for the 4 original
  vendors. Commvault, IBM and others now legitimately answer "which vendors can protect AIX"
  and push the 4-vendor gold articles out. Restricted to the 4 vendors, most lost questions
  find their gold again (cv-03 rank 1, cv-09 2, cv-08 4).

**What the user wants from a "which vendors" question (their words):** the answer should
list *all* the known vendors that support it. Both kinds of error are bad, but naming a
vendor that does not support it is worse than omitting one that does. And the answer
should also say which vendors it checked and found nothing documented for.

## 2. Goals, non-goals, anti-goals

**Goals**
1. An eval that can score cross-vendor questions on the 43-vendor corpus. It comes first,
   so the retrieval changes are measured against it.
2. Enumeration questions: retrieval considers every vendor, and the answer lists each vendor
   whose documents support the claim, with citations. It ends with a deterministic line
   naming the vendors checked with nothing documented.
3. Comparison questions: each named vendor or product gets a fair share of the pool.
4. No regression on single-vendor and multi-article questions.

**Non-goals**
- The MCP server.
- Claim extraction and the capability table (M3/M4). A capability table would answer
  enumeration directly. This design gets as far as retrieval can.
- New scope aliases beyond what the eval shows is missing (see §5).

**Anti-goals**
- **A vendor named without evidence.** Every vendor in an enumeration answer must be backed
  by a cited chunk *from that vendor*. This is checked deterministically after the model
  answers, not left to the prompt.
- **"Not documented" stated as "not supported".** The closing line says what the corpus
  lacks, never what a product cannot do.
- **Paid runs without a go-ahead.** Retrieval checks are free (one query embedding per
  question). Rerank and answer runs are paid, and each one needs the user's OK.
- **Grading the eval with the system under test.** Labelling candidates must come from a
  wider search than the one being evaluated (§3.2).

## 3. The eval (first)

### 3.1 Question shapes

The 20 cross-vendor questions split into two shapes:
- **Enumeration (11):** ex-rt-10, ex-rt-12, cv-01 to cv-09. They ask which vendors or
  products do something, and name none.
- **Comparison (9):** ex-rt-07 to ex-rt-09, cv-10 to cv-15. They name two vendors or
  products.

Add about 10 new enumeration questions. They are written from the 39 new vendors' documents,
so the set does not lean on the original four. The rules follow `eval/README.md`. *(My call:
11 enumeration questions are too few to measure vendor recall.)*

### 3.2 Labelling expected vendors

Each enumeration question gets `expected_vendors`: the vendors whose documents state that
they support the capability, each with one or more supporting article ids. The labelling
works like this:

1. **Candidates.** For every one of the 43 vendors, run the question through a vendor-scoped
   hybrid search with a wide pool: the top 10 chunks per vendor. Add a keyword search with
   hand-written key terms for the capability ("AIX", "pre-job script"). This is wider than
   anything the system retrieves, so the label does not inherit its blind spots.
2. **Judgement.** Opus subagents read each vendor's candidates and answer: does this vendor's
   documentation state the capability? (yes / no / unclear), citing the chunk. "Unclear"
   counts as not expected. *(My call: Claude subagents rather than a paid judge, per earlier
   practice. Free to the Voyage budget apart from query embeddings, about 43 × 21 per
   question.)*
3. **Spot check.** A second subagent re-judges a 10% sample. Disagreements are resolved, and
   the agreement rate is reported.

Comparison questions keep their gold sources. Their score is computed per named facet (gold
article in the pool for each facet).

### 3.3 Metrics

The free retrieval check is a new mode of `eval/retrieval_compare.py`, or a sibling script
(builder's call). It reports:
- **Enumeration vendor recall:** the share of `expected_vendors` that have at least one chunk
  in the pool handed to the answer step.
- **Enumeration vendor noise:** the pool's vendors with no expected support. This is
  informational: noise in the pool is fine if the answer step filters it.
- **Comparison facet recall:** gold in the pool for each named facet.
- **The existing metrics for all questions,** so regressions show.

The paid answer eval (later, with an OK) adds:
- **Answer vendor precision:** named vendors that are in `expected_vendors`.
- **Answer vendor recall.**
- **Closing-line accuracy:** no expected vendor listed as "nothing documented".

## 4. Retrieval and answer changes

### 4.1 Shape detection

A question is:
- **comparison** when scope detection finds two or more distinct facets (vendors or
  products);
- **enumeration** when the existing cross-vendor wording matches (`_CROSS_VENDOR`) and no
  facet is named, or the wording is "which of these vendors/products" and the like;
- **single** otherwise.

This is deterministic: no model call. The patterns are extended from the eval's enumeration
questions (cv-03 "Which of these vendors document…", cv-04 "Which products let us…").

### 4.2 Enumeration retrieval

- For each vendor in the corpus (43, no ceiling), run the hybrid search scoped to that
  vendor. Keep its top `per_vendor` chunks (default 3).
- Rerank all of them in one Voyage rerank call: 43 × 3 ≈ 130 documents, about 60K rerank
  tokens per question.
- A vendor survives when its best rerank score is at least `enumeration_min_score`. The
  floor is calibrated on the eval for vendor recall at near-zero false vendors.
- If the rerank is unusable (an error), keep every vendor and flag it, rather than dropping
  any silently.
- Survivors go to the answer step, ordered by best score, at most `per_vendor` chunks each.

The searches run concurrently on one connection pool, or serially if latency is acceptable;
measure it.

### 4.3 Comparison retrieval

Search each facet separately, with an equal share of the 40-candidate pool. Then merge and
rerank. A facet with no hits is reported, not silently relaxed.

### 4.4 Keyword-search weighting

The vector-only ranking beats the hybrid merge on recall@5 (0.88 against 0.62 in the A/B),
because the OR-everything full-text query favours long, term-dense chunks. Try down-weighting
the full-text ranking in RRF, or capping its contribution, and measure both on the eval.
Adopt only if all-question pool recall does not drop. *(My call: an experiment, not a
committed change.)*

### 4.5 The enumeration answer

- A prompt variant asks for one line per vendor, in the form `Vendor: how it does it [n]`.
  It also asks for no line for a vendor whose sources do not state the capability.
- **Deterministic check after the reply.** A vendor line is kept only when at least one of
  its markers resolves to a chunk whose `vendor` is that vendor. Other lines are dropped,
  and the drop is counted in the result.
- **Closing line, built in code, not by the model:** "Checked with nothing documented: …".
  It names the vendors that were searched and not kept, both those below the floor and
  those the model did not list. The wording is "nothing documented", never "not
  supported".
- If every line is dropped, refuse as today.

## 5. Scope detection

Detection is not the cause of the failures seen (§1). Aliases are added only where the new
eval questions show a miss. *(My call: no speculative alias list.)*

## 6. Cost

- **Retrieval checks:** free apart from query embeddings. About 60K Voyage tokens per full
  run with enumeration fan-out; about 95M voyage-4 tokens remain.
- **Labelling:** query embeddings only, about 1M tokens at most, plus subagent time.
- **Rerank in production:** about 60K rerank tokens per enumeration question, from the
  separate rerank allowance.
- **Paid answer eval:** asked for separately.

## 7. Testing

- **Unit:** shape detection on every eval question; the enumeration answer's line check and
  closing line (lines with wrong-vendor markers dropped; vendors listed as unchecked).
- **Integration (testcontainers):** per-vendor fan-out returns at most `per_vendor` per
  vendor and covers every vendor; per-facet comparison shares; the rerank-failure path keeps
  every vendor.

## 8. Order

1. Eval: shapes, new questions, labelling, metrics; baseline on production.
2. Enumeration and comparison retrieval; measure.
3. Keyword-weighting experiment; measure.
4. The enumeration answer and closing line. The paid answer eval waits for a go-ahead.

## 9. Builder-check answers (my calls, after approval)

1. **What the answer sees.** Guarantees and metrics apply to the hits handed to the answer
   step, not to the pre-rerank pool.
   - Comparison: per-facet quotas survive the rerank. The answer gets the top 5 hits per
     facet, so 10 for two facets.
   - Enumeration: the answer gets the surviving vendors' chunks, at most `per_vendor` each,
     capped at 60 chunks in total.
2. **Who owns precision; targets.** The answer-step check owns precision (§4.5). The
   rerank floor is tuned for vendor recall: it drops only vendors that are clearly off-topic.
   - Targets: enumeration vendor recall at the answer-step hits of 0.8 or more; comparison
     facet recall of 0.9 or more; single-vendor and multi-article pool recall within 0.02
     of the production baseline.
   - The floor is calibrated on the 11 existing enumeration questions. The ~10 new ones are
     a holdout, reported separately.
3. **Shape before scope.**
   - Shape detection runs its own facet detection, ignoring `_CROSS_VENDOR`.
   - Two or more facets make a comparison. Cross-vendor wording with zero facets makes an
     enumeration. Cross-vendor wording with exactly one facet ("other vendors besides Veeam")
     also makes an enumeration over every vendor, including the named one. Anything else is
     single.
   - `detect_scope` is split so the facet list is available either way.
4. **The unit is the vendor, and lines name products.** Each answer line is
   `Vendor (Product, …): how [n]`.
   - Fan-out: up to 3 chunks per vendor, but a vendor with several products gets its best
     chunk per product, up to 5.
   - `expected_vendors` stays vendor-level, with the supporting product noted.
5. **Latency and cost.** Target: under 10 s per enumeration question, before the answer
   model.
   - Concurrency: a small connection pool (4).
   - Rerank tokens are treated as paid. They are measured per question and reported. If they
     exceed 100K per question, shrink `per_vendor`.
6. **What "supports" means for labelling.** The vendor's own documentation states that its
   own product can do it; a same-vendor add-on counts.
   - These do not count: a third party doing it through an integration, and a mention only
     in a compatibility list without saying it is supported.
   - "Unclear" vendors are excluded from scoring. They count neither as expected nor as a
     false vendor.
7. **A comparison facet with no hits** goes in the result envelope (`facets_empty`) and in
   the deterministic closing line ("nothing documented for: X"). The comparison prompt is
   otherwise unchanged.
8. **Vendor matching.** The prompt lists the exact vendor names. A line must start with one
   of them, otherwise it is dropped. No fuzzy matching.
