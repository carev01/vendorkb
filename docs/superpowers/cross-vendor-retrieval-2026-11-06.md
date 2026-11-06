# Cross-vendor retrieval: results (2026-11-06)

Implements `specs/2026-11-06-cross-vendor-retrieval-design.md` (plan
`plans/2026-11-06-cross-vendor-retrieval.md`). Everything below was measured on production
(120,637 documents, 43 vendors), through a `kubectl port-forward`.

## What changed

- **Question shape** (`shape.py`). Every question is classified as single, comparison or
  enumeration, deterministically. On the production scope index, all 131 eval questions get
  their labelled shape.
- **Enumeration retrieval** (`modes.enumerate_vendors`).
  - All 43 vendors are fetched in one batched query set, about 5 statements whatever the
    vendor count. Each vendor contributes 2–3 chunks.
  - Everything goes through one rerank. Vendors below the floor of 0.46 are dropped.
- **Comparison retrieval** (`modes.compare_facets`). Each named facet keeps a quota of 5
  chunks after the rerank. There is one full-text query for all facets, and the per-facet
  reranks run concurrently.
- **Enumeration answers** (`answer.py`).
  - The answer is one line per vendor, checked in code. A line is kept only if it cites a
    chunk from its own vendor. Product groups are checked against that vendor's hits.
  - A line that cites another, non-workload vendor and names it is dropped.
  - The closing line is deterministic: "Checked with nothing documented: …".
    (Wording changed after this run to "No supporting passage found for: …".)
- **Eval.** 23 enumeration questions are labelled for all 43 vendors: 11 existing and 12 new
  held-out ones. Opus subagents did the labelling, and a 10% independent re-label agreed on
  87% of pairs. The 13 disagreements are marked unclear and excluded from scoring. The result
  is 419 expected and 77 unclear (question, vendor) pairs.

## Numbers against the spec's targets

| Target | Measured | Status |
|---|---|---|
| Enumeration vendor recall ≥ 0.8 at the answer step's hits | 0.949 calibration / 0.954 holdout at floor 0.46 (paid rerank run) | met |
| Enumeration noise | 4.8 / 5.0 non-expected vendors per question, down from 17.5 / 25 without a floor | — |
| Comparison facet recall ≥ 0.9 | not measured with the rerank (cap reached, see below). Free mode gives 0.667 on 10 hits, against legacy 0.778 on a 40-chunk pool, which is not comparable. At the answer level, 9 of 9 comparison answers addressed every side | **open** |
| Single-vendor pool recall within 0.02 of baseline | 0.950, legacy 0.950 | met |
| Multi-article pool recall within 0.02 | 1.000, legacy 1.000 | met |
| Enumeration latency < 10 s before the answer model | rerank run p50 11.8 s / p90 16.0 s (calibration), 10.0 / 13.1 s (holdout); free mode p50 10.4 / 8.6 s | **missed narrowly** |
| Rerank tokens < 100K per question | about 80K at 2/3 chunks per vendor (about 115K at the spec's 3/5) | met after the §9.5 shrink |
| Comparison latency | free mode p50 0.6 s / p90 2.7 s, against 17 / 36 s before | improved |

Legacy enumeration (today's single search) found 0.409 / 0.573 of the expected vendors.

## Answer quality (paid subset, graded by Opus subagents)

17 questions: 8 enumeration and all 9 comparisons. Cost: $0.059 on OpenRouter. The answers
were produced at commit `6ca0ec3`, before the final review's fix wave (I-6 drop, max_tokens
8000, comparison batching).

| Shape | n | Fact coverage | Unsupported claims | Vendor precision | Vendor recall | Other |
|---|---|---|---|---|---|---|
| enumeration | 7 | 0.944 | 3 | 0.956 | 0.902 | 10 expected vendors wrongly in the closing line |
| comparison | 9 | 0.732 | 3 | — | — | every side addressed in 9 of 9 |
| cv-09 | 1 | 0 | 0 | — | 0 | answered by the single path, a shape bug ("Own" matched "own") since fixed; not re-run because of the cap |

The main remaining weakness: in 6 of the 8 enumeration answers, an expected vendor was missing
from the list and so appeared in "Checked with nothing documented". The cause is the model
skipping a vendor whose chunk was in the prompt, or the vendor's best chunk falling below the
floor. The line reads as a claim about the corpus.

## Keyword weighting (free)

| `FTS_WEIGHT` | single pool / recall@10 | multi pool / recall@10 |
|---|---|---|
| 1.0 (shipped) | 0.950 / 0.833 | 1.000 / 0.737 |
| 0.5 | 0.967 / 0.883 | 1.000 / 0.842 |
| 0.25 | 0.967 / 0.933 | 1.000 / 0.895 |

0.25 is clearly better for single and multi-article questions. It was **not** adopted, because
the weight also changes the enumeration candidates, and the floor was calibrated at 1.0.
Adopting it needs one paid enumeration re-calibration (about 1.85M rerank tokens).

## Cost

| Item | Tokens or money |
|---|---|
| Voyage rerank, smoke | ~0.23M |
| Voyage rerank, calibration | 1.85M |
| Voyage rerank, answer subset | 0.88M |
| Voyage rerank, total | ~2.96M (cap 3M) |
| Voyage embeddings | query embeddings only (a few thousand tokens) |
| OpenRouter | $0.06 |

## Open items

1. Comparison facet recall with the rerank, and the full answer-quality run. Both need rerank
   budget.
2. `FTS_WEIGHT` 0.25 plus an enumeration re-calibration.
3. Enumeration latency: apply the prepared Postgres memory settings (chart values, not yet
   applied to production) and re-measure.
4. Closing-line wording: "nothing documented" is wrong when the model skipped a vendor whose
   evidence it had. Consider "no supporting passage found for: …".
5. A workload vendor in "X or Y" wording ("Can Veeam back up to AWS or Microsoft Azure?")
   still counts as a comparison side.
