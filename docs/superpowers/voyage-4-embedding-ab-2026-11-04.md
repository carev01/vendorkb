# voyage-4 vs. jina-embeddings-v5-text-nano: retrieval A/B (2026-11-04)

**Result: voyage-4 is adopted.** It was promoted in the local database the same day, and
the code defaults now point at it.

## Setup

- **Corpus:** the 20,899 articles (32,280 chunks) from the head-to-head. Each chunk's
  stored text, including its title and section prefix, was re-embedded with
  `vendorkb reembed` into `chunks.embedding_next`.
- **Model settings:** voyage-4 at 1,024 dimensions, with `input_type` set to `document`
  for chunks and `query` for questions. The jina side ran as in the head-to-head:
  768 dimensions, no prefixes.
- **Indexes:** both columns had an HNSW index.
- **Cost:** 16.8M Voyage tokens in 304 s (128 texts per request, 4 requests in flight, no
  rate-limit responses). Voyage counted 0.84× the chars/4 estimate. The A/B queries used
  6,104 tokens.
- **Script:** `eval/retrieval_compare.py`. It runs retrieval the way `answer()` does:
  scope detection, hybrid search, and relaxing the scope when nothing is found. It stops
  before the paid rerank.
- **Questions:** the 99 answerable questions in `eval/questions.yaml`: 80 deciding and 19
  reported.
- **Measured:** the rank of the first gold article in the 40-chunk rerank pool, and in
  the vector-only top 50.

## Results

| group | metric | jina (live) | voyage-4 |
|---|---|---|---|
| all (99) | gold in rerank pool | 0.778 | **0.949** |
| all | gold in pool top 10 | 0.545 | **0.788** |
| all | vector-only recall@10 | 0.707 | **0.919** |
| all | vector-only MRR | 0.482 | **0.686** |
| deciding (80) | gold in rerank pool | 0.838 | **0.950** |
| cross-vendor (15) | gold in rerank pool | 0.800 | **0.933** |
| single-vendor (50) | gold in rerank pool | 0.820 | **0.960** |
| multi-article (15) | gold in rerank pool | 0.933 | 0.933 |

Per question:
- The gold article's position in the pool improved on 53 questions and worsened on 19.
- Two questions lost the gold article from the pool: sv-b-11 and ma-b-06.
- Three questions vendorkb refused in the head-to-head now have their gold article
  ranked well:
  - sv-d-06: from absent to rank 3 in the pool and rank 1 by vector.
  - sv-b-12: from absent to rank 19 in the pool and rank 8 by vector.
  - sv-d-01: from rank 18 to rank 6 in the pool.

## Follow-up

**Keyword search now dilutes the ranking.** With voyage-4, vector-only retrieval ranks
gold articles well above the hybrid merge: recall@5 is 0.879 by vector against 0.616 for
the merged pool. The full-text side ORs every query term, so long changelog and alarm-table
chunks match many terms and take the top positions. The reranker sees all 40 pool chunks,
so pool recall is what matters for answers. Even so, try down-weighting the full-text
ranking in the merge, or capping how many chunks it contributes, before the next answer
eval.
