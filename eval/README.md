> Moved from graph-rag, where this set decided the 2026-11-04 head-to-head (report in
> `docs/superpowers/head-to-head-report-2026-11-04.md`). The validator, runner and judge
> harness (`scripts/headtohead/`) stay in graph-rag because they depend on the graph stack,
> as do `corpus.json`, `ingest_report.json` and the raw run outputs. Paths below are the
> original graph-rag ones.

# Head-to-head eval set: format and writing rules

`eval/headtohead/questions.yaml` holds about 136 questions: 100 new questions, which
decide the result, plus 36 existing ones, which are reported separately. It is a YAML
list. Validate it with:

    uv run --extra dev python -m scripts.headtohead.questions --check eval/headtohead/questions.yaml

The check needs `eval/headtohead/corpus.json` and `eval/headtohead/ingest_report.json`
(exit 2 if either is missing or unreadable, or if the questions file is not a YAML list).
Every problem is printed, prefixed by the question id. Article ids may be written as
strings or integers; both are compared as strings.

## Fields

Each question has:
- `id` (unique) and `type`: `single_vendor`, `multi_article`, `cross_vendor` or `unanswerable`;
- `question`: the question text;
- `gold_facts`: short, individually checkable statements;
- `gold_sources`: article ids, with `section` where known (non-empty unless unanswerable);
- `vendors`: the vendor(s) involved, each one of the corpus vendors (the `vendor` values in
  `corpus.json`: Veeam, Microsoft, AWS, Cohesity);
- `written_from`: the article id(s) it was written from (non-empty unless unanswerable);
- `decides`: true exactly when `origin` is `new`;
- `origin`: `new`, `golden_questions` or `router_golden_tier1`;
- `search_check`: unanswerable questions only, a non-empty record of the search used to
  check that the corpus does not answer it.

Every gold source and `written_from` id must be in the 20,899-article corpus and must not
have changed since the graph was built (the ingest report's `changed_since_graph`).

## New questions (100)

Written by Opus subagents from the articles themselves, sampled across many sources, not
concentrated in a few. Single-vendor questions are stratified by vendor, and the mix is
reported.

| Type | Count | Notes |
|---|---|---|
| single-vendor detail (`single_vendor`) | 50 | biased towards limits, conditions, versions, prerequisites |
| multi-article (`multi_article`) | 15 | answering needs two articles, e.g. a prerequisite on one page, a limit on another |
| cross-vendor (`cross_vendor`) | 15 | only on topics at least two of the four vendors' extracted articles cover |
| unanswerable (`unanswerable`) | 20 | plausible questions the 20,899 articles do not answer, with the search used to check that recorded |

The count check applies to `origin: new` questions only. Existing-origin questions are
not limited.

## Gold-fact rules

- At most 8 gold facts per question. No "list everything" questions whose complete answer
  would be longer than that.
- Gold facts are the *minimum* a perfect answer must contain. Extra items that are correct
  and supported by their citations never count against an answer.
- **Cross-vendor gold** is built by checking every one of the four vendors' extracted
  articles for the topic. The gold says, per vendor, "documented: ..." or "no evidence in
  the corpus", so finding every vendor is rewarded. A cross-vendor question has at least 2
  `vendors`, and its gold facts name every one of them (case-insensitive whole word).
- An unanswerable question has `gold_facts: []`, a non-empty `search_check`, and may have
  empty `gold_sources` and `written_from`.

## Writing rules

- Phrase the question the way a backup admin would ask it, **not in the article's words**.
  Distinctive terms that a user would genuinely type (product names, feature names) are
  allowed.
- Write gold facts so that a judge can check them in an answer.
- Each subagent gets the article text and the rules, and writes a fixed number of
  questions.

## Existing questions (36), reported but not deciding

answer-api's retrieval and router were tuned against these, and they predate the
rewording rule. Letting them count would tilt the decision towards answer-api.
- `golden_questions.json` (15) and `router_golden_tier1.json` (21) have expected article
  ids but no gold facts.
- An Opus subagent adds gold facts from the expected articles.
- A question whose expected articles are outside the 20,899, or have changed, is dropped
  and reported.

They carry `decides: false` and their `origin`.

## User review

Before any system is run, the user reviews a sample of about 20 questions across all types
for realism. Their edits apply to the set, and the set is frozen and committed before the
run.

## Example records

```yaml
- id: sv-001
  type: single_vendor
  question: How many restore points can a backup copy job keep before it starts merging?
  gold_facts:
    - "Veeam: a backup copy job keeps at most N restore points (value from the article)."
  gold_sources:
    - {article_id: "12345", section: "Retention"}
  vendors: [Veeam]
  written_from: ["12345"]
  decides: true
  origin: new

- id: ma-001
  type: multi_article
  question: What do I need in place before enabling X, and what is its size limit?
  gold_facts:
    - "Prerequisite stated in the first article."
    - "Limit stated in the second article."
  gold_sources:
    - {article_id: "12345"}
    - {article_id: "23456", section: "Limits"}
  vendors: [Veeam]
  written_from: ["12345", "23456"]
  decides: true
  origin: new

- id: cv-001
  type: cross_vendor
  question: Which of the big backup vendors can restore a VM straight into Azure?
  gold_facts:
    - "Veeam: documented: ..."
    - "Microsoft: documented: ..."
    - "AWS: no evidence in the corpus"
    - "Cohesity: documented: ..."
  gold_sources:
    - {article_id: "34567"}
  vendors: [Veeam, Microsoft, AWS, Cohesity]
  written_from: ["34567"]
  decides: true
  origin: new

- id: un-001
  type: unanswerable
  question: Does the product support backing up a mainframe tape library?
  gold_facts: []
  gold_sources: []
  vendors: [Veeam]
  written_from: []
  decides: true
  origin: new
  search_check: "FTS 'mainframe tape library' and vector search top 10: nothing relevant."
```
