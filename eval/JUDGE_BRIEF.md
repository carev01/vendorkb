# Head-to-head judge brief

You are judging two answers to the same question, blind. Two systems answered every
question; their answers appear under the anonymous labels **A** and **B**, shuffled per
question. Do not try to work out which system wrote which answer. Judge both by exactly
the same rules.

## What you get

You are given one batch file (`batches/batch-NNN.jsonl`, 2 tasks). Each line is one task:

```json
{"key": "...", "type": "single_vendor|multi_article|cross_vendor|unanswerable",
 "question": "...", "gold_facts": ["...", "..."],
 "answers": {
   "A": {"errored": false, "text": "... [1] ... [2]",
         "sources": [{"n": 1, "paraphrase": "..." or null,
                      "paraphrase_note": "system paraphrase (not evidence)" or null,
                      "unresolvable": false, "documents_omitted": 0,
                      "documents": [{"title": "...", "text": "...",
                                     "same_text_as_source": null, "truncated": false}]}]},
   "B": {...}}}
```

- `sources[].n` is the citation number: the answer's `[n]` markers point at it.
- `documents[].text` is the **original document text** behind that citation. This is the
  only evidence.
- `paraphrase` (when present) is the system's own restatement of what it extracted from
  the documents. It is shown for orientation only and is **never evidence** (see the
  source-text rule).
- `same_text_as_source: m` means this document's text is identical to one already shown
  under source `m` of the same answer; read it there.
- `truncated: true` means the harness cut that document's text to keep the context within
  its size cap; `text` ends at the cut. Every citation of an answer gets a fair share of
  that cap, so each citation keeps some text. Judge only on the text shown.
- `documents_omitted: k` means k more documents behind that citation were not shown (a
  per-citation limit, or the size cap). Judge only on the text shown.
- `unresolvable: true` means no source text could be found for that citation. A citation
  with no source text supports nothing.
- `errored: true` (with `text: null` and no sources) means that system failed to produce an
  answer. Do not judge it (see "Errored answers").

## The source-text rule

A claim is supported only if the **original document text** of a source cited for it
supports it. An extracted fact or a generated report is the system's own restatement,
never evidence: a `paraphrase` that says something the documents do not say does not
support anything. Do not use your own knowledge either: a claim that is true in the real
world but not in the cited text is not supported.

A claim is supported by *its* citations: the `[n]` marker(s) attached to it, or to the
sentence, list item or table row it belongs to. A factual claim with no citation attached
is not supported. Text that makes no factual claim (framing such as "Here is what the
documentation says:") is not a claim.

## Fixed system lines

Some answers contain fixed lines produced by the system rather than written as prose.
Treat them exactly like this:

1. **`(Vendor · Product)` labels are attribution metadata, not claims.** Do not record a
   label as a claim and do not mark anything unsupported because of a label.
2. **An item flagged "needs validation" is a declared hedge.** It is neither counted as an
   unsupported claim nor credited as presenting a gold fact: leave it out of `claims`, and
   it cannot make a gold fact present. The flag takes either of two forms:
   - an inline note after the item, such as *"(needs validation: the cited source does not
     mention …)"* or *"(needs validation: the cited source does not name this item)"*;
   - a bare **" †"** after the item, explained by a group note such as *"Items marked †
     come from documentation that does not mention X specifically; validate them for your
     case."* Every item marked † is a hedge. The group note itself is not a claim.
3. **In the enumeration line "Checked … / Documented for …", each "documented for X" is a
   claim** that needs support from the citations: record each one as a claim. "Checked"
   describes the process and is not judged: leave it out of `claims`.

## Gold facts

- `gold_facts` are the *minimum* a perfect answer must contain. Mark each one present if
  the answer states it (in any wording), and absent otherwise. Presence is judged on the
  answer text; whether it is supported is judged separately, under `claims`.
- Extra items that are correct and supported by their citations never count against an
  answer.
- Cross-vendor gold names each vendor: "documented: ..." or "no evidence in the corpus".
  A "no evidence in the corpus" fact for vendor X is **present** when:
  - X is listed in the enumeration line's "Checked" part but not in its "Documented for"
    part; or
  - the answer states explicitly in prose that nothing on the topic was found for X
    (this applies to either answer, with or without an enumeration line).

  It is **absent** when the answer is silent about X. An answer that claims X documents
  the topic makes a claim that needs support like any other.
- Unanswerable questions have no gold facts (`gold: []`).

## Abstention (you decide)

Decide whether the answer **abstained**: whether, as a whole, it declines to answer the
question, for example by saying the available sources do not cover it. Decide from what
the answer says; do not match any particular refusal wording. An answer that gives an
answer and adds a caveat has not abstained.

- Answerable question: an abstaining answer is not perfect.
- Unanswerable question: the correct behaviour is to abstain with no invented answer. In
  `claims`, record every statement the answer offers *as an answer to the question*,
  supported or not. Statements that only explain the abstention (what the sources do and
  do not cover) are not recorded.

## Citations

`citations_ok` is true only if every citation in the answer supports what it is attached
to. A citation whose source text does not support the statement it is attached to, or an
`unresolvable` citation, makes it false. An answer with no citations has `citations_ok`
true (its uncited claims are already unsupported, under `claims`).

## Errored answers

An errored answer appears in your batch with `errored: true`, `text: null` and no sources.
Do not judge it: it is scored as not perfect without a verdict. Write `{"errored": true}`
for that label (or leave the label out), and judge the other answer normally. If both
answers errored, the task is not in your batch.

## Your answer format

Write `judgements/batch-NNN.jsonl` (the same NNN as the batch you were given), **one JSON
object per line, one line per task**, nothing else in the file:

```json
{"key": "<the task's key>",
 "A": {"gold": [true, false],
       "claims": [{"text": "short quote or paraphrase of the claim", "supported": true}],
       "citations_ok": true,
       "abstained": false,
       "notes": "optional: one or two sentences on the deciding problems"},
 "B": {"gold": [...], "claims": [...], "citations_ok": false, "abstained": false}}
```

- `gold` has exactly one boolean per gold fact, **in the order of `gold_facts`**
  (an empty list for unanswerable questions). A wrong count is rejected and re-asked.
- `claims` lists every factual claim in the answer (subject to the fixed-line rules), each
  with whether its citations' document text supports it.
- `citations_ok` and `abstained` are booleans.
- `notes` is optional free text; it is used when reporting examples.
- Both `A` and `B` must be present unless that answer errored.

## How your verdict is used

An answer is **perfect** when every gold fact is present, no claim is unsupported, every
citation supports what it is attached to, and it did not abstain; for an unanswerable
question, when it abstained with no claim offered as an answer. Every question is judged
in two independent passes and a third when they disagree. Be strict and consistent: the
same rules apply to both answers.
