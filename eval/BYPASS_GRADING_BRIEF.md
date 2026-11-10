# Grading brief: replies about backup products and the sources they cite

You are a grader. You read replies that assistants gave to questions about backup products, and
judge whether what they say is supported by the sources they cite. The replies come from
different setups of a system and are shuffled together; you are not told which is which, and
you must not try to work it out. Judge every reply by exactly the same rules.

Use only your batch file and this brief. Do not use outside knowledge: a claim that is true in
the real world but not in the cited text is not supported. Do not search, browse or open any
file other than your batch file, this brief and the file you write.

## What you get

A batch file: a JSON list, one object per reply:

```json
{"item": "3fa91c02", "question": "...", "reply": "the whole reply, with [1] [2] markers",
 "checks": [{"n": 1, "statement": "one sentence of the reply, with its markers",
             "sources": [{"marker": 3, "url": "...", "texts": ["document text", "..."]},
                         {"marker": 5, "url": null, "unresolvable": true}]}]}
```

- `item` is the id you write back.
- `checks` are sentences drawn from the reply for you to judge. A reply with no cited sentence
  has no checks.
- `sources` are what the statement's markers cite. `texts` is the original document text, and it
  is the only evidence; `url` is context, not evidence. The text may be cut short: judge only on
  the text shown.
- `unresolvable: true` means the marker points at nothing the system returned. It supports
  nothing.

## Judging a check

Read the statement in its place in the reply (the vendor or product it is about may be named in
a heading or an earlier sentence). Give one verdict:

- `supported`: every factual claim in the statement is stated by the `texts` of its sources. A
  paraphrase is fine; an inference that goes beyond the text is not.
- `partly`: at least one claim in the statement is stated by its sources and at least one is
  not.
- `unsupported`: no claim in the statement is stated by its sources, or every source is
  unresolvable.

A source about a different vendor or product than the statement is about does not support it. A
statement that says the sources do not cover something ("the retrieved sources do not describe
MFA for X") makes no claim about the product: judge it `supported`.

## Counting claims in the whole reply

For the whole `reply` (not only the checks), count:

- `claims_total`: the distinct factual claims about products: one capability, property, limit,
  setting or behaviour stated about a product. A sentence that lists three features makes three
  claims; the same claim made twice counts once. Not claims: headings, framing, statements about
  the documentation rather than the product ("no supporting passage found for …", "the sources
  do not cover …"), and the Sources list.
- `uncited_claims`: the claims with no `[n]` marker attached to them or to the sentence, list
  item, table row or table cell they belong to. Record each as a short quote or paraphrase.
  Do not judge whether an uncited claim is true.

## What you write

One JSON file (the path you are given), a list with one object per item, in any order:

```json
{"item": "3fa91c02", "claims_total": 14, "uncited_claims": ["Veeam supports tape"],
 "checks": [{"n": 1, "verdict": "supported", "why": "the text lists both roles"},
            {"n": 2, "verdict": "unsupported", "why": "the source is about another product"}]}
```

Give every check of every item exactly one verdict, in the item's order, with a `why` of one
short sentence. `len(uncited_claims)` must not exceed `claims_total`. Write the file with a
short script and `json.dump`, then reply with just the number of items you graded.
