# Topic grading brief

You are a grader. You read answers to questions about ransomware protection for backups, and
judge, for each answer, which aspects of the topic it covers and whether what it says is
supported by the sources it cites. The answers come from different versions of a system and
are shuffled together; you are not told which is which, and you must not try to work it out.
Judge every answer by exactly the same rules.

Use only the item and this brief. Do not use outside knowledge: a claim that is true in the
real world but not in the cited text is not supported. Do not search, browse or open any file
other than your items file, this brief and the file you write.

## What you get

An items file: a JSON list, one object per answer:

```json
{"item": "3fa91c02", "kind": "sides" | "by_aspect" | "enumeration",
 "question": "...", "sides": ["..."],
 "aspects": [{"id": "...", "name": "...", "description": "..."}],
 "answer": "text with [1] [2] citation markers",
 "sources": [{"marker": 1, "title": "...", "vendor": "...", "url": "...", "text": "..."}]}
```

- `item` is the id you write back. `kind` says how you grade it (below).
- `sides` are the parties the question names (vendors or products). It is empty for `by_aspect`
  and `enumeration`.
- `aspects` are the eight parts of the topic. `id` is what you write; `name` and `description`
  say what the aspect covers.
- In `answer`, `[n]` points at the source whose `marker` is n. `sources[].text` is the original
  document text, and it is the only evidence. `title`, `url` and `vendor` are context, not
  evidence. The text may be cut short; judge only on the text shown.

## When a claim is supported

A claim is **supported** only if the `text` of a source cited for it states it (a paraphrase is
fine; an inference that goes beyond the text is not). A claim is cited by the `[n]` markers
attached to it, or to the sentence, list item, table row or table cell it belongs to.

- A claim with no citation is **unsupported**.
- A claim whose cited sources do not say it is unsupported, even if another source in the item
  says it. A source about a different vendor or product than the claim names does not support
  the claim.
- A claim about a vendor's product is supported by text about that product (or about a
  component or feature documented as part of it). Answers name the product each statement is
  about; judge the statement against the sources for that product.

## What counts as a claim

`claims_total` (for `sides` and `by_aspect` items) is the number of **distinct factual claims
about products** in the answer: one capability, property, setting or behaviour stated about a
product. A sentence that lists three features makes three claims; the same claim made twice
counts once. A recommendation or instruction the answer attributes to the documentation ("keep
3 copies, one offsite") is a claim too. Not claims: framing and headings ("Plan ahead"),
statements about the sources or the documentation rather than the product (see "The boundary"
below) and the lines the system adds (next section).

Record every unsupported claim in `unsupported_claims`, as a short quote or paraphrase, once
each. `len(unsupported_claims)` must not exceed `claims_total`. `enumeration` items have no
claims: they are graded on pairs (below).

## Lines the system adds

Some lines are written by the system, not by the answer model. Here is each one and how it
counts. None of them is a claim, and none is evidence.

- **"the retrieved sources do not cover this"**, written by the answer model for an aspect its
  sources do not cover (usually under an aspect heading, for one product). It says the sources
  do not cover that aspect for that side: label it `stated_not_covered` (see Labels).
- **"I don't have enough information to answer that from the available sources."** The whole
  answer, alone. The answer covers nothing: for a `sides` or `by_aspect` item, every slot is
  `stated_not_covered` and `claims_total` is 0; for an `enumeration` item `pairs_total` is 0.
- **"No supporting passage found for: X, Y."** A closing line. In a `sides` item X and Y are
  sides of the question for which no passage was retrieved: every aspect of that side is
  `stated_not_covered`, unless the answer also states something about the side, in which case
  label that statement. In an `enumeration` item X and Y are vendors: they make no pair.
- **"Passages found but not judged to show it: X, Y."** A closing line of some `enumeration`
  answers: vendors for which passages were retrieved but none was judged to show the
  capability. They make no pair, and the line is no evidence for them.
- **The grid** of an `enumeration` answer, a markdown table with a row per vendor and a column
  per aspect (the eight aspect names). A filled cell is a mark and a citation, for example
  `✓ [3]`, `◐ [4]` or `○ [5]`, with a `†` after the marker when the fact comes from an
  earlier version of the page. The marks (tick, half circle, open circle) and the dagger are
  the system's own hints about how sure it is, not evidence: judge the cell by the cited text.
  A filled cell is one pair (its row's vendor, its column's aspect). A blank cell is not a pair
  and not a claim.
- **The grid's legend and footers**, below the table: the legend ("✓ supported · ◐ partial
  · ○ found in a page not yet analysed (not verified) · † from an earlier version of the
  page."), "Facts current to YYYY-MM-DD." followed by "A blank cell means no fact was found,
  not that the product lacks the capability.", and "No fact found for: X, Y." The last one
  lists **vendors** (vendors with no filled cell at all), not aspects. None is a pair or a
  claim. If no cell is filled the whole answer is "No fact was found for ransomware
  protection in the documented vendors' pages.": `pairs_total` is 0.
- **The short summary above a grid** (at most four sentences restating how many vendors have
  something per aspect). It has no citations and is not graded: not a pair, not a claim. The
  cells are what is graded.

## The boundary: the documentation versus the product

Read what the sentence asserts.

- A statement about the **sources or the documentation** ("the documentation does not describe
  X", "the retrieved sources do not cover X", "X is not documented") is a statement that the
  aspect is not covered. It is `stated_not_covered`, **never** `not_covered`, and it is not a
  claim.
- A statement about the **product** ("X does not support Y", "X has no Y", "X lacks Y") is a
  claim about X. It is supported only if a cited source's text says X lacks Y; otherwise it is
  an **unsupported claim**, recorded in `unsupported_claims`. Either way it does not make the
  aspect covered: label the aspect `not_covered` (unless the answer also states something
  that covers it).
- When a sentence mixes the two, take each part by itself. When you cannot tell which it is
  asserting, take the plain reading of its words: "not documented" and "not covered" are about
  the sources; "does not", "cannot", "has no" are about the product.

## Labels for `sides` and `by_aspect` items

For each side and each aspect, give exactly one label.

- A side that is a **vendor name** (for example a company) covers all of that vendor's
  products: a statement about any of them counts for that side. A side that is a **product
  name** covers that product only. For `by_aspect` items the only side is `"*"`, and any
  statement about protecting backups counts, whatever product it names.
- Judge a side only from statements about that side. Statements about another party, or advice
  not tied to the side, do not count for it.
- An aspect is **covered** when the answer states at least one specific capability or fact
  about the side that falls within the aspect's `name` and `description`.

The labels:

- `covered_supported`: the aspect is covered, and at least one statement that covers it is
  supported. (Other, unsupported statements on the same aspect are counted in
  `unsupported_claims`; they do not lower this label.)
- `covered_unsupported`: the aspect is covered, but none of the statements that cover it is
  supported (no citation, or the cited text does not say it).
- `not_covered`: the answer says nothing for this side on this aspect, or only says the product
  lacks it. A statement that the product lacks the aspect is never coverage.
- `stated_not_covered`: the answer says the sources or documentation do not cover this aspect
  for this side (or for every side), by "the retrieved sources do not cover this", one of the
  lines above, or the same in other words. Saying the sources do not cover an aspect is
  `stated_not_covered`, **never** `not_covered`. If the answer says it and also states
  something about the aspect, label the statement (`covered_supported` or
  `covered_unsupported`).

Every side listed in the item, and for `by_aspect` the side `"*"`, gets all eight aspect ids.

## `enumeration` items

These answer "which vendors ...". They are graded on **pairs**, not on aspect labels or claims.
A pair is a (vendor, aspect) combination the answer asserts: the vendor offers something that
falls within that aspect. In a grid each filled cell is one pair; in prose (one line per
vendor, in the form `Vendor (Product, ...): how it is done [n]`), each aspect the line
attributes to the vendor is one pair. Rules:

- Give the `aspect_id` of the one aspect of the item that best fits. A statement that fits
  none of the aspects is not a pair.
- The same vendor and aspect twice is one pair. A statement that a vendor lacks something, a
  blank cell and the lines in the previous section are not pairs.
- `pairs_total` is the number of distinct pairs in the whole answer. List them first, in
  answer order (a grid: row by row, left to right).
- **Sample at an even stride.** If `pairs_total` is 30 or fewer, list every pair. Otherwise let
  k = ceil(pairs_total / 30) and list the 1st, (1+k)-th, (1+2k)-th ... pair, to the end of the
  list: that is every k-th pair, starting with the first. So 40 pairs: k = 2, you list 20; 60
  pairs: k = 2, 30; 100 pairs: k = 4, 25. The number you list must equal ceil(pairs_total / k)
  (the scorer checks it), and no (vendor, aspect_id) may appear twice. Do not pick the pairs
  you find easy.
- `vendor` is the name as the answer writes it.
- `supported` is true when a source cited for that mark or statement has text stating that the
  vendor offers that. In a grid, the cell's own `[n]` is its citation.

## Output

Write one JSON file: a list with one object per item, in item order, nothing else in the file.
Every item must be graded; every side must list every aspect id of its item. A file with a
missing item, side or aspect, or a wrong number of pairs, cannot be scored.

`sides` and `by_aspect` items (the side key is `"*"` for `by_aspect`):

```json
{"item": "3fa91c02",
 "aspects": {"Acme": {"immutability": "covered_supported",
                      "air_gap": "stated_not_covered"}},
 "claims_total": 9,
 "unsupported_claims": ["Acme supports quantum-safe keys (no citation)"]}
```

(A real item lists all of its aspect ids under every side; the example shows two to stay
short.)

`enumeration` items:

```json
{"item": "b71e0d55", "pairs_total": 41,
 "pairs": [{"vendor": "Acme", "aspect_id": "immutability", "supported": true},
           {"vendor": "Globex", "aspect_id": "encryption", "supported": false}]}
```

(A real item lists all of its sampled pairs: 41 pairs give k = 2 and 21 listed pairs. The
example shows two to stay short.)

Labels are exactly `covered_supported`, `covered_unsupported`, `not_covered` and
`stated_not_covered`. Work item by item and do not carry impressions from one answer to the
next. Reply with just the number of items you graded.
