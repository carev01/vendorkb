# Topic expansion: "ransomware protection" (design, 2026-11-08)

Two parts, built in this order:

- **Part A: facts survive page updates.** A prerequisite: the capability table must not lose a
  page's facts every time the page changes.
- **Part B: the "ransomware protection" topic.** Broad ransomware questions answered with the
  whole protection kit.

# Part A: facts survive page updates

## Why

Today a re-synced page replaces its whole document row, and the deletion cascades through its
chunks, extractions, claims and capability rows: one edited line removes every fact the page
supported until the page is extracted again (and while the refresh job is off, it never is). The
user's view: capabilities are rarely dropped; they usually improve, and a page usually changes
because of an improvement. A real deprecation can wait for the next extraction to reach the
table.

## Goals

1. When a page is updated, its previous facts stay in the capability table, and in every query
   and answer that reads it, until the updated page has been extracted and mapped; then the new
   facts replace them.
2. Facts kept from an earlier version are marked as such wherever they are shown (capability
   query results, the topic grid, MCP output), so a reader knows the page has changed since.
3. Citations of such facts still resolve to the page (its URL and title), and their evidence
   quote is the text it was extracted from.

## Decisions

- *(My call)* A page **removed** upstream loses its facts at removal, as today: its citations
  would point to a page that no longer exists, and no extraction will ever replace them.
- *(My call)* A page updated several times before it is extracted keeps the facts of the last
  version that was extracted; nothing older accumulates.
- Replacement happens per page, once every chunk of the updated page has a final extraction
  (done or failed), so a partially extracted page never loses its old facts early.
- Left to the builder: how chunks, extractions and claims of the previous version are kept
  (for example, claims keep their document and quote while their chunk is replaced), and the
  migration. It is a schema change: a pg_dump before deploy, applied by the sync job's
  `init_schema`; tested on a populated database.

# Part B: the "ransomware protection" topic

## Intent and why

Ransomware protection is the security topic users ask about most, and it is critical to them.
Asked broadly ("How does Veeam protect backups against ransomware?"), today's retrieval returns the
pages that mention ransomware, mostly malware detection, and misses the rest: in eval question
ex-rt-08 the Veeam side covered malware detection and encryption but never immutability, because
the immutability and hardened-repository pages were not even among its candidates (the question
does not say "immutable").

A broad ransomware question should be answered with the whole protection kit, for every product
in the question, from evidence retrieved for each part of the kit.

## The kit (the user's definition)

Eight aspects: the user's five, then the three additions the user approved. Each maps to existing
taxonomy dimensions.

| # | aspect | what it covers | dimensions |
|---|---|---|---|
| 1 | Built-in security | RBAC, MFA, SSO; audit logging and SIEM alerting ride along | `security.rbac`, `security.mfa`, `security.sso`, `security.audit_logging`, `cyber.siem_integration` |
| 2 | Immutability / WORM | backups that cannot be changed or deleted before retention ends | `security.immutability`, `security.immutability_level`, `security.immutability_mode`, `security.immutability_scope`, `security.immutability_period` |
| 3 | Anomaly detection | suspicious activity: abnormal numbers of changes, deletes or renames, spikes in backup size, similar techniques | `cyber.detection_behavior` |
| 4 | Malware detection | antivirus signatures, machine-learning or entropy analysis, YARA rules, IoC and hash scanning, third-party scanners | `cyber.detection_signatures`, `cyber.detection_entropy`, `cyber.detection_rules`, `cyber.indicators_of_compromise`, `cyber.third_party_scanners` |
| 5 | Clean recovery | scanning backups during restore, quarantining infected data, finding the last clean restore point, clean-room recovery, so threats are not reintroduced | `cyber.scan_on_restore`, `cyber.quarantine`, `cyber.clean_point_identification`, `cyber.clean_room` |
| 6 | Air gap / isolated copies | logically or physically isolated backup copies | `security.air_gap` |
| 7 | Deletion safeguards | soft delete of backups, a second approver for destructive operations | `security.soft_delete_of_backups`, `security.multi_person_authorization` |
| 8 | Backup encryption | encryption at rest and in transit, client-side, customer-managed keys | `security.encryption_at_rest`, `security.encryption_in_transit`, `security.client_side_encryption`, `security.customer_managed_keys` |

## Goals

1. A curated topic, defined once in the reviewed taxonomy file (`taxonomy/taxonomy.yaml`),
   listing the topic's trigger phrases and its aspects, each aspect with its dimensions and the
   phrasings retrieval uses for it. The loader validates it (every dimension must exist).
2. A broad question on the topic gets an answer that addresses every aspect for every product in
   it, with citations, in all three answer shapes: single vendor or product, comparison, and
   "which vendors" (enumeration).
3. Evidence is retrieved per aspect with a small quota each, so one aspect's pages cannot crowd
   out another's.
4. MCP agents can use the same definition: `query_capabilities` can filter by the topic (facts
   across its dimensions, grouped by aspect), and `list_dimensions` finds the topic and its
   aspects when asked about ransomware.
5. Measured before and after on labelled ransomware questions: aspect coverage rises, precision
   (unsupported claims) does not fall.

## Non-goals

- Model-generated expansions. The topic is curated so its behaviour is predictable and testable.
- Other topics now. The design is generic (a list of topics in the YAML), but only ransomware
  protection is defined in this work.
- Narrowing a question that names one aspect to that aspect. Such a question keeps today's
  behaviour.

## Anti-goals (failure even if it "works")

- An answer saying a product lacks an aspect because nothing was retrieved for it. Absence of
  retrieved evidence is reported as "the retrieved sources do not cover this", never as "not
  supported".
- Expansion firing on an aspect-specific question ("Does Veeam support YARA rules?", "How does
  immutability protect against ransomware?") and diluting it with the other seven aspects.
- Unbounded rerank spend: an expanded question must stay within a fixed rerank budget.
- A slower or worse answer for every other question: questions that do not trigger the topic
  behave exactly as today.

## How (decided in conversation)

- **Where the topic lives:** in `taxonomy/taxonomy.yaml`, next to the dimensions it points to,
  reviewed in git and versioned with the taxonomy. It is loaded into the database with the
  taxonomy, so the MCP server reads the loaded version like the rest of the taxonomy.
- **When it fires:** the question names the topic (a trigger phrase such as "ransomware") and
  names none of the aspects. A question that names an aspect keeps today's retrieval.
- **Retrieval:** per aspect, using the aspect's phrasings (together with the question's products),
  a small quota of passages per aspect and per product, duplicates removed, within a fixed rerank
  budget per question.
- **Answer:** the answer model is told the question is about the topic and asked to address each
  aspect per product where the sources state it, and to say when the retrieved sources do not
  cover an aspect.
- **Capability facts as aspect evidence (the user agreed):** because the topic names its
  dimensions, the answer path may read the capability table for them without mapping the
  question. For single-vendor and comparison questions, the chunks behind supported or partial
  facts on an aspect's dimensions join that aspect's candidates, next to the phrasing search.
  (This revisits BACKLOG 8 for topic questions only; its reason, mapping a question to
  dimensions, does not apply here.)
- **"Which vendors" questions get a grid (the user agreed):** built in code, not by the model,
  from the capability table: one row per vendor, one column per aspect; a cell holds the best
  fact (supported over partial, current over superseded, newest page first) as a mark with a
  citation; a blank means no fact was found and a footnote says it is not evidence of absence.
  A short model-written summary sits above the grid.
- **Catch-up search (the user agreed):** for each aspect, a search over only the pages that have
  no facts yet (synced since their last extraction) adds any hits to the grid as cited marks,
  so a recent page is not missing just because extraction has not reached it.
- **Date footer (the user agreed):** the grid states the date its facts are current to.
- *(My call)* A question naming a **vendor** treats the vendor as one unit, and every claim names
  its product (the citation labels already carry vendor and product); a question naming
  products keeps them as separate sides.
- *(My call)* A broad ransomware question naming **no vendor or product** ("How can I protect my
  backups against ransomware?") is answered by aspect, from per-aspect searches across the
  corpus, citing the vendors that document each aspect, with no claim of completeness; the grid
  is reserved for "which vendors" wording.

## Constraints

- Precision first, as everywhere in vendorkb: every claim cited, no outside knowledge, no claim
  of absence.
- Rerank tokens are the main cost (Voyage). Today a comparison question uses about 25k, an
  enumeration about 65k. An expanded question must have a hard cap (the builder proposes one,
  measured).
- A schema change needs a pg_dump of production before deploy (standing rule); production DDL
  is applied by the sync CronJob's `init_schema`.
- Agents' behaviour is guarded by the agent eval gate (`eval/agent_eval.py`): rerun it after the
  MCP change.

## Evaluation

- A small labelled set of broad ransomware questions (my call: about ten): single-vendor for the
  Tier 1 vendors (Veeam, Commvault, Rubrik, Cohesity), a few comparisons, one vendor-neutral
  question and two "which vendors" questions, in `eval/questions.yaml` with the existing format.
- Blind grading by two Claude graders (Opus and Fable, as for the other labelled sets),
  disagreements adjudicated: for each answer, product and aspect, covered with a supporting
  citation, covered without support, or not covered; and every unsupported claim. For the grid,
  a sample of marks is graded for support, and blanks are spot-checked against a search.
- *(My call)* **Pass:** aspect coverage (aspects covered with support, over the eight per
  product) rises clearly over today's answers, and the unsupported-claim *rate* (per claim, since
  longer answers make more claims) rises by no more than two points. An aspect a product's
  corpus does not document, reported as "the retrieved sources do not cover it", is not counted
  against the answer.
- The before/after run uses the same questions and production data; the agent eval gate is
  rerun after the MCP change.

## Left to the builder

- The exact trigger phrases, aspect phrasings and aspect cues (words that mean an aspect is
  named), reviewed like any taxonomy change.
- Table names and the loader's validation messages.
- Per-shape retrieval details: candidates per aspect, passages per aspect and product, how the
  aspect passages and the question's own top passages are combined, and the rerank cap.
- The prompt wording.
- Tests and the report.
