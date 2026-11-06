# Taxonomy v2 and claims v3: making comparison information findable

Status: draft for review, revision 2, 2026-11-06.
Supersedes: the taxonomy shape in DESIGN.md §5.4, taxonomy v1 (PR #1), and the `claims-v2` prompt
(never run beyond a 20-chunk smoke).

## 1. Why

vendorkb is a knowledge source, not a report generator. Two kinds of consumer query it:
- an agent connected through MCP;
- people asking it directly.

Either may be building a vendor comparison. The system's job is to **find the information that
work needs**, in the claims and the chunks behind them, quickly and with its evidence. It does not
build, lay out or rate comparison tables.

**What "the information a comparison needs" means.** The user shared a set of internal comparison
tables (mostly Veeam, one Commvault). They show:
- which aspects comparisons cover;
- at what level of detail;
- with what nuances (editions, offering variants, availability stage, default state, quantities,
  explicit gaps and their reasons).

Those aspects define what must be findable. They must also extrapolate: when Okta protection ships,
a consumer asking about it should find the same kinds of facts it finds for Entra ID today.

**Why v1 falls short:**
- The taxonomy grew bottom-up from Haiku claims as a flat list. It has no notion of workload kinds,
  so nothing tells a consumer which aspects apply to Okta.
- It has nothing for licensing metrics, support terms, certifications, service-level versus
  storage-level immutability, and similar aspects.
- Claims carry a 4-value support and free-text conditions, so qualifiers cannot be filtered:
  "only in Premium", "preview", "up to 99 years".

**The comparison tables are internal, and this repository is public.** Nothing from them is
committed: no files, no quotes, no row lists. Templates in the repo are written in our own words.
The findability check (§7.1) reads them from outside the repo.

## 2. Goals, non-goals, anti-goals

**Goals**
1. For every aspect a comparison covers, and documentation can answer, a consumer can find the
   relevant facts for any vendor. Through the capability table when claims exist, and through
   chunk retrieval otherwise.
2. The taxonomy tells a consumer which aspects apply to a workload, including a new workload of a
   known kind.
3. Qualifiers are structured, so consumers can filter and compare. Examples: "minimum RPO across
   vendors", "what is preview-only", "what needs the top edition".
4. The Opus teacher set is labelled once, in the final format, so the 9B student learns that
   format.

**Non-goals**
- **Building comparison tables, verdicts or ratings.** That is the consumer's work.
- **Pricing.** It comes from other sources and is rarely in documentation. Licensing metrics are
  in scope: per user, per instance, per node, what each edition includes, free tiers, portability.
- **Inferred dependencies.** "Needs product X because product Y lacks Z" is the consumer's
  conclusion. When a document states a dependency outright, it goes in `conditions`.
- **Known issues, bugs and procedures.** They stay excluded, as in v2.

**Anti-goals**
- **A taxonomy that names a vendor's product as a dimension.** Dimensions are vendor-neutral, and
  products are targets or offerings.
- **Inventing coverage.** No result means "no documented evidence found", never "not supported".
  Product-wide facts are never presented as workload-specific (§6.2).

## 3. Claims v3

`PROMPT_VERSION = "claims-v3"`. One claim is one fact from one chunk, with a verbatim evidence
quote. The evidence gate is unchanged.

### 3.1 Fields

| Field | Type | Meaning |
|---|---|---|
| kind | enum | `capability`: what the product does, or explicitly does not do. `licensing`: license metric, what an edition or license includes, free tiers or editions, license portability. `service`: technical support terms, certifications and compliance attestations, release cadence, marketplace availability. |
| capability | string | Short normalized phrase, lowercase, no product names. As today. |
| target | string or null | The workload, platform, service or storage it applies to. As today. Never the product's own components. |
| target_scope | string or null | The object type or variant within the target: "users", "conditional access policies", "sandbox orgs", "FSx for Lustre", "RDS for Oracle". A capability stated for several object types becomes one claim per object type. |
| support | enum | `supported`, `partial`, `not_supported`, `unclear`, with the v2 definitions (prerequisites and defaults are never "partial"). |
| offering | string or null | The vendor's offering or deployment variant, only when the text names one narrower than the Product header: "Veeam Data Cloud", "self-managed", "Grid appliance", "Express". |
| edition | string or null | The edition or license tier the text attaches: "Premium", "Advanced and above". |
| availability | enum | `generally_available`, `preview`, `early_access`, `limited_availability`, `planned`, `deprecated`, or `unstated`. Only what the text says: a plain present-tense statement is `unstated`. |
| default_state | enum or null | `on` or `off` when the text says so, else null. |
| value | number or null | The quantity, for a limit or a measure: retention 99, RPO 10, regions 23, records 500000. |
| unit | string or null | The unit of `value`: "years", "minutes", "regions", "records per job", "attributes", "hours". |
| bound | enum or null | `max`, `min` or `exact` for `value`. "Up to 99 years" is max. "As often as every 15 minutes" is min (the smallest interval). |
| conditions | string or null | Everything else the text attaches: platforms ("VMware and Hyper-V only"), prerequisites, a dependency the text states outright, the reason for a "not supported". |
| evidence_quote | string | As today. |

That is 14 fields. The JSON schema is strict, with every field required and nullable where shown,
like `CLAIM_SCHEMA`.

**Why there is no `requires` field.** The first draft had one. Dependencies on another product are
almost never stated outright in documentation. A reviewer usually infers them by comparing what
two products cover. A field that is nearly always null invites the extractor to guess, so a stated
dependency goes in `conditions` instead.

### 3.2 Prompt changes from v2

- Licensing and support facts are claims (`kind`), no longer excluded.
- Coverage claims are one per object type, so a sentence that lists six object types gives six
  claims sharing one quote.
- Each qualifier gets a short definition with one counter-example, like the v2 "partial" rule.
  The `offering` and `availability` rules say "only when the text says so", to keep guessing out.
- Pricing stays excluded: a price is not a claim.
- Still excluded: procedures, actions in other systems, known issues, documentation
  announcements, and component version tables.

### 3.3 Storage

- New nullable columns on `raw_claims` for each new field. `kind` defaults to `capability` for
  existing rows.
- The gate and the regate are unchanged.
- v1 claims stay as they are, marked by `prompt_version`. Taxonomy v2 builds only from v3 claims.

## 4. Taxonomy v2

One YAML file, `taxonomy/taxonomy.yaml` with `version: 2`, reviewed as a git PR. It has five
sections: dimensions, classes, targets, offerings, and platform scope (a rule, §4.5).

### 4.1 Dimensions (vendor-neutral, cross-cutting)

A dimension is one comparable aspect, for example `recovery.cross_tenant` or
`retention.maximum_period`. It has:
- `id`, prefixed by its family;
- `name`, `definition` and `aliases`;
- `applies_to`: `all`, or a list of classes;
- optionally `value_unit`, when the dimension is a quantity.

Families (the seed list; names are our own):

| Family | Covers |
|---|---|
| coverage | Which object types of a target are protected, by which operation: `coverage.backup`, `coverage.restore`, `coverage.export`, `coverage.compare`, each paired with an object type (§4.2). |
| protection | Snapshot, image-level, cloud-native service backup, agent, application consistency, changed block tracking, log backup, backup copy, replication and CDP. |
| schedule | Minimum RPO, schedule types, per-object or per-group policies, SLA-based policies. |
| retention | Maximum period, granularity, per-object retention, GFS. |
| storage | See the list below. |
| recovery | Granularity (tenant down to attribute or field), destinations (original, alternate, cross-account, cross-region, cross-tenant, sandbox, cross-platform), instant recovery, bulk and its limits, relationship and hierarchy restore, automation bypass, export formats, preview and compare, search scope and filters, self-service portal. |
| security | Immutability (service or storage level, compliance mode, what is excluded), encryption (algorithm, in transit, customer-managed keys and key services, field level, default), access control (built-in and custom roles, scoping, MFA, SSO and identity providers, IP restrictions, multi-user authorization and its scope), audit logs, platform hardening. |
| cyber | Detection methods (entropy, signatures, rules, indicators of compromise, mass changes), third-party scanners, incident intake API, quarantine and containment, clean room, scan on restore, recoverability testing, recovery orchestration, response services. |
| operations | Dashboards, reports, notifications (events and channels), multi-tenant and central management, APIs and automation, AI assistance (modes, limits, MCP), private networking, throttling against SaaS API limits, deployment and upgrade. |
| governance | Managing the data itself, beyond keeping a recoverable copy: archiving production data (moving records or items out of the live system while keeping them retrievable), legal hold and eDiscovery, sensitive-data classification, ROT (redundant, obsolete, trivial) discovery and clean-up, per-person privacy requests (find, export, delete). |
| data_use | Putting backup data to other uses: sandbox seeding, masking and anonymization for non-production copies, data integration and reuse (mounting, querying, exporting to other tools), analytics over backups, migration between platforms or tenants. |
| vendor | Licensing metric, edition inclusion, free tier, license portability, exit strategy (export, free restore tool), support terms, certifications, release cadence, marketplaces, appliance procurement, installation and hardware support. |

The storage family covers:
- deployment model (SaaS, self-managed, MSP);
- bring-your-own storage and cloud choice;
- region count and choice, and changing location after setup;
- redundancy;
- tiers, including archive tiers for backups, and tiering;
- data reduction;
- repository types and tape;
- durability and availability commitments;
- sovereign or government clouds.

**Why archiving is split from data_use (revision 2).** Archiving a SaaS workload moves data out of
production under policy. That is data management, so it goes in governance with legal hold,
classification, ROT and privacy requests. An archive *storage tier* for backups is a storage
choice, so it goes in storage. ROT clean-up acts on production data, not on backups, so it is
governance, not data_use. data_use keeps only uses of the backup copy itself.

### 4.2 Classes (workload kinds) with object-type templates

A class is a kind of workload. It lists:
- its **object types**;
- any **class dimensions** that only make sense for it, such as password-reset options on restore
  for identity, or automation bypass for CRM.

Classes nest by dotted id: `kubernetes.vm` inherits from `kubernetes`. An `applies_to` entry
matches its class and every class beneath it.

Seed classes:
- **SaaS:** `saas.productivity`, `saas.identity`, `saas.crm`, `saas.devops`, `saas.work_management`
  (issue trackers, wikis, ITSM), `saas.file_sharing`.
- **Cloud:** `cloud.compute`, `cloud.database`, `cloud.file`, `cloud.object`,
  `cloud.network_config`.
- **Virtualization and containers:** `virtualization.hypervisor`, `kubernetes`, `kubernetes.vm`.
- **On-premises:** `app.database` (databases and enterprise applications), `file.nas` (SMB, NFS,
  NDMP), `endpoint.server` and `endpoint.workstation` (by operating system).
- **Backup infrastructure:** `storage.service` (backup storage as a service), `backup.appliance`.

Example: `saas.identity` object types, as generic ids with aliases:
- `user`, `group`, `role`, `admin_unit`, `application` (enterprise app or app instance);
- `app_registration`, `access_policy` (conditional access, sign-on, authentication policies),
  `device`, `device_policy`;
- `privileged_access`, `tenant_config`, `audit_log` (audit, sign-in and system logs), `contact`.

Okta maps "sign-on policies" to `access_policy` and "System Log" to `audit_log`. A consumer asking
"what does vendor X cover for Okta?" therefore gets the same checklist of aspects as for Entra ID.

### 4.3 Targets (instances of a class)

A target is a concrete workload, for any workload a corpus vendor protects. It has:
- `id`, `name` and `aliases`;
- `class`;
- optional `parent`, for hierarchies such as `aws` → `aws.rds` → `aws.rds.oracle`;
- `object_aliases`, mapping the target's own vocabulary to class object types;
- optional `variants`, for named sub-kinds that are not object types: Salesforce clouds and org
  types, FSx file-system types.

Adding a workload means adding a target entry. Its aspects come from its class.

### 4.4 Offerings

Per vendor: `vendor`, `id`, `name`, `aliases`, and `editions` with their aliases. A claim's free-text
`offering` and `edition` map here, so "VDC", "Veeam Data Cloud" and "the SaaS offering" group
together.

### 4.5 Platform scope

Some dimensions describe the backup product itself, not a workload: hardening, AI assistance,
support, certifications, licensing. Claims for them have a null target and map with
`target_id = null`.

## 5. Building v2

1. **Seed, top-down.** Claude writes the families, dimensions, classes, object types, targets and
   offerings. The sources are the comparison-table structure, domain knowledge and the v1
   taxonomy, all in our own words.
2. **Findability check against the comparison tables (§7.1)**, iterated until every in-scope
   aspect maps to an id.
3. **Aliases, bottom-up.** Cluster the distinct v3 claim strings, as in v1. Propose aliases for
   existing ids, and new ids only where no existing one fits.
4. **Human review as a PR.** PR #1 (v1) is closed as superseded. Its branch is kept as an alias
   source.

## 6. Mapping, the capability table, queries

### 6.1 Mapper

For each v3 claim:
- **Target, variant or object type,** with `target_scope` resolved in that order:
  - a child target ("RDS for Oracle" → `aws.rds.oracle`);
  - then a variant ("FSx for Lustre" → a variant of `aws.fsx`);
  - then an object type ("users" → `user`), through the target's `object_aliases` and then the
    class aliases.
- **Target:** alias match first. When that fails, nearest neighbour gives up to 5 candidates for
  LLM adjudication.
- **Dimension:** alias match, then LLM adjudication over the top 5 candidates, with "none" allowed.
  Nearest neighbour alone never maps: the v1 calibration showed 15–60% precision. Coverage
  operations come from `capability` through the `coverage.*` dimensions' aliases.
- **Offering and edition:** through the vendor's `offerings` aliases. Text that does not resolve
  stays as free text.

The adjudication model is a setting, evaluated before use on about 300 claims labelled by Opus
subagents and checked by the user. Anything left unmapped is kept, still found by chunk retrieval,
and surfaced for the next review.

### 6.2 `capabilities` v2

One row per mapped claim, with these columns:
- `vendor`, `product`, `offering_id`, `edition_id`;
- `dimension_id`, `class_id`, `target_id`, `variant_id`, `object_type_id`;
- `support`, `availability`, `default_state`, `value`, `unit`, `bound`, `conditions`;
- `claim_id`, `doc_version`, `taxonomy_version`.

The conflict key is (vendor, product, offering, edition, dimension, target, variant, object type).
The latest document version wins, and the conflict is recorded (DESIGN.md §5.5).

**Scope labels, so nothing is presented as more specific than its evidence:**
- A claim with no offering is **product-level**. It is never presented as a fact about a specific
  offering.
- A claim with no target that describes a workload dimension (for example "retention up to 99
  years") is **product-wide**. A query for a target returns it separately, labelled
  product-wide, never as that target's fact.

### 6.3 Query interface (what MCP and direct queries use)

Two calls. The MCP server (backlog item 12) exposes them, along with the existing search and
answer tools.

- **`aspects(target | class)`** returns the aspects that apply: object types times coverage
  operations, plus every applicable dimension, with names and definitions. This is how a consumer
  learns what to ask about a workload it has never seen, Okta included.
- **`capabilities(filters)`** returns facts with their qualifiers, evidence quote and source URL.
  - Filters: vendor, offering, edition, dimension or family, target (with descendants), object
    type, support, availability, value ranges.
  - Product-level and product-wide facts come back in their own labelled groups (§6.2).
  - When nothing matches, the response says so and points to chunk retrieval for that
    vendor and aspect. The citation-checked answer path stays the fallback.

## 7. Evaluation

### 7.1 Findability against the comparison tables (local only)

A mapping file outside the repo (`~/Documents/Shared/projects/vendorkb-temp/coverage-map.yaml`,
never committed) lists each comparison row with its taxonomy ids. Claude writes and maintains it in
the main session. Subagents never read the internal tables and see only ids and gap lists. A script
in `eval/` takes the path as an argument and reports three things.

1. **Taxonomy coverage:** in-scope rows with no id. The target is 100%. Out of scope: pricing,
   pure verdicts, "competitor differentiators", documentation quality.
2. **Claim findability:** in-scope rows for which `capabilities` returns at least one fact for the
   table's vendor. This separates taxonomy gaps from extraction or corpus gaps.
3. **Chunk findability:** for rows with no claim, does hybrid retrieval return a relevant chunk in
   the top 10 for a question built from the row? Relevance is judged by Opus subagents, using only
   the question and the chunk.

The facts in the tables may be outdated, so this measures whether evidence for the aspect can be
found, never whether it agrees with the table.

**The extrapolation test:** `aspects(okta target)` must include every object-type aspect and every
applicable dimension that `aspects(entra id target)` has, except where the classes legitimately
differ.

### 7.2 Claim precision (v3)

- 100 random claims from about 300 fresh chunks.
- Graded by Fable subagents, not Opus, so the teacher does not grade itself.
- A claim is correct only if every field is right. Core-field and qualifier-field accuracy are
  reported too.
- Gate: ≥95% before the teacher capture.
- If it fails: up to 3 prompt revisions, each re-checked on fresh chunks, about $15 more at most.
  If the third still fails, the run stops and the decision goes to the user, with field-level
  numbers.

### 7.3 Existing gates

The retrieval and answer evals are unchanged. Capabilities in `answer()` (backlog item 8) waits for
v2, and is re-measured on it.

## 8. Order and cost

1. claims-v3 schema, prompt, storage and tests. Opus smoke on the same 20 chunks plus 10 chosen for
   qualifiers: about $0.50.
2. Taxonomy v2 seed, plus the taxonomy-coverage check iterated to 100% (free).
3. Precision check (§7.2): about $5, up to about $20 with revisions.
4. Teacher capture: about 10k stratified chunks through the Opus Batches API, capped at $100. The
   sample includes every document for the workloads the comparison tables cover, so findability
   can be measured early.
5. Alias clustering, then PR review of v2.
6. Mapper v2 and its labelled sample, `capabilities` v2, the two query calls, the findability
   report.
7. Fine-tune Qwen3.5-9B on the teacher set (backlog item 18). Findability is measured again after
   the student's full-corpus run.

## 9. Decisions made by Claude (check these)

1. The internal tables are read only in the main session. The coverage map lives outside the repo.
2. Coverage operations are dimensions (`coverage.*`), paired with an object type.
3. `target_scope` resolves to a child target, then a variant, then an object type.
4. Offerings and editions are normalised per vendor. Facts without an offering are product-level;
   facts without a target are product-wide. Neither is presented as more specific than its
   evidence.
5. Classes nest by dotted id, and `applies_to` matches descendants.
6. v1 is replaced in place. Nothing in production reads `capabilities`, so the v1 tables and
   loader are dropped.
7. The precision-gate fallback is 3 revisions and about $15, then back to the user.

## 10. Left to the builder

- Exact dimension ids and wording, as long as they are vendor-neutral and fit a family.
- How the prompt words the qualifier definitions. It is judged by the smoke and the precision
  check.
- The internal layout of the findability script.
- Whether `licensing` and `service` claims need their own extraction pass (only if the smoke shows
  that one prompt drops them).
