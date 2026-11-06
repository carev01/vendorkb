# Taxonomy v2 and claims v3: built for vendor comparisons

Status: draft for review, 2026-11-06.
Supersedes: the taxonomy shape in DESIGN.md §5.4, taxonomy v1 (PR #1), and the `claims-v2` prompt
(never run beyond a 20-chunk smoke).

## 1. Why

The system exists to feed vendor comparison tables. The user shared a set of internal comparison
tables (mostly Veeam, one Commvault). Each row is one dimension: a short verdict, a red, amber or
green rating, and notes holding the facts and their nuances. Their features, level of detail and
nuances must be in the taxonomy. They must also extrapolate to workloads that do not exist yet.
For example, when Okta protection ships, it should get a comparison with the same coverage as
Entra ID, without anyone designing it from scratch.

Taxonomy v1 cannot do that:
- It grew bottom-up from Haiku claims as a flat list of 2,071 features.
- It covers many technical features, but has nothing for licensing, support, certifications,
  service-level versus storage-level immutability, and similar.
- It has no structure that carries a template to a new workload.

The claim format cannot hold the nuances either: a capability, a target, a 4-value support and a
free-text conditions field. The tables depend on qualifiers such as:
- edition;
- offering variant;
- "requires another product";
- availability stage;
- default state;
- quantities with units.

**The comparison tables are internal, and this repository is public.** Nothing from them is
committed: no files, no quotes, no row lists. Templates in this repo are written in our own words.
The coverage check (§7) reads them from outside the repo.

## 2. Goals, non-goals, anti-goals

**Goals**
1. Every comparison row that documentation can answer maps to a taxonomy id, and claims can fill
   it.
2. A new workload of a known kind inherits its class's template, so it gets a full set of rows on
   day one.
3. Qualifiers are structured, so cross-vendor questions can filter and compare. Examples: "minimum
   RPO", "which vendors need a second product for S3", "what is preview-only".
4. The Opus teacher set is labelled once, in the final format, so the 9B student learns that format.

**Non-goals**
- **Pricing.** It comes from other sources and is rarely in documentation. Licensing metrics
  (per user, per instance, per node, what each edition includes, free tiers) are in scope.
- **Verdicts and ratings.** These are judgements made across vendors, so the system extracts the
  facts and a rating layer comes later.
- **Known issues, bugs and procedures.** They stay excluded, as in v2.
- **A comparison-table generator UI.** §6.3 defines the query it would use.

**Anti-goals**
- A taxonomy that names a vendor's product as a dimension. Dimensions are vendor-neutral, and
  products are targets or offerings.
- Inventing coverage. An empty cell means "no documented evidence found", never "not supported".

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
| offering | string or null | The vendor's offering or deployment variant the text names, when narrower than the Product header: "Veeam Data Cloud", "self-managed", "Grid appliance", "Express". |
| edition | string or null | The edition or license tier the text attaches: "Premium", "Advanced and above". |
| requires | string or null | Another product, component, plug-in or license the capability depends on: "Veeam Backup & Replication", "Kasten", "a plug-in". Ordinary prerequisites (permissions, enabling an API) stay in conditions. |
| availability | enum | `generally_available`, `preview`, `early_access`, `limited_availability`, `planned`, `deprecated`, or `unstated`. Only what the text says: a plain present-tense statement is `unstated`, not `generally_available`. |
| default_state | enum or null | `on` or `off` when the text says so, else null. |
| value | number or null | The quantity, for a limit or a measure: retention 99, RPO 10, regions 23, records 500000, nines 12. |
| unit | string or null | The unit of `value`: "years", "minutes", "regions", "records per job", "attributes", "hours". |
| bound | enum or null | `max`, `min` or `exact` for `value`. "Up to 99 years" is max. "As often as every 15 minutes" is min (the smallest interval). |
| conditions | string or null | Everything else the text attaches: platforms ("VMware and Hyper-V only"), prerequisites, the reason for a "not supported". |
| evidence_quote | string | As today. |

The JSON schema is strict, with every field required and nullable where shown, like
`CLAIM_SCHEMA`.

### 3.2 Prompt changes from v2

- Licensing and support facts are claims (`kind`), no longer excluded.
- Coverage claims are one per object type, so a sentence that lists six object types gives six
  claims sharing one quote.
- Each qualifier gets a short definition with one counter-example, like the v2 "partial" rule.
- Pricing stays excluded: a price is not a claim.
- Still excluded: procedures, actions in other systems, known issues, documentation
  announcements, and component version tables.

### 3.3 Storage

- New nullable columns on `raw_claims` for each new field. `kind` defaults to `capability` for
  existing rows.
- The gate and the regate are unchanged.
- v1 claims stay as they are, marked by `prompt_version`. Taxonomy v2 builds only from v3 claims.

## 4. Taxonomy v2

One YAML file, `taxonomy/taxonomy.yaml` with `version: 2`, in four sections. It is reviewed as a
git PR, as before.

### 4.1 Dimensions (vendor-neutral, cross-cutting)

A dimension is one comparable aspect, for example `recovery.cross_tenant` or
`retention.maximum_period`. It has:
- `id`, prefixed by its family;
- `name`, `definition` and `aliases`;
- `applies_to`: `all`, or a list of classes;
- optionally `value_unit`, when the dimension is a quantity. The rating layer and the queries can
  then compare values: `retention.maximum_period` is measured in years, `schedule.minimum_rpo` in
  minutes.

Families (the seed list; names are our own):

| Family | Covers |
|---|---|
| coverage | Which object types of a target are protected, by which operation: backup, restore, export, compare. Handled through object types (§4.2), not one dimension per object. |
| protection | Snapshot, image-level, cloud-native service backup, agent, application consistency, changed block tracking, log backup, backup copy, replication and CDP. |
| schedule | Minimum RPO, schedule types, per-object or per-group policies, SLA-based policies. |
| retention | Maximum period, granularity, per-object retention, GFS, legal hold. |
| storage | Deployment model (SaaS, self-managed, MSP), bring-your-own storage, cloud choice, region count and choice, changing location after setup, redundancy, tiers and tiering, data reduction, repository types, tape, durability and availability commitments, sovereign or government clouds. |
| recovery | Granularity (tenant down to attribute or field), destinations (original, alternate, cross-account, cross-region, cross-tenant, sandbox, cross-platform), instant recovery, bulk and its limits, relationship and hierarchy restore, automation bypass, export formats, preview and compare, search scope and filters, self-service portal. |
| security | Immutability (service or storage level, compliance mode, what is excluded), encryption (algorithm, in transit, customer-managed keys and key services, field level, default), access control (built-in and custom roles, scoping, MFA, SSO and identity providers, IP restrictions, multi-user authorization and its scope), audit logs, platform hardening. |
| cyber | Detection methods (entropy, signatures, rules, indicators of compromise, mass changes), third-party scanners, incident intake API, quarantine and containment, clean room, scan on restore, recoverability testing, recovery orchestration, response services. |
| operations | Dashboards, reports, notifications (events and channels), multi-tenant and central management, APIs and automation, AI assistance (modes, limits, MCP), private networking, throttling against SaaS API limits, deployment and upgrade. |
| data_use | Archiving, sandbox seeding, masking, classification, ROT clean-up, per-person privacy deletion, data integration or reuse, migration. |
| vendor | Licensing metric, edition inclusion, free tier, license portability, exit strategy (export, free restore tool), support terms, certifications, release cadence, marketplaces, appliance procurement, installation and hardware support. |

### 4.2 Classes (workload kinds) with object-type templates

A class is a kind of workload. It lists its **object types**, and any **class dimensions** that
only make sense for it, such as password-reset options on restore for identity, or Salesforce-style
automation bypass for CRM.

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

Okta maps "sign-on policies" to `access_policy` and "System Log" to `audit_log`, so Okta inherits
the same rows as Entra ID.

### 4.3 Targets (instances of a class)

A target is a concrete workload. It has:
- `id`, `name` and `aliases`;
- `class`;
- optional `parent`, for hierarchies such as `aws` → `aws.rds` → `aws.rds.oracle`;
- `object_aliases`, mapping the target's own vocabulary to class object types;
- optional `variants`, for named sub-kinds that are not object types: Salesforce clouds and org
  types, FSx file-system types.

Adding a workload means adding a target entry. Its rows come from its class.

### 4.4 Platform scope

Some dimensions describe the backup product itself, not a workload: hardening, AI assistance,
support, certifications, licensing. Claims for them have a null target. They map with
`target_id = null`, and appear in every comparison for that vendor's offering.

## 5. Building v2

1. **Seed, top-down.** Claude writes the families, dimensions, classes, object types and the
   targets of the 43 corpus vendors' main workloads. The sources are the comparison-table
   structure, domain knowledge and the v1 taxonomy, all in our own words.
2. **Coverage check against the comparison tables (§7.1),** iterated until every answerable row
   maps.
3. **Aliases, bottom-up.** Cluster the distinct v3 claim strings, as in v1. Propose aliases for
   existing ids, and new ids only where no existing one fits.
4. **Human review as a PR.** PR #1 (v1) is closed as superseded. Its branch is kept as an alias
   source.

## 6. Mapping, capabilities, queries

### 6.1 Mapper

For each v3 claim:
- **Target:** alias match. When that fails, nearest neighbour gives up to 5 candidates for the
  LLM.
- **Object type:** `target_scope` through the target's `object_aliases`, then the class aliases.
- **Dimension:** alias match, then LLM adjudication over the top 5 candidates, with "none" allowed.
  Nearest neighbour alone never maps: the v1 calibration showed 15–60% precision.

The adjudication model is a setting, evaluated on a labelled sample before use. That sample should
be about 300 claims, labelled by Opus subagents and checked by the user. Anything left unmapped is
kept and surfaced for the next review.

### 6.2 `capabilities` v2

One row per mapped claim, with these columns: `vendor`, `product`, `offering`, `edition`,
`dimension_id`, `class_id`, `target_id`, `object_type_id`, `support`, `availability`,
`default_state`, `requires`, `value`, `unit`, `bound`, `conditions`, `claim_id`, `doc_version`,
`taxonomy_version`.

Conflicts are handled per DESIGN.md §5.5: the latest document version wins, and the conflict is
recorded.

### 6.3 The comparison query (interface, not UI)

`capabilities compare --target T [--vendors ...]` returns the rows of T's class: its object types
times the coverage operations, then every applicable dimension. Each row carries one cell per
vendor offering, holding the facts. Empty cells say "no documented evidence".

## 7. Evaluation

### 7.1 Coverage of the comparison tables (local only)

A mapping file outside the repo (`~/Documents/Shared/projects/vendorkb-temp/coverage-map.yaml`,
never committed) lists each comparison row with its taxonomy ids. A script in `eval/` takes that
path as an argument and reports:
- **Taxonomy coverage:** rows with no id. Target 100%, after removing out-of-scope rows (pricing,
  pure verdicts, "competitor differentiators", documentation quality).
- **Fill rate:** rows whose cell is empty for the table's vendor. This separates taxonomy gaps
  from extraction or corpus gaps.
- **Extrapolation:** for Okta, a test that rows generated from `saas.identity` include every
  Entra ID object-type row and every applicable dimension.

The facts in the tables may be outdated, so this measures coverage, never correctness.

### 7.2 Claim precision (v3)

- 100 random claims from about 300 fresh chunks.
- Graded by Fable subagents, not Opus, so the teacher does not grade itself.
- A claim is correct only if every field is right. Field-level accuracy is reported too.
- Gate: ≥95% before the teacher capture.

### 7.3 Existing gates

The retrieval and answer evals are unchanged. Capabilities in `answer()` (backlog item 8) waits for
v2.

## 8. Order and cost

1. claims-v3 schema, prompt, storage and tests. Opus smoke on the same 20 chunks plus 10 chosen for
   qualifiers: about $0.50.
2. Taxonomy v2 seed, plus the coverage check iterated to 100% (free).
3. Precision check (§7.2): about $5.
4. Teacher capture: about 10k stratified chunks through the Opus Batches API, about $80–100,
   capped at $100.
5. Alias clustering, then PR review of v2.
6. Mapper v2 and its labelled sample, `capabilities` v2, the compare query, fill-rate report.
7. Fine-tune Qwen3.5-9B on the teacher set (backlog item 18).

Steps 1–3 need no decisions beyond this spec. Step 4 spends the approved teacher budget only after
§7.2 passes.

## 9. Decisions settled after the builder check

Each is my call, unless noted, made where the first draft left a builder guessing.

1. **The coverage map** (§7.1) is written and maintained by Claude in the main session, which may
   read the internal tables locally. Subagents never read them. They receive only taxonomy ids and
   gap lists, which contain no table content. Format: one entry per row, with `table`, `row`
   (the row label), `ids` (a list of taxonomy ids) and `out_of_scope` (a reason or null).
2. **Coverage operations are dimensions:** `coverage.backup`, `coverage.restore`,
   `coverage.export`, `coverage.compare`. A coverage fact is a `capabilities` row with one of these
   `dimension_id`s plus an `object_type_id`. The mapper reads the operation from `capability`,
   through these dimensions' aliases.
3. **Where a `target_scope` goes, in order:** a child target, then a variant, then an object type.
   "RDS for Oracle" becomes target `aws.rds.oracle`; "FSx for Lustre" becomes a variant of
   `aws.fsx`; "users" becomes object type `user`. `capabilities` gets a `variant_id` column.
4. **Offerings and editions are normalised** per vendor in a fifth YAML section, `offerings:`
   (vendor, id, name, aliases, editions with aliases). The claim's free text maps to them.
   - A claim with no offering fills a **product-level** cell. It is never copied into a specific
     offering's cell, since that would invent coverage.
   - The compare query shows offering cells only where offering-specific claims exist, next to
     the product cell.
   - The conflict key is (vendor, product, offering, edition, dimension, target, variant,
     object type).
5. **Fill rate is measured on the teacher sample first.** The teacher set is stratified so that
   it includes all documents of the table vendors' compared workloads. The final fill rate comes
   after the student's full-corpus run.
6. **Claims with no target describing a workload dimension** ("retention up to 99 years" with no
   target) fill a **product-wide** row in every comparison for that vendor, labelled as
   product-wide. They are never copied into a target's cell.
7. **If the precision gate fails:** up to 3 prompt revisions, each re-checked on fresh chunks.
   That is about $15 more at most, within the approved budget. All-fields-correct stays the
   gate; core fields and qualifier fields are also reported separately.
   - If the third revision still fails, the run stops and the decision goes back to the user,
     with the field-level numbers.
   - **The user may want to set this budget or bar differently.**
8. **Classes nest by dotted id.** `kubernetes.vm` inherits the object types and dimensions of
   `kubernetes`, and adds its own. In `applies_to`, an entry matches its class and every class
   beneath it, so `saas` covers every `saas.*` class.
9. **v1 is replaced in place.** Nothing in production reads `capabilities`, and `answer()` does
   not use it. The v1 tables and loader are dropped, and backlog item 8 is re-measured on v2. The
   v1 branch stays as an alias source only.

## 10. Left to the builder


- Exact dimension ids and wording, as long as they are vendor-neutral and fit a family.
- How the prompt words the qualifier definitions. It is judged by the smoke and the precision
  check.
- The internal layout of the coverage-check script.
- Whether `licensing` and `service` claims need their own extraction pass (only if the smoke shows
  that one prompt drops them).
