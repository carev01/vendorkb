CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id uuid PRIMARY KEY,
    vendor text NOT NULL,
    product text,
    source text,
    title text,
    url text,
    content_hash text NOT NULL,
    fetched_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id uuid PRIMARY KEY,
    doc_id uuid NOT NULL REFERENCES documents ON DELETE CASCADE,
    section_path text[] NOT NULL,
    ordinal int NOT NULL,
    text text NOT NULL,
    token_count int NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding vector({embed_dim}) NOT NULL
);

-- 2026-11-08 (docs/superpowers/specs/2026-11-08-topic-expansion-design.md, Part A): a re-synced
-- page keeps each chunk whose text is unchanged, with its extraction and claims
-- (``store.write_document``). A chunk the new version drops is kept when one of its claims is
-- mapped, with ``superseded_at`` set (NULL = the current version), until the new version is
-- extracted and mapped (``store.retire_superseded``); it is kept for its claims only: search,
-- extraction and the refresh trigger skip it. So (doc_id, ordinal) is unique among the current
-- chunks only: the inline UNIQUE of the earlier schema (named chunks_doc_id_ordinal_key)
-- becomes a partial unique index. That constraint was also the only full index leading with
-- doc_id, which the cascade from documents needs (``DELETE FROM ONLY chunks WHERE $1 = doc_id``;
-- neither partial index can serve it: without chunks_doc_idx every page removal scans chunks),
-- so chunks_doc_idx replaces it for that. chunks_superseded_idx stays: ``retire_superseded``
-- scans the superseded chunks of every page, and the partial index keeps that small.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS superseded_at timestamptz;
ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_doc_id_ordinal_key;
CREATE UNIQUE INDEX IF NOT EXISTS chunks_doc_ordinal_current_idx ON chunks (doc_id, ordinal)
    WHERE superseded_at IS NULL;
CREATE INDEX IF NOT EXISTS chunks_superseded_idx ON chunks (doc_id)
    WHERE superseded_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
-- 2026-11-10 (BACKLOG 35): a column added to a filled table has no planner statistics until the
-- table is next analysed, and adding it changes no row, so autovacuum never does. Without them
-- ``superseded_at IS NULL`` (in every search) was estimated at 0.5% of the chunks, and the
-- planner sorted every chunk by distance instead of using the HNSW index: 2.2 s a vector
-- search, 35 s of a two-vendor topic question. One column: cheap, and enough.
ANALYZE chunks (superseded_at);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS documents_vendor_idx ON documents (vendor);
CREATE INDEX IF NOT EXISTS documents_product_idx ON documents (product);

-- Upgrades for databases created before ``sync`` (idempotent; after the CREATE TABLEs).
ALTER TABLE documents DROP COLUMN IF EXISTS graph_content_hash;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_id uuid;

CREATE TABLE IF NOT EXISTS sync_state (
    key text PRIMARY KEY,
    value text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sources (
    source_id uuid PRIMARY KEY,
    name text NOT NULL,
    refreshed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sync_failures (
    doc_id uuid PRIMARY KEY,
    reason text NOT NULL,
    failed_at timestamptz NOT NULL DEFAULT now()
);
-- The content hash a quarantined record failed with: a replay of that same content is
-- skipped without embedding. NULL for every other failure.
ALTER TABLE sync_failures ADD COLUMN IF NOT EXISTS content_hash text;

-- Article ids the current bootstrap has seen; at its end every other document is pruned.
CREATE TABLE IF NOT EXISTS sync_seen (
    doc_id uuid PRIMARY KEY
);

-- M3/M4: claim extraction and the capability table. A chunk whose text changed is replaced
-- (new chunk_id) when its document is re-synced, so everything keyed on chunk_id cascades with
-- it: the new chunks are extracted afresh, and no claim outlives the evidence it quotes. A
-- chunk with a mapped claim is kept, superseded, until the new version is mapped
-- (``chunks.superseded_at``). A chunk whose text matches (exactly or by ``chunker.reuse_key``)
-- keeps its row, extraction and claims, and takes the new version's text, embedding, ordinal
-- and section_path.
-- Reserved (BACKLOG 14): never written. DocExtractor's delta records carry no version (only
-- its products endpoint has one, per product), so documents.version and the claims'
-- version stay NULL, and answers come from the latest documents.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS version text;

-- One row per (chunk, model, prompt version): the idempotency key. ``status`` is 'done'
-- (claims written, possibly zero), 'errored' (retried by the next run) or 'failed' (errored
-- ``attempts`` times; skipped from then on). ``last_batch_id`` is the batch (or sync run)
-- whose result the row holds, so a resumed batch never applies a result twice. Its spend
-- is also in ``extract_batches``, which survives chunk deletion and is what the budget sums.
CREATE TABLE IF NOT EXISTS extractions (
    extraction_id uuid PRIMARY KEY,
    chunk_id uuid NOT NULL REFERENCES chunks ON DELETE CASCADE,
    extractor_model text NOT NULL,
    prompt_version text NOT NULL,
    status text NOT NULL,
    error text,
    attempts int NOT NULL DEFAULT 0,
    last_batch_id text,
    n_claims int NOT NULL DEFAULT 0,
    input_tokens int NOT NULL DEFAULT 0,
    output_tokens int NOT NULL DEFAULT 0,
    cost_usd double precision NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (chunk_id, extractor_model, prompt_version)
);
ALTER TABLE extractions ADD COLUMN IF NOT EXISTS attempts int NOT NULL DEFAULT 0;
ALTER TABLE extractions ADD COLUMN IF NOT EXISTS last_batch_id text;
-- The status check, added only when absent or from before 'failed' (never dropped and re-added
-- on every init_schema: that took ACCESS EXCLUSIVE and re-validated the table each sync run).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'extractions_status_check'
                     AND conrelid = 'extractions'::regclass
                     AND pg_get_constraintdef(oid) LIKE '%failed%') THEN
        ALTER TABLE extractions DROP CONSTRAINT IF EXISTS extractions_status_check;
        ALTER TABLE extractions ADD CONSTRAINT extractions_status_check
            CHECK (status IN ('done', 'errored', 'failed'));
    END IF;
END
$$;

-- Every paid submission: a Message Batch, or a sync-mode run ('sync-<uuid>'). No foreign key
-- to chunks: sync re-chunks documents while a batch is in flight, and spent money must not
-- vanish with them. The budget's spend so far is sum(coalesce(actual_cost_usd,
-- est_cost_usd)). ``status`` is 'submitted' until every result is stored, then 'processed';
-- a batch-mode run resumes any batch not yet processed before submitting new work.
CREATE TABLE IF NOT EXISTS extract_batches (
    batch_id text PRIMARY KEY,
    extractor_model text NOT NULL,
    prompt_version text NOT NULL,
    chunk_ids text[] NOT NULL,
    est_cost_usd double precision NOT NULL DEFAULT 0,
    actual_cost_usd double precision,
    status text NOT NULL CHECK (status IN ('submitted', 'processed')),
    submitted_at timestamptz NOT NULL DEFAULT now(),
    ended_at timestamptz
);

-- `taxonomy propose` spend, one row per run, updated after every call so a crash keeps it:
-- the cumulative cap (settings.propose_budget_usd) sums every run (BACKLOG 15, M7).
CREATE TABLE IF NOT EXISTS propose_runs (
    run_id text PRIMARY KEY,
    kind text NOT NULL,
    model text NOT NULL,
    spent_usd double precision NOT NULL DEFAULT 0,
    started_at timestamptz NOT NULL DEFAULT now(),
    ended_at timestamptz
);

-- vendor / product / version are copied from the chunk's document, never extracted.
CREATE TABLE IF NOT EXISTS raw_claims (
    claim_id uuid PRIMARY KEY,
    extraction_id uuid NOT NULL REFERENCES extractions ON DELETE CASCADE,
    chunk_id uuid NOT NULL REFERENCES chunks ON DELETE CASCADE,
    vendor text NOT NULL,
    product text,
    version text,
    capability_raw text NOT NULL,
    target_raw text,
    support text NOT NULL
        CHECK (support IN ('supported', 'partial', 'not_supported', 'unclear')),
    conditions text,
    evidence_quote text NOT NULL,  -- the matched chunk span, verbatim
    model_quote text,              -- the quote as the model wrote it
    gate text NOT NULL,
    extractor_model text NOT NULL,
    prompt_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS model_quote text;
CREATE INDEX IF NOT EXISTS raw_claims_chunk_idx ON raw_claims (chunk_id);
-- Indexes on the cascading foreign keys: a chunk deletion in sync cascades through
-- extractions -> raw_claims / rejected_claims, a seq scan per row without them.
CREATE INDEX IF NOT EXISTS raw_claims_extraction_idx ON raw_claims (extraction_id);
CREATE INDEX IF NOT EXISTS raw_claims_vendor_idx ON raw_claims (vendor);

-- Claims the evidence gate (or validation) refused, kept for analysis.
CREATE TABLE IF NOT EXISTS rejected_claims (
    claim_id uuid PRIMARY KEY,
    extraction_id uuid NOT NULL REFERENCES extractions ON DELETE CASCADE,
    chunk_id uuid NOT NULL REFERENCES chunks ON DELETE CASCADE,
    vendor text NOT NULL,
    product text,
    version text,
    capability_raw text,
    target_raw text,
    support text,
    conditions text,
    evidence_quote text,  -- NULL: a rejected quote matched no chunk span
    model_quote text,
    reason text NOT NULL,
    extractor_model text NOT NULL,
    prompt_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE rejected_claims ADD COLUMN IF NOT EXISTS model_quote text;
CREATE INDEX IF NOT EXISTS rejected_claims_extraction_idx ON rejected_claims (extraction_id);
CREATE INDEX IF NOT EXISTS rejected_claims_chunk_idx ON rejected_claims (chunk_id);

-- claims-v3 qualifiers (``claims.extract.CLAIM_SCHEMA``). Rows from earlier prompt versions
-- keep kind 'capability' and NULL for the rest. The checks mirror KIND_VALUES,
-- AVAILABILITY_VALUES, DEFAULT_STATE_VALUES and BOUND_VALUES; rejected_claims has the same
-- columns with no checks (a claim can be rejected for an invalid value).
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'capability';
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS target_scope text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS offering text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS edition text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS availability text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS default_state text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS value double precision;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS unit text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS bound text;
-- Each check added only when absent (never re-added on every init_schema).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'raw_claims_kind_check'
                     AND conrelid = 'raw_claims'::regclass) THEN
        ALTER TABLE raw_claims ADD CONSTRAINT raw_claims_kind_check
            CHECK (kind IN ('capability', 'licensing', 'service'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'raw_claims_availability_check'
                     AND conrelid = 'raw_claims'::regclass) THEN
        ALTER TABLE raw_claims ADD CONSTRAINT raw_claims_availability_check
            CHECK (availability IN ('generally_available', 'preview', 'early_access',
                                    'limited_availability', 'planned', 'deprecated',
                                    'unstated'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'raw_claims_default_state_check'
                     AND conrelid = 'raw_claims'::regclass) THEN
        ALTER TABLE raw_claims ADD CONSTRAINT raw_claims_default_state_check
            CHECK (default_state IN ('on', 'off'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'raw_claims_bound_check'
                     AND conrelid = 'raw_claims'::regclass) THEN
        ALTER TABLE raw_claims ADD CONSTRAINT raw_claims_bound_check
            CHECK (bound IN ('max', 'min', 'exact'));
    END IF;
END
$$;
ALTER TABLE rejected_claims ADD COLUMN IF NOT EXISTS kind text;
ALTER TABLE rejected_claims ADD COLUMN IF NOT EXISTS target_scope text;
ALTER TABLE rejected_claims ADD COLUMN IF NOT EXISTS offering text;
ALTER TABLE rejected_claims ADD COLUMN IF NOT EXISTS edition text;
ALTER TABLE rejected_claims ADD COLUMN IF NOT EXISTS availability text;
ALTER TABLE rejected_claims ADD COLUMN IF NOT EXISTS default_state text;
ALTER TABLE rejected_claims ADD COLUMN IF NOT EXISTS value double precision;
ALTER TABLE rejected_claims ADD COLUMN IF NOT EXISTS unit text;
ALTER TABLE rejected_claims ADD COLUMN IF NOT EXISTS bound text;

-- The loaded taxonomy's version (key 'taxonomy_version'): "2-<sha8 of the YAML>".
CREATE TABLE IF NOT EXISTS taxonomy_meta (
    key text PRIMARY KEY,
    value text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- Taxonomy v2 replaces v1 in place (``taxonomy.load``). One-time: a database that still has
-- a v1-only table shape (``capabilities.feature_id``, ``targets.kind``, or ``aliases`` without
-- ``scope``) loses the v1 capabilities, aliases and targets tables and its v1 taxonomy
-- version; later runs find none and do nothing. raw_claims keeps its rows and its v1 mapping
-- columns. ``features`` is deliberately NOT a signature: the v1 schema creates it on its own
-- (e.g. one sync run of a rolled-back image), and keying on it would drop the v2 ``targets``
-- here, which the v2 ``variants`` foreign key refuses, failing every later init_schema.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = current_schema()
                 AND ((table_name = 'capabilities' AND column_name = 'feature_id')
                      OR (table_name = 'targets' AND column_name = 'kind')))
       OR (to_regclass('aliases') IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                           WHERE table_schema = current_schema()
                             AND table_name = 'aliases' AND column_name = 'scope')) THEN
        DROP TABLE IF EXISTS capabilities;
        DROP TABLE IF EXISTS aliases;
        DROP TABLE IF EXISTS targets;
        DELETE FROM taxonomy_meta WHERE key = 'taxonomy_version';
    END IF;
END
$$;
-- ``features`` is a retired v1 table name: v2 never creates it, so drop it whenever present
-- (after the block above, which drops the v1 capabilities that referenced it).
DROP TABLE IF EXISTS features;

-- Taxonomy v2, loaded from taxonomy/taxonomy.yaml (the source of truth is git). A load
-- replaces every row. ``applies_to`` NULL means every class; an entry matches its class and
-- every class beneath it. Parents come from the dotted ids.
CREATE TABLE IF NOT EXISTS dimensions (
    dimension_id text PRIMARY KEY,
    family text NOT NULL,
    name text NOT NULL,
    definition text NOT NULL,
    applies_to text[],
    value_unit text,
    platform boolean NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS classes (
    class_id text PRIMARY KEY,
    parent_id text REFERENCES classes,
    name text NOT NULL,
    definition text NOT NULL
);

-- A class's own object types (inherited ones live on the ancestor); ``ordinal`` is YAML order.
CREATE TABLE IF NOT EXISTS object_types (
    class_id text NOT NULL REFERENCES classes,
    object_type_id text NOT NULL,
    name text NOT NULL,
    ordinal int NOT NULL,
    PRIMARY KEY (class_id, object_type_id)
);

CREATE TABLE IF NOT EXISTS targets (
    target_id text PRIMARY KEY,
    parent_id text REFERENCES targets,
    class_id text NOT NULL REFERENCES classes,
    name text NOT NULL
);

CREATE TABLE IF NOT EXISTS variants (
    target_id text NOT NULL REFERENCES targets,
    variant_id text NOT NULL,
    name text NOT NULL,
    PRIMARY KEY (target_id, variant_id)
);

CREATE TABLE IF NOT EXISTS offerings (
    offering_id text PRIMARY KEY,
    vendor text NOT NULL,
    name text NOT NULL
);

CREATE TABLE IF NOT EXISTS editions (
    offering_id text NOT NULL REFERENCES offerings,
    edition_id text NOT NULL,
    name text NOT NULL,
    PRIMARY KEY (offering_id, edition_id)
);

-- Normalized aliases (``cluster.normalize``), each entity's name among them. ``scope`` by
-- entity_type: dimension and target '', object_type the class_id, target_object_type,
-- variant the target_id, offering the vendor, edition the offering_id.
CREATE TABLE IF NOT EXISTS aliases (
    alias text NOT NULL,
    entity_type text NOT NULL CHECK (entity_type IN ('dimension', 'target', 'object_type',
                                                     'target_object_type', 'variant',
                                                     'offering', 'edition')),
    scope text NOT NULL DEFAULT '',
    entity_id text NOT NULL,
    PRIMARY KEY (alias, entity_type, scope)
);
-- 2026-11-08 (BACKLOG 31): false for a ``lookup_aliases`` entry, matched exactly like any alias
-- but never a candidate-search key of the taxonomy mapper, so adding one does not change the
-- candidates (and stored decisions) of other phrases.
ALTER TABLE aliases ADD COLUMN IF NOT EXISTS search boolean NOT NULL DEFAULT true;

-- Topics (2026-11-08), loaded with the taxonomy from its ``topics`` section: a curated topic
-- (a broad subject such as ransomware protection) with the phrases that name it, and its
-- aspects in YAML order (``ordinal``). An aspect's ``dimensions`` are dimension ids (the load
-- checks them; no foreign key, like ``dimensions.applies_to``), ``query`` is the phrasing
-- retrieval uses for it and ``cues`` are the words that mean a question names it. ``triggers``
-- and ``cues`` are stored normalized (lowercase, whitespace collapsed), hyphens as written. A
-- load replaces every row.
CREATE TABLE IF NOT EXISTS topics (
    topic_id text PRIMARY KEY,
    name text NOT NULL,
    triggers text[] NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_aspects (
    topic_id text NOT NULL REFERENCES topics ON DELETE CASCADE,
    aspect_id text NOT NULL,
    ordinal int NOT NULL,
    name text NOT NULL,
    query text NOT NULL,
    cues text[] NOT NULL,
    dimensions text[] NOT NULL,
    PRIMARY KEY (topic_id, aspect_id)
);

-- M4 mapping (taxonomy map, v1): each raw claim's feature and target under the taxonomy
-- version it was mapped with. ``*_method`` is 'alias' (exact normalized alias or name),
-- 'nn' (embedding nearest neighbour at or above the threshold) or 'unmapped'; a claim with
-- no target_raw has a NULL target_method. ``*_conf`` is the cosine (1.0 for an alias).
-- Kept, with their values, after the v1 tables are dropped.
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS feature_id text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS feature_method text
    CHECK (feature_method IN ('alias', 'nn', 'unmapped'));
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS feature_conf real;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS target_id text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS target_method text
    CHECK (target_method IN ('alias', 'nn', 'unmapped'));
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS target_conf real;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS mapped_version text;

-- Taxonomy v2 mapping (taxonomy map, ``taxonomy.mapper``), on v3 claims only: the dimension,
-- the target (target_id, kept from v1) with its class, variant and object type, and the
-- offering and edition, under ``mapped_version``. ``dimension_method`` and ``target_method``
-- are 'alias' (exact normalized alias), 'llm' (adjudicated among nearest candidates) or
-- 'unmapped'; target_method is NULL for a claim with no target text. 'nn' stays allowed in
-- target_method for v1 rows mapped before v2.
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS dimension_id text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS dimension_method text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS class_id text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS variant_id text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS object_type_id text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS offering_id text;
ALTER TABLE raw_claims ADD COLUMN IF NOT EXISTS edition_id text;
-- Each check added only when absent, or (target_method) when it predates 'llm'.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'raw_claims_dimension_method_check'
                     AND conrelid = 'raw_claims'::regclass) THEN
        ALTER TABLE raw_claims ADD CONSTRAINT raw_claims_dimension_method_check
            CHECK (dimension_method IN ('alias', 'llm', 'unmapped'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'raw_claims_target_method_check'
                     AND conrelid = 'raw_claims'::regclass
                     AND pg_get_constraintdef(oid) LIKE '%llm%') THEN
        ALTER TABLE raw_claims DROP CONSTRAINT IF EXISTS raw_claims_target_method_check;
        ALTER TABLE raw_claims ADD CONSTRAINT raw_claims_target_method_check
            CHECK (target_method IN ('alias', 'nn', 'llm', 'unmapped'));
    END IF;
END
$$;

-- capabilities v2 (``taxonomy.capabilities``): one row per claims-v3 claim mapped under the
-- current taxonomy version with a dimension, rebuilt from raw_claims by ``capabilities
-- build``. Keyed on the claim, so a claim's deletion (its page removed, or retired after a
-- re-sync) cascades here.
-- ``doc_version`` is the claim's ``documents.fetched_at``. ``scope``: 'product_wide' (no
-- target, and a workload dimension; an offering may be named), else 'offering' (an offering is
-- named), else 'product'. Rows sharing the conflict key (vendor, product, offering, edition,
-- dimension, target, variant, object type) but differing in support or in a stated value
-- share a ``conflict_group``; ``is_current`` marks the key's rows from its most recently
-- fetched document (a row on a superseded chunk ranks below every current-version row). No
-- foreign keys into the taxonomy tables: a ``taxonomy load`` replaces them, and stale rows
-- wait for map and build.
-- (No ``feature_id`` column: that is what the v1-drop block above keys on.)
CREATE TABLE IF NOT EXISTS capabilities (
    claim_id uuid PRIMARY KEY REFERENCES raw_claims ON DELETE CASCADE,
    vendor text NOT NULL,
    product text,
    offering_id text,
    edition_id text,
    dimension_id text NOT NULL,
    class_id text,
    target_id text,
    variant_id text,
    object_type_id text,
    kind text NOT NULL,
    support text NOT NULL,
    availability text,
    default_state text,
    value double precision,
    unit text,
    bound text,
    conditions text,
    doc_version timestamptz NOT NULL,
    taxonomy_version text NOT NULL,
    scope text NOT NULL CHECK (scope IN ('offering', 'product', 'product_wide', 'target_unmapped')),
    conflict_group uuid,
    is_current boolean NOT NULL
);
-- 2026-11-08: a claim whose target text named no taxonomy target has its own scope (it used to
-- count as product-wide). The table is derived: `vendorkb capabilities build` refills it.
ALTER TABLE capabilities DROP CONSTRAINT IF EXISTS capabilities_scope_check;
ALTER TABLE capabilities ADD CONSTRAINT capabilities_scope_check
    CHECK (scope IN ('offering', 'product', 'product_wide', 'target_unmapped'));
-- 2026-11-08: the claim's chunk is an earlier version of its page (``chunks.superseded_at``),
-- kept until the new version is extracted and mapped: the fact is shown as from that version.
ALTER TABLE capabilities ADD COLUMN IF NOT EXISTS evidence_superseded boolean NOT NULL DEFAULT false;
CREATE INDEX IF NOT EXISTS capabilities_dimension_idx ON capabilities (dimension_id);
CREATE INDEX IF NOT EXISTS capabilities_target_idx ON capabilities (target_id);
CREATE INDEX IF NOT EXISTS capabilities_vendor_idx ON capabilities (lower(vendor));
CREATE INDEX IF NOT EXISTS capabilities_conflict_idx ON capabilities (conflict_group)
    WHERE conflict_group IS NOT NULL;

-- The claim support check (claims.support, BACKLOG 36): does the claim's evidence quote show
-- that the vendor's product has the capability? ``supported`` and ``checker`` are the checker's
-- verdict and id (model@effort/prompt version); ``review_supported`` and ``reviewer`` the
-- reviewer's, NULL until a review. ``cost_usd`` sums both calls. ``capabilities build`` leaves
-- out only the claims the reviewer rejected.
CREATE TABLE IF NOT EXISTS claim_support (
    claim_id uuid PRIMARY KEY REFERENCES raw_claims ON DELETE CASCADE,
    supported boolean NOT NULL,
    checker text NOT NULL,
    checked_at timestamptz NOT NULL DEFAULT now(),
    review_supported boolean,
    reviewer text,
    reviewed_at timestamptz,
    cost_usd double precision NOT NULL DEFAULT 0
);

-- Taxonomy-map adjudication decisions (BACKLOG 23), reused whenever the same question comes up
-- again: the same prompt rules, model family, kind, phrase, workload class and candidates
-- (``key`` hashes them). A later run, a re-map after a taxonomy change, or a run on another
-- backend serving the same model pays only for questions it has not seen. ``source``: 'llm'
-- (decided by the model) or 'seed' (recorded from the mappings already stored).
CREATE TABLE IF NOT EXISTS map_decisions (
    key text PRIMARY KEY,
    kind text NOT NULL,
    phrase text NOT NULL,
    class_id text,
    candidates text[] NOT NULL,
    choice text,
    model text NOT NULL,
    source text NOT NULL CHECK (source IN ('llm', 'seed')),
    decided_at timestamptz NOT NULL DEFAULT now()
);
-- 2026-11-08: the prompt rules a decision was made under (mapper.PROMPT_RULES), so a question
-- whose candidates changed can carry an earlier answer made under the same rules only.
ALTER TABLE map_decisions ADD COLUMN IF NOT EXISTS rules text;
CREATE INDEX IF NOT EXISTS map_decisions_phrase_idx ON map_decisions (kind, phrase);

-- Embeddings of short strings (taxonomy keys, claim phrases) reused across taxonomy-map runs
-- (BACKLOG 27): a re-map embeds only strings it has never seen. ``vec`` is the provider's vector
-- as little-endian float32 bytes (bit-identical on reuse, so candidate lists, and the decisions
-- keyed on them, do not drift). Strings over 1,000 characters are never cached.
CREATE TABLE IF NOT EXISTS embedding_cache (
    model text NOT NULL,
    input_type text NOT NULL,
    text text NOT NULL,
    vec bytea NOT NULL,
    PRIMARY KEY (model, input_type, text)
);

-- 2026-11-12: vendors hidden from a database role (`vendorkb access restrict`). An MCP key can
-- be served through such a role (MCP_RESTRICTED_DSN, mcp_app): row-level security then hides
-- the vendor's documents, chunks, claims, facts and offerings from every query that role runs,
-- written or not yet written, so no forgotten WHERE can show them. The owner (a superuser)
-- bypasses it: sync, extraction and every other command see everything, unchanged. Vendors
-- compare case-insensitively. ``session_user``: inside a SECURITY DEFINER function
-- ``current_user`` is the definer.
CREATE TABLE IF NOT EXISTS access_hidden_vendors (
    role_name text NOT NULL,
    vendor text NOT NULL,
    PRIMARY KEY (role_name, vendor)
);
CREATE OR REPLACE FUNCTION vendorkb_hidden_vendors() RETURNS text[]
    LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$
    SELECT coalesce(array_agg(lower(vendor)), ARRAY[]::text[]) FROM access_hidden_vendors
    WHERE role_name = session_user $$;
-- pg_stats hides the statistics of a table under row-level security from a role it applies
-- to, and ``search.common_lexemes`` reads the chunks' lexeme frequencies there: corpus-wide
-- word frequencies, no text, so a restricted role reads them through this function.
CREATE OR REPLACE FUNCTION vendorkb_chunk_lexeme_freqs() RETURNS TABLE (e text, f real)
    LANGUAGE sql STABLE SECURITY DEFINER SET search_path FROM CURRENT AS $$
    SELECT unnest(most_common_elems::text::text[]),
           unnest(most_common_elem_freqs[1:cardinality(most_common_elems::text::text[])])
    FROM pg_stats WHERE schemaname = current_schema() AND tablename = 'chunks'
        AND attname = 'tsv' $$;
-- The policies apply to every role but the owner. ``CAST((SELECT …))``: a scalar subquery runs
-- the lookup once per query (bare, ``<> ALL (SELECT …)`` would compare with each row of it).
-- A chunk or edition is visible when its document or offering is (their own policy applies).
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS vendor_visible ON documents;
CREATE POLICY vendor_visible ON documents
    USING (lower(vendor) <> ALL (CAST((SELECT vendorkb_hidden_vendors()) AS text[])));
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS vendor_visible ON chunks;
CREATE POLICY vendor_visible ON chunks
    USING (EXISTS (SELECT 1 FROM documents d WHERE d.doc_id = chunks.doc_id));
ALTER TABLE raw_claims ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS vendor_visible ON raw_claims;
CREATE POLICY vendor_visible ON raw_claims
    USING (lower(vendor) <> ALL (CAST((SELECT vendorkb_hidden_vendors()) AS text[])));
ALTER TABLE capabilities ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS vendor_visible ON capabilities;
CREATE POLICY vendor_visible ON capabilities
    USING (lower(vendor) <> ALL (CAST((SELECT vendorkb_hidden_vendors()) AS text[])));
ALTER TABLE offerings ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS vendor_visible ON offerings;
CREATE POLICY vendor_visible ON offerings
    USING (lower(vendor) <> ALL (CAST((SELECT vendorkb_hidden_vendors()) AS text[])));
ALTER TABLE editions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS vendor_visible ON editions;
CREATE POLICY vendor_visible ON editions
    USING (EXISTS (SELECT 1 FROM offerings o WHERE o.offering_id = editions.offering_id));
-- an offering alias is scoped by its vendor, an edition alias by its offering
ALTER TABLE aliases ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS vendor_visible ON aliases;
CREATE POLICY vendor_visible ON aliases
    USING (CASE entity_type
           WHEN 'offering' THEN lower(scope) <> ALL (CAST((SELECT vendorkb_hidden_vendors()) AS text[]))
           WHEN 'edition' THEN EXISTS (SELECT 1 FROM offerings o WHERE o.offering_id = aliases.scope)
           ELSE true END);
