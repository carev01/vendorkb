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
    embedding vector({embed_dim}) NOT NULL,
    UNIQUE (doc_id, ordinal)
);

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

-- M3/M4: claim extraction and the capability table. A chunk is replaced (new chunk_id) when
-- its document is re-synced, so everything keyed on chunk_id cascades with it: the new
-- chunks are extracted afresh, and no claim outlives the evidence it quotes.
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
-- build``. Keyed on the claim, so a claim's deletion (a document re-sync) cascades here.
-- ``doc_version`` is the claim's ``documents.fetched_at``. ``scope``: 'product_wide' (no
-- target, and a workload dimension; an offering may be named), else 'offering' (an offering is
-- named), else 'product'. Rows sharing the conflict key (vendor, product, offering, edition,
-- dimension, target, variant, object type) but differing in support or in a stated value
-- share a ``conflict_group``; ``is_current`` marks the key's rows from its most recently
-- fetched document. No foreign keys into the
-- taxonomy tables: a ``taxonomy load`` replaces them, and stale rows wait for map and build.
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
    scope text NOT NULL CHECK (scope IN ('offering', 'product', 'product_wide')),
    conflict_group uuid,
    is_current boolean NOT NULL
);
CREATE INDEX IF NOT EXISTS capabilities_dimension_idx ON capabilities (dimension_id);
CREATE INDEX IF NOT EXISTS capabilities_target_idx ON capabilities (target_id);
CREATE INDEX IF NOT EXISTS capabilities_vendor_idx ON capabilities (lower(vendor));
CREATE INDEX IF NOT EXISTS capabilities_conflict_idx ON capabilities (conflict_group)
    WHERE conflict_group IS NOT NULL;
