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
ALTER TABLE extractions DROP CONSTRAINT IF EXISTS extractions_status_check;
ALTER TABLE extractions ADD CONSTRAINT extractions_status_check
    CHECK (status IN ('done', 'errored', 'failed'));

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

-- Taxonomy, loaded from taxonomy/taxonomy.yaml (the source of truth is git).
CREATE TABLE IF NOT EXISTS features (
    feature_id text PRIMARY KEY,
    name text NOT NULL,
    category text,
    definition text,
    parent_id text REFERENCES features
);

CREATE TABLE IF NOT EXISTS targets (
    target_id text PRIMARY KEY,
    name text NOT NULL,
    kind text,  -- workload | platform | storage
    parent_id text REFERENCES targets
);

CREATE TABLE IF NOT EXISTS aliases (
    alias text NOT NULL,
    entity_type text NOT NULL CHECK (entity_type IN ('feature', 'target')),
    entity_id text NOT NULL,
    PRIMARY KEY (alias, entity_type)
);

-- Normalized capability facts, rebuilt from raw_claims and the current taxonomy version.
CREATE TABLE IF NOT EXISTS capabilities (
    cap_id uuid PRIMARY KEY,
    vendor text NOT NULL,
    product text,
    version text,
    feature_id text NOT NULL REFERENCES features,
    target_id text REFERENCES targets,
    support text NOT NULL,
    conditions text,
    claim_id uuid NOT NULL REFERENCES raw_claims ON DELETE CASCADE,
    mapping_conf real,
    taxonomy_version text NOT NULL
);
CREATE INDEX IF NOT EXISTS capabilities_feature_idx ON capabilities (feature_id);
CREATE INDEX IF NOT EXISTS capabilities_target_idx ON capabilities (target_id);
CREATE INDEX IF NOT EXISTS capabilities_vendor_idx ON capabilities (vendor);
