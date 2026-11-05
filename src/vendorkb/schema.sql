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
