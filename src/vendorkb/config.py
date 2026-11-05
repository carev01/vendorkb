from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class VendorkbSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    vendorkb_dsn: str = "postgresql://postgres:@127.0.0.1:55442/vendorkb"
    # voyage-4 since 2026-11-04: it beat jina-embeddings-v5-text-nano on the retrieval A/B
    # (gold article in the rerank pool for 95% of questions vs 78%;
    # docs/superpowers/voyage-4-embedding-ab-2026-11-04.md). Queries and chunks must share
    # one model and dimension: change them only with `reembed` + `promote-embedding`.
    embed_base_url: str = "https://api.voyageai.com/v1"
    embed_model: str = "voyage-4"
    embed_dim: int = 1024
    embed_max_batch: int = 128  # Voyage: <= 1,000 texts and 320K tokens per request
    embed_api_key: str = ""  # sent as a Bearer token when set (Voyage needs it)
    # Voyage options: input_type ("document" for chunks, "query" for questions) and
    # output_dimension. Turn both off for an OpenAI-compatible endpoint that rejects them
    # (the homelab TEI).
    embed_input_type: bool = True
    embed_output_dimension: int | None = 1024
    embed_concurrency: int = 4  # concurrent batches in `vendorkb reembed`
    rerank_base_url: str = ""
    rerank_model: str = "rerank-3"
    rerank_api_key: str = ""
    judge_base_url: str = ""
    judge_model: str = "z-ai/glm-5.3-flash"
    judge_api_key: str = ""
    answer_reasoning_effort: str = "low"
    docext_base_url: str = ""
    docext_read_key: str = ""
    docext_verify_tls: bool = True
    sync_page_size: int = 200  # feed records per embed call + transaction in `vendorkb sync`


@lru_cache
def get_settings() -> VendorkbSettings:
    return VendorkbSettings()
