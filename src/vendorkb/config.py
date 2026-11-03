from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class VendorkbSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    vendorkb_dsn: str = "postgresql://postgres:@127.0.0.1:55432/vendorkb"
    embed_base_url: str = "http://srv-llm.home.lan:8082/v1"
    embed_model: str = "jinaai/jina-embeddings-v5-text-nano-retrieval"
    embed_dim: int = 768
    embed_max_batch: int = 32
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


@lru_cache
def get_settings() -> VendorkbSettings:
    return VendorkbSettings()
