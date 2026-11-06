from __future__ import annotations

from functools import lru_cache
from typing import Literal

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
    # Claim extraction (M3): Claude Haiku 5.5 via the Anthropic SDK. The key falls back to
    # the ANTHROPIC_API_KEY environment variable when empty. The budget is a hard cap on the
    # estimated spend recorded in `extractions` (all runs together). The prompt version is
    # not a setting: it is `claims.extract.PROMPT_VERSION`, bumped with the prompt in code.
    anthropic_api_key: str = ""
    extract_model: str = "claude-haiku-5-5"
    extract_budget_usd: float = 25.0
    # "openai": any OpenAI-compatible endpoint (e.g. OpenRouter, extract_model
    # "anthropic/claude-haiku-5.5:batch"), sync mode only, extract_concurrency calls at once.
    extract_backend: Literal["anthropic", "openai"] = "anthropic"
    extract_base_url: str = ""
    extract_api_key: str = ""
    extract_concurrency: int = 16
    # max_tokens per extraction request (both backends; thinking counts against it). Chunks
    # cut off at it are retried with `vendorkb extract --retry-errored` after raising it.
    extract_max_tokens: int = 2000
    # Claude's effort ("low" .. "max"); Opus 5.5 always thinks, so effort is its only control.
    extract_effort: str = "low"
    # Price override (USD per million tokens) for a model not in extract.PRICES_PER_MTOK;
    # both must be set (> 0) to take effect.
    extract_price_in_per_mtok: float = 0.0
    extract_price_out_per_mtok: float = 0.0


@lru_cache
def get_settings() -> VendorkbSettings:
    return VendorkbSettings()
