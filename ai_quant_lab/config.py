"""Settings loaded from environment variables.

Single source of truth for runtime knobs. Modules import `settings` rather than
re-reading os.environ — keeps overrides predictable and testable.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _read_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Settings(BaseModel):
    """Runtime configuration. Loaded once at import time; override per-test via `Settings(...)`."""

    anthropic_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"),
        description="Optional. Examples 01-05 run without it.",
    )
    model: str = Field(
        default_factory=lambda: _read_env("AI_QUANT_LAB_MODEL", "claude-sonnet-4-6"),
    )

    # ----- Free LLM providers -----
    llm_provider: str = Field(
        default_factory=lambda: _read_env("AI_QUANT_LAB_LLM_PROVIDER", "anthropic"),
        description="anthropic | groq | ollama",
    )
    llm_fallback: str | None = Field(
        default_factory=lambda: os.environ.get("AI_QUANT_LAB_LLM_FALLBACK"),
        description="If primary fails, try this. e.g. 'ollama'.",
    )
    groq_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY"),
    )
    groq_model: str = Field(
        default_factory=lambda: _read_env("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )
    ollama_url: str = Field(
        default_factory=lambda: _read_env("OLLAMA_URL", "http://localhost:11434"),
    )
    ollama_model: str = Field(
        default_factory=lambda: _read_env("OLLAMA_MODEL", "qwen2.5-coder:7b"),
    )

    max_llm_calls: int = Field(
        default_factory=lambda: int(_read_env("AI_QUANT_LAB_MAX_LLM_CALLS", "200")),
        ge=1,
    )
    target_survivors: int = Field(
        default_factory=lambda: int(_read_env("AI_QUANT_LAB_TARGET_SURVIVORS", "3")),
        ge=1,
    )

    dsr_pvalue_max: float = Field(
        default_factory=lambda: float(_read_env("AI_QUANT_LAB_DSR_PVALUE_MAX", "0.05")),
        gt=0.0,
        lt=1.0,
    )
    max_correlation: float = Field(
        default_factory=lambda: float(_read_env("AI_QUANT_LAB_MAX_CORRELATION", "0.6")),
        ge=0.0,
        le=1.0,
    )

    cost_bps: float = Field(
        default_factory=lambda: float(_read_env("AI_QUANT_LAB_COST_BPS", "8.0")),
        ge=0.0,
    )
    annualization: int = Field(
        default_factory=lambda: int(_read_env("AI_QUANT_LAB_ANNUALIZATION", "252")),
        ge=1,
    )

    memory_db: Path = Field(
        default_factory=lambda: Path(_read_env("AI_QUANT_LAB_MEMORY_DB", "./memory.db")),
    )

    def require_api_key(self) -> str:
        """Used by agent modules. Raises with a helpful message if the key is missing."""
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Examples 06-08 and `python -m ai_quant_lab.run` need a key. "
                "Run `cp .env.example .env` and fill it in, or `export ANTHROPIC_API_KEY=...`."
            )
        return self.anthropic_api_key


settings = Settings()
