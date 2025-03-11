"""Configuration models for text chunking."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseChunkerConfig(BaseModel):
    """Base configuration for text chunkers."""

    type: str = Field(init=False)
    """Type identifier for the chunker."""

    chunk_overlap: int = 200
    """Number of characters to overlap between chunks."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class LlamaIndexChunkerConfig(BaseChunkerConfig):
    """Configuration for LlamaIndex chunkers."""

    type: Literal["llamaindex"] = Field(default="llamaindex", init=False)

    chunker_type: Literal["sentence", "token", "fixed", "markdown"] = "markdown"
    """Which LlamaIndex chunker to use."""

    chunk_size: int = 1000
    """Target size of chunks."""

    include_metadata: bool = True
    """Whether to include document metadata in chunks."""

    include_prev_next_rel: bool = False
    """Whether to track relationships between chunks."""


class MarkdownChunkerConfig(BaseChunkerConfig):
    """Configuration for markdown-based chunker."""

    type: Literal["markdown"] = Field(default="markdown", init=False)
    """Type discriminator for markdown chunker."""

    min_chunk_size: int = 200
    """Minimum characters per chunk."""

    max_chunk_size: int = 1500
    """Maximum characters per chunk."""

    max_markdown_header_level: int = Field(default=3, ge=1, le=6)
    """Maximum header level to use for chunking (1-6)."""


class AiChunkerConfig(BaseChunkerConfig):
    """Configuration for AI-based chunker."""

    type: Literal["ai"] = Field(default="ai", init=False)
    """Type discriminator for AI chunker."""

    model: str = "openrouter:google/gemini-2.0-flash-lite-001"
    """LLM model to use for chunking."""

    min_chunk_size: int = 200
    """Minimum characters per chunk."""

    max_chunk_size: int = 1500
    """Maximum characters per chunk."""

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    """Sampling temperature for LLM."""

    max_tokens: int | None = None
    """Maximum tokens in model response."""

    prompt: str | None = None
    """Custom prompt to override default chunk extraction prompt."""


ChunkerConfig = Annotated[
    LlamaIndexChunkerConfig | MarkdownChunkerConfig | AiChunkerConfig,
    Field(discriminator="type"),
]
