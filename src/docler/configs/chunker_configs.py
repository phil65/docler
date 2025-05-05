"""Configuration models for text chunking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field
from schemez import ModelIdentifier  # noqa: TC002

from docler.common_types import DEFAULT_CHUNKER_MODEL
from docler.provider import ProviderConfig


if TYPE_CHECKING:
    from docler.chunkers.base import TextChunker


DEFAULT_CHUNKER_SYSTEM_PROMPT = """
You are an expert at dividing text into meaningful chunks
while preserving context and relationships.

The task is to act like a chunker in an RAG pipeline.

Analyze the text and split it into coherent chunks.

All indexes are 1-based. Be accurate with line numbers.
Extract key terms and concepts as keywords
If any block is related to another block, you can add that info.
"""

DEFAULT_CHUNKER_USER_TEMPLATE = """
Here's the text with line numbers:

<text>
{numbered_text}
</text>
"""

ChunkerShorthand = Literal["markdown", "llamaindex", "ai"]


class BaseChunkerConfig(ProviderConfig):
    """Base configuration for text chunkers."""


class LlamaIndexChunkerConfig(BaseChunkerConfig):
    """Configuration for LlamaIndex chunkers."""

    type: Literal["llamaindex"] = Field(default="llamaindex", init=False)

    chunker_type: Literal["sentence", "token", "fixed", "markdown"] = "markdown"
    """Which LlamaIndex chunker to use."""

    chunk_overlap: int = Field(default=200, ge=0)
    """Number of characters to overlap between chunks."""

    chunk_size: int = Field(default=1000, ge=1)
    """Target size of chunks."""

    include_metadata: bool = True
    """Whether to include document metadata in chunks."""

    include_prev_next_rel: bool = False
    """Whether to track relationships between chunks."""

    def get_provider(self) -> TextChunker:
        """Get the chunker instance."""
        from docler.chunkers.llamaindex_chunker import LlamaIndexChunker

        return LlamaIndexChunker(**self.get_config_fields())


class MarkdownChunkerConfig(BaseChunkerConfig):
    """Configuration for markdown-based chunker."""

    type: Literal["markdown"] = Field(default="markdown", init=False)
    """Type discriminator for markdown chunker."""

    min_chunk_size: int = Field(default=200, ge=1)
    """Minimum characters per chunk."""

    max_chunk_size: int = Field(default=1500, ge=1)
    """Maximum characters per chunk."""

    chunk_overlap: int = Field(default=200, ge=0)
    """Number of characters to overlap between chunks."""

    def get_provider(self) -> TextChunker:
        """Get the chunker instance."""
        from docler.chunkers.markdown_chunker import MarkdownChunker

        return MarkdownChunker(**self.get_config_fields())


class AiChunkerConfig(BaseChunkerConfig):
    """Configuration for AI-based chunker."""

    type: Literal["ai"] = Field(default="ai", init=False)
    """Type discriminator for AI chunker."""

    model: ModelIdentifier = DEFAULT_CHUNKER_MODEL
    """LLM model to use for chunking."""

    system_prompt: str = DEFAULT_CHUNKER_SYSTEM_PROMPT
    """Custom prompt to override default chunk extraction prompt."""

    user_prompt: str = DEFAULT_CHUNKER_USER_TEMPLATE
    """Custom prompt to override default chunk extraction prompt."""

    def get_provider(self) -> TextChunker:
        """Get the chunker instance."""
        from docler.chunkers.ai_chunker import AIChunker

        return AIChunker(**self.get_config_fields())


class TokenAwareChunkerConfig(BaseChunkerConfig):
    """Configuration for token-aware chunker."""

    type: Literal["token_aware"] = Field(default="token_aware", init=False)
    """Type discriminator for token-aware chunker."""

    model: ModelIdentifier = "gpt-4"
    """Model ID to use for tokenization calculation."""

    max_tokens_per_chunk: int = Field(default=4000, ge=100)
    """Maximum tokens per chunk."""

    chunk_overlap_lines: int = Field(default=20, ge=0)
    """Number of lines to overlap between chunks."""

    def get_provider(self) -> TextChunker:
        """Get the chunker instance."""
        from docler.chunkers.token_chunker import TokenAwareChunker

        return TokenAwareChunker(**self.get_config_fields())


ChunkerConfig = Annotated[
    LlamaIndexChunkerConfig
    | MarkdownChunkerConfig
    | AiChunkerConfig
    | TokenAwareChunkerConfig,
    Field(discriminator="type"),
]
