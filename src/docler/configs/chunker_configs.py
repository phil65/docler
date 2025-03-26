"""Configuration models for text chunking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

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


class BaseChunkerConfig(ProviderConfig):
    """Base configuration for text chunkers."""


class LlamaIndexChunkerConfig(BaseChunkerConfig):
    """Configuration for LlamaIndex chunkers."""

    type: Literal["llamaindex"] = Field(default="llamaindex", init=False)

    chunker_type: Literal["sentence", "token", "fixed", "markdown"] = "markdown"
    """Which LlamaIndex chunker to use."""

    chunk_overlap: int = 200
    """Number of characters to overlap between chunks."""

    chunk_size: int = 1000
    """Target size of chunks."""

    include_metadata: bool = True
    """Whether to include document metadata in chunks."""

    include_prev_next_rel: bool = False
    """Whether to track relationships between chunks."""

    def get_provider(self) -> TextChunker:
        """Get the chunker instance."""
        from docler.chunkers.llamaindex_chunker import LlamaIndexChunker

        return LlamaIndexChunker(
            chunker_type=self.chunker_type,
            chunk_size=self.chunk_size,
            include_metadata=self.include_metadata,
            include_prev_next_rel=self.include_prev_next_rel,
        )


class MarkdownChunkerConfig(BaseChunkerConfig):
    """Configuration for markdown-based chunker."""

    type: Literal["markdown"] = Field(default="markdown", init=False)
    """Type discriminator for markdown chunker."""

    min_chunk_size: int = 200
    """Minimum characters per chunk."""

    max_chunk_size: int = 1500
    """Maximum characters per chunk."""

    chunk_overlap: int = 200
    """Number of characters to overlap between chunks."""

    def get_provider(self) -> TextChunker:
        """Get the chunker instance."""
        from docler.chunkers.markdown_chunker import MarkdownChunker

        return MarkdownChunker(
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
        )


class AiChunkerConfig(BaseChunkerConfig):
    """Configuration for AI-based chunker."""

    type: Literal["ai"] = Field(default="ai", init=False)
    """Type discriminator for AI chunker."""

    model: str = DEFAULT_CHUNKER_MODEL
    """LLM model to use for chunking."""

    system_prompt: str = DEFAULT_CHUNKER_SYSTEM_PROMPT
    """Custom prompt to override default chunk extraction prompt."""

    user_prompt: str = DEFAULT_CHUNKER_USER_TEMPLATE
    """Custom prompt to override default chunk extraction prompt."""

    def get_provider(self) -> TextChunker:
        """Get the chunker instance."""
        from docler.chunkers.ai_chunker import AIChunker

        return AIChunker(
            model=self.model,
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
        )


ChunkerConfig = Annotated[
    LlamaIndexChunkerConfig | MarkdownChunkerConfig | AiChunkerConfig,
    Field(discriminator="type"),
]
