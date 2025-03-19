"""Models for AI chunker."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    """A chunk of text with semantic metadata."""

    start_row: int
    """Start line number (1-based)"""

    end_row: int
    """End line number (1-based)"""

    keywords: list[str]
    """Key terms and concepts in this chunk"""

    references: list[int]
    """Line numbers that this chunk references or depends on"""

    model_config = ConfigDict(use_attribute_docstrings=True)


class Chunks(BaseModel):
    """Collection of chunks with their metadata."""

    chunks: list[Chunk]
    """A list of chunks to extract from the document."""
