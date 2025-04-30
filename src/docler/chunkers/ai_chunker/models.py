"""Models for AI chunker."""

from __future__ import annotations

from schemez import Schema


class Chunk(Schema):
    """A chunk of text with semantic metadata."""

    start_row: int
    """Start line number (1-based)"""

    end_row: int
    """End line number (1-based)"""

    keywords: list[str]
    """Key terms and concepts in this chunk"""

    references: list[int]
    """Line numbers that this chunk references or depends on"""


class Chunks(Schema):
    """Collection of chunks with their metadata."""

    chunks: list[Chunk]
    """A list of chunks to extract from the document."""
