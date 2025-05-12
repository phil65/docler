"""Data models for document representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from mkdown import Document, TextChunk
from pydantic import Field


if TYPE_CHECKING:
    import numpy as np


ImageReferenceFormat = Literal["inline_base64", "file_paths", "keep_internal"]


class ChunkedDocument(Document):
    """Document with derived chunks.

    Extends the Document model to include chunks derived from the original content.
    """

    chunks: list[TextChunk] = Field(default_factory=list)
    """List of chunks derived from this document."""

    @classmethod
    def from_document(
        cls, document: Document, chunks: list[TextChunk]
    ) -> ChunkedDocument:
        """Create a ChunkedDocument from an existing Document and its chunks.

        Args:
            document: The source document
            chunks: List of chunks derived from the document
        """
        return cls(**document.model_dump(), chunks=chunks)


@dataclass
class VectorStoreInfo:
    """A single vector search result."""

    db_id: str
    name: str
    created_at: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single vector search result."""

    chunk_id: str
    score: float  # similarity score between 0-1
    metadata: dict[str, Any]
    text: str | None = None


@dataclass
class Vector:
    """A single vector."""

    id: str
    data: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
