"""Data models for document representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    import numpy as np


ImageReferenceFormat = Literal["inline_base64", "file_paths", "keep_internal"]


class Image(BaseModel):
    """Represents an image within a document."""

    id: str
    """Internal reference id used in markdown content."""

    content: bytes | str = Field(repr=False)
    """Raw image bytes or base64 encoded string."""

    mime_type: str
    """MIME type of the image (e.g. 'image/jpeg', 'image/png')."""

    filename: str | None = None
    """Optional original filename of the image."""

    model_config = ConfigDict(use_attribute_docstrings=True)


class Document(BaseModel):
    """Represents a processed document with its content and metadata."""

    content: str
    """Markdown formatted content with internal image references."""

    images: list[Image] = Field(default_factory=list)
    """List of images referenced in the content."""

    title: str | None = None
    """Document title if available."""

    author: str | None = None
    """Document author if available."""

    created: datetime | None = None
    """Document creation timestamp if available."""

    modified: datetime | None = None
    """Document last modification timestamp if available."""

    source_path: str | None = None
    """Original source path of the document if available."""

    mime_type: str | None = None
    """MIME type of the source document if available."""

    page_count: int | None = None
    """Number of pages in the source document if available."""

    model_config = ConfigDict(use_attribute_docstrings=True)


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
class TextChunk:
    """Chunk of text with associated metadata and images."""

    text: str
    source_doc_id: str
    chunk_index: int
    page_number: int | None = None
    images: list[Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


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
