"""Data models for document representation."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
