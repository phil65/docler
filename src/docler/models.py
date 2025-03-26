"""Data models for document representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    import numpy as np

    from docler.common_types import StrPath


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

    description: str | None = None
    """Description of the image."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata of the image."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    @classmethod
    async def from_file(
        cls,
        file_path: StrPath,
        image_id: str | None = None,
        description: str | None = None,
    ) -> Image:
        """Create an Image instance from a file.

        Args:
            file_path: Path to the image file
            image_id: Optional ID for the image (defaults to filename without extension)
            description: Optional description of the image

        Returns:
            Image instance with content loaded from the file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file type is not supported
        """
        import mimetypes

        import upath
        import upathtools

        path = upath.UPath(file_path)
        if not path.exists():
            msg = f"Image file not found: {file_path}"
            raise FileNotFoundError(msg)

        mime_type, _ = mimetypes.guess_type(str(path))
        if image_id is None:
            image_id = path.stem

        content = await upathtools.read_path(path, mode="rb")
        filename = path.name
        file_stats = path.stat()
        metadata = {
            "size_bytes": file_stats.st_size,
            "created_time": file_stats.st_ctime,
            "modified_time": file_stats.st_mtime,
            "source_path": str(path),
        }

        return cls(
            id=image_id,
            content=content,
            mime_type=mime_type or "image/jpeg",
            filename=filename,
            description=description,
            metadata=metadata,
        )

    @property
    def dimensions(self) -> tuple[int, int] | None:
        """Get the width and height of the image.

        Returns:
            A tuple of (width, height) if dimensions can be determined, None otherwise
        """
        try:
            from io import BytesIO

            from PIL import Image as PILImage

            # Convert content to bytes if it's a base64 string
            if isinstance(self.content, str):
                import base64

                # Handle data URLs
                if self.content.startswith("data:"):
                    # Extract the base64 part after the comma
                    base64_data = self.content.split(",", 1)[1]
                    image_data = base64.b64decode(base64_data)
                else:
                    # Regular base64 string
                    image_data = base64.b64decode(self.content)
            else:
                image_data = self.content

            # Open the image and get dimensions
            with PILImage.open(BytesIO(image_data)) as img:
                return (img.width, img.height)
        except (ImportError, Exception):
            return None


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

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata of the document."""

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
