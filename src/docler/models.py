"""Data models for document representation."""

from __future__ import annotations

import base64
import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
import mimetypes
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Base64Str, Field
from schemez import MimeType, Schema
import upath
import upathtools


if TYPE_CHECKING:
    import numpy as np

    from docler.common_types import StrPath


ImageReferenceFormat = Literal["inline_base64", "file_paths", "keep_internal"]


class Image(Schema):
    """Represents an image within a document."""

    id: str
    """Internal reference id used in markdown content."""

    content: bytes | Base64Str = Field(repr=False)
    """Raw image bytes or base64 encoded string."""

    mime_type: MimeType
    """MIME type of the image (e.g. 'image/jpeg', 'image/png')."""

    filename: str | None = None
    """Optional original filename of the image."""

    description: str | None = None
    """Description of the image."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata of the image."""

    def to_base64(self) -> str:
        """Convert image content to base64 string.

        Returns:
            Base64 encoded string of the image content.
        """
        if isinstance(self.content, bytes):
            return base64.b64encode(self.content).decode()
        # Handle data URL format (e.g., "data:image/jpeg;base64,...")
        if isinstance(self.content, str) and self.content.startswith("data:"):
            return self.content.split(",", 1)[1]
        # Already a base64 string
        return self.content

    def to_base64_url(self) -> str:
        """Convert image content to base64 data URL.

        Args:
            data: Raw bytes or base64 string of image data
            mime_type: MIME type of the image

        Returns:
            Data URL format of the image for embedding in HTML/Markdown
        """
        b64_content = self.to_base64()
        return f"data:{self.mime_type};base64,{b64_content}"

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
            from PIL import Image as PILImage

            if isinstance(self.content, str):
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


class Document(Schema):
    """Represents a processed document with its content and metadata."""

    @classmethod
    async def from_file(cls, file_path: StrPath, *, load_images: bool = True) -> Document:
        """Load a Document from a markdown file, parsing embedded images and metadata.

        Args:
            file_path: Path to the markdown file.
            load_images: Whether to parse and load images (inline base64 or file paths).

        Returns:
            Document instance reconstructed from the markdown file.
        """
        import yaml

        path = upath.UPath(file_path)
        text = path.read_text(encoding="utf-8")

        # Parse frontmatter if present
        frontmatter = {}
        content = text
        if text.startswith("---"):
            fm_end = text.find("---", 3)
            if fm_end != -1:
                fm_block = text[3:fm_end]
                try:
                    frontmatter = yaml.safe_load(fm_block)
                except Exception:  # noqa: BLE001
                    frontmatter = {}
                content = text[fm_end + 3 :].lstrip("\n")

        # Find all image references: ![id](url)
        image_pattern = re.compile(r"!\[([^\]]+)\]\(([^)]+)\)")
        images: list[Image] = []
        image_ids_seen: set[str] = set()

        def _parse_image_ref(match):
            img_id = match.group(1)
            img_url = match.group(2)
            if img_id in image_ids_seen:
                return
            image_ids_seen.add(img_id)
            if img_url.startswith("data:"):
                # Inline base64 image
                mime_type = img_url.split(";")[0][5:]
                b64_data = img_url.split(",", 1)[1]
                images.append(
                    Image(
                        id=img_id,
                        content=b64_data,
                        mime_type=mime_type,
                        filename=None,
                        description=None,
                        metadata={},
                    )
                )
            elif load_images:
                # File path reference, try to load file if possible
                img_path = path.parent / img_url
                if img_path.exists():
                    mime_type = None
                    with contextlib.suppress(Exception):
                        mime_type, _ = mimetypes.guess_type(str(img_path))
                    content_bytes = img_path.read_bytes()
                    images.append(
                        Image(
                            id=img_id,
                            content=content_bytes,
                            mime_type=mime_type or "application/octet-stream",
                            filename=img_url,
                            description=None,
                            metadata={"source_path": str(img_path)},
                        )
                    )
                else:
                    # Image file missing, skip or add as placeholder
                    images.append(
                        Image(
                            id=img_id,
                            content=b"",
                            mime_type="application/octet-stream",
                            filename=img_url,
                            description="Image file not found",
                            metadata={},
                        )
                    )

        for match in image_pattern.finditer(content):
            _parse_image_ref(match)

        # Remove frontmatter from content if present
        doc = cls(
            content=content,
            images=images,
            title=frontmatter.get("title"),
            author=frontmatter.get("author"),
            created=None,
            modified=None,
            source_path=str(path),
            mime_type=frontmatter.get("mime_type"),
            page_count=frontmatter.get("page_count"),
            metadata=frontmatter.get("metadata", {}),
        )
        if frontmatter.get("created"):
            with contextlib.suppress(Exception):
                doc.created = datetime.fromisoformat(frontmatter["created"])
        if frontmatter.get("modified"):
            with contextlib.suppress(Exception):
                doc.modified = datetime.fromisoformat(frontmatter["modified"])
        return doc

    @classmethod
    async def from_directory(
        cls,
        dir_path: StrPath,
        *,
        md_filename: str | None = None,
        load_images: bool = True,
    ) -> Document:
        """Load a Document from a directory containing markdown and image files.

        Args:
            dir_path: Directory containing the markdown and image files.
            md_filename: Name of the markdown file
                         (defaults to document.md or first .md file).
            load_images: Whether to load images referenced in the markdown.

        Returns:
            Document instance reconstructed from the directory.
        """
        import upath

        dirp = upath.UPath(dir_path)
        if not dirp.exists() or not dirp.is_dir():
            msg = f"Directory not found: {dir_path}"
            raise FileNotFoundError(msg)

        # Find markdown file
        md_path = None
        if md_filename:
            candidate = dirp / md_filename
            if candidate.exists():
                md_path = candidate
        if md_path is None:
            # Fallback: look for document.md or any .md file
            for name in ("document.md", "index.md"):
                candidate = dirp / name
                if candidate.exists():
                    md_path = candidate
                    break
            if md_path is None:
                md_files = list(dirp.glob("*.md"))
                if md_files:
                    md_path = md_files[0]
        if md_path is None:
            msg = f"No markdown file found in {dir_path}"
            raise FileNotFoundError(msg)

        # Use from_file to parse markdown and images
        doc = await cls.from_file(md_path, load_images=load_images)
        doc.source_path = str(dirp)
        return doc

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

    mime_type: MimeType | None = None
    """MIME type of the source document if available."""

    page_count: int | None = None
    """Number of pages in the source document if available."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata of the document."""

    def _build_markdown(
        self,
        *,
        include_frontmatter: bool = False,
        image_format: ImageReferenceFormat = "file_paths",
    ) -> str:
        """Create a markdown document with optional frontmatter and image handling.

        Args:
            include_frontmatter: Whether to include YAML frontmatter with doc metadata
            image_format: How to handle images in the output

        Returns:
            Complete markdown document as string
        """
        import yaml

        # Create frontmatter if requested
        frontmatter_content = ""
        if include_frontmatter:
            # Collect frontmatter data
            fm_data: dict[str, Any] = {}
            if self.title:
                fm_data["title"] = self.title
            if self.author:
                fm_data["author"] = self.author
            if self.created:
                fm_data["created"] = self.created.isoformat()
            if self.modified:
                fm_data["modified"] = self.modified.isoformat()
            if self.source_path:
                fm_data["source_path"] = self.source_path
            if self.mime_type:
                fm_data["mime_type"] = self.mime_type
            if self.page_count:
                fm_data["page_count"] = self.page_count

            # Include document metadata
            if self.metadata:
                fm_data["metadata"] = self.metadata

            # Generate YAML frontmatter
            if fm_data:
                yaml_text = yaml.dump(fm_data, default_flow_style=False, sort_keys=False)
                frontmatter_content = f"---\n{yaml_text}---\n\n"

        # Handle different image formats
        processed_content = self.content

        if image_format == "inline_base64":
            # Replace image references with base64 data URLs
            for image in self.images:
                if image.filename:
                    escaped_filename = re.escape(image.filename)
                    pattern = rf"!\[({re.escape(image.id)})\]\(({escaped_filename})\)"
                    data_url = image.to_base64_url()
                    replacement = f"![{image.id}]({data_url})"
                    processed_content = re.sub(pattern, replacement, processed_content)

        # Combine frontmatter and processed content
        return frontmatter_content + processed_content

    async def export_to_directory(
        self,
        output_dir: StrPath,
        *,
        include_frontmatter: bool = True,
        md_filename: str | None = None,
    ):
        """Export the document content and images to a directory.

        Saves the markdown content to 'document.md' and images to separate files
        within the specified directory. Assumes markdown content uses relative paths.

        Args:
            output_dir: The directory path to export to.
            include_frontmatter: Whether to include YAML frontmatter
            md_filename: Filename of the markdown file (defaults to document.md)
        """
        dir_path = upath.UPath(output_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Build markdown with requested options
        markdown_content = self._build_markdown(
            include_frontmatter=include_frontmatter,
            image_format="file_paths",
        )

        # Save markdown content
        md_path = dir_path / (md_filename or "document.md")
        md_path.write_text(markdown_content, encoding="utf-8")

        # Save images
        for image in self.images:
            if image.filename:
                img_path = dir_path / image.filename
                if isinstance(image.content, str):
                    # Decode if base64 string
                    img_bytes = base64.b64decode(image.to_base64())
                else:
                    img_bytes = image.content
                img_path.write_bytes(img_bytes)

    async def export_to_markdown_file(
        self,
        output_path: StrPath,
        *,
        include_frontmatter: bool = True,
        inline_images: bool = True,
    ):
        """Export the document as a markdown file with configurable options.

        Args:
            output_path: The file path to save the markdown file to.
            include_frontmatter: Whether to include YAML frontmatter
            inline_images: Whether to embed images as base64 data URLs (True)
                          or keep as file paths (False)
        """
        # Build markdown with requested options
        markdown_content = self._build_markdown(
            include_frontmatter=include_frontmatter,
            image_format="inline_base64" if inline_images else "file_paths",
        )

        # Save to file
        md_path = upath.UPath(output_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown_content, encoding="utf-8")

        # If not using inline images, save the image files
        if not inline_images and self.images:
            for image in self.images:
                if image.filename:
                    img_path = md_path.parent / image.filename
                    if isinstance(image.content, str):
                        # Decode if base64 string
                        img_bytes = base64.b64decode(image.to_base64())
                    else:
                        img_bytes = image.content
                    img_path.write_bytes(img_bytes)

    def to_markdown(
        self,
        *,
        include_frontmatter: bool = False,
        inline_images: bool = False,
    ) -> str:
        """Convert document to markdown with optional frontmatter and inline images.

        Args:
            include_frontmatter: Whether to include YAML frontmatter
            inline_images: Whether to embed images as base64 data URLs

        Returns:
            Complete markdown document as string
        """
        return self._build_markdown(
            include_frontmatter=include_frontmatter,
            image_format="inline_base64" if inline_images else "file_paths",
        )


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

    content: str
    source_doc_id: str
    chunk_index: int
    page_number: int | None = None
    images: list[Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_numbered_text(self, start_line: int | None = None) -> str:
        """Convert chunk text to numbered format.

        Args:
            start_line: The starting line number (1-based)
                        Defaults to metadata value if available

        Returns:
            Text with line numbers prefixed
        """
        if start_line is None:
            start_line = self.metadata.get("start_line", 1)

        lines = self.content.splitlines()
        return "\n".join(f"{start_line + i:5d} | {line}" for i, line in enumerate(lines))  # pyright: ignore


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
