"""Data models for document representation."""

import base64
import contextlib
from datetime import datetime
import mimetypes
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field
from schemez import MimeType, Schema
import upath

from docler.markdown.image import Image
from docler.markdown.utils import create_image_reference


if TYPE_CHECKING:
    from upath.types import JoinablePathLike


ImageReferenceFormat = Literal["inline_base64", "file_paths", "keep_internal"]


class Document(Schema):
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

    mime_type: MimeType | None = None
    """MIME type of the source document if available."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata of the document."""

    @property
    def page_count(self) -> int:
        return self.content.count("<!-- docler:page_break ")

    @classmethod
    async def from_file(cls, file_path: JoinablePathLike, *, load_images: bool = True) -> Document:
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
                image = Image(id=img_id, content=b64_data, mime_type=mime_type)
                images.append(image)
            elif load_images:
                # File path reference, try to load file if possible
                img_path = path.parent / img_url
                if img_path.exists():
                    mime_type = None
                    with contextlib.suppress(Exception):
                        mime_type, _ = mimetypes.guess_type(str(img_path))
                    content_bytes = img_path.read_bytes()
                    image = Image(
                        id=img_id,
                        content=content_bytes,
                        mime_type=mime_type or "application/octet-stream",
                        filename=img_url,
                        metadata={"source_path": str(img_path)},
                    )
                    images.append(image)
                else:
                    # Image file missing, skip or add as placeholder
                    image = Image(
                        id=img_id,
                        content=b"",
                        mime_type="application/octet-stream",
                        filename=img_url,
                        description="Image file not found",
                    )
                    images.append(image)

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
            metadata=frontmatter.get("metadata", {}),
        )
        if created_val := frontmatter.get("created"):
            with contextlib.suppress(Exception):
                doc.created = datetime.fromisoformat(str(created_val))
        if modified_val := frontmatter.get("modified"):
            with contextlib.suppress(Exception):
                doc.modified = datetime.fromisoformat(str(modified_val))
        return doc

    @classmethod
    async def from_directory(
        cls,
        dir_path: JoinablePathLike,
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
                    replacement = create_image_reference(image.id, data_url)
                    processed_content = re.sub(pattern, replacement, processed_content)

        # Combine frontmatter and processed content
        return frontmatter_content + processed_content

    async def export_to_directory(
        self,
        output_dir: JoinablePathLike,
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
        output_path: JoinablePathLike,
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
