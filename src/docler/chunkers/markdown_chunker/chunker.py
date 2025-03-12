"""Base markdown chunking implementation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from docler.chunkers.base import TextChunk, TextChunker


if TYPE_CHECKING:
    from collections.abc import Iterator

    from docler.models import Document, Image


class MarkdownChunker(TextChunker):
    """Header-based markdown chunker with fallback to size-based chunks."""

    def __init__(
        self,
        *,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
        chunk_overlap: int = 50,
    ):
        """Initialize chunker.

        Args:
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_by_headers(self, text: str) -> Iterator[tuple[str, str, int]]:
        """Split text by markdown headers.

        Returns:
            Iterator of (header, content, level) tuples
        """
        # Matches markdown headers (# to ######)
        header_pattern = r"^(#{1,6})\s+(.+)$"

        current_header = ""
        current_level = 0
        current_content: list[str] = []

        for line in text.splitlines():
            if match := re.match(header_pattern, line):
                # Yield previous section if exists
                if current_content:
                    yield current_header, "\n".join(current_content), current_level
                    current_content = []

                current_level = len(match.group(1))
                current_header = match.group(2)
            else:
                current_content.append(line)

        # Yield final section
        if current_content:
            yield current_header, "\n".join(current_content), current_level

    def _assign_images(
        self, content: str, all_images: list[Image]
    ) -> tuple[str, list[Image]]:
        """Find images referenced in content and assign them to chunk.

        Returns:
            Tuple of (content, chunk_images)
        """
        chunk_images: list[Image] = []
        image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

        for match in re.finditer(image_pattern, content):
            image_path = match.group(2)
            for image in all_images:
                if image.filename == image_path:
                    chunk_images.append(image)
                    break

        return content, chunk_images

    def _fallback_split(
        self, content: str, images: list[Image]
    ) -> Iterator[tuple[str, list[Image]]]:
        """Split content by size when no headers exist."""
        start = 0
        while start < len(content):
            chunk_content = content[start : start + self.max_chunk_size]
            chunk_content, chunk_images = self._assign_images(chunk_content, images)
            yield chunk_content, chunk_images
            start += self.max_chunk_size - self.chunk_overlap

    async def split(
        self,
        doc: Document,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Split document into chunks."""
        chunks: list[TextChunk] = []
        chunk_idx = 0

        # Try header-based splitting first
        header_sections = list(self._split_by_headers(doc.content))
        if not header_sections:
            # Fallback to size-based if no headers
            for content, images in self._fallback_split(doc.content, doc.images):
                chunk = TextChunk(
                    text=content,
                    source_doc_id=doc.source_path or "",
                    chunk_index=chunk_idx,
                    images=images,
                    metadata=extra_metadata or {},
                )
                chunks.append(chunk)
                chunk_idx += 1
            return chunks

        # Process header sections
        for header, content, level in header_sections:
            if len(content) > self.max_chunk_size:
                # Split large sections
                for sub_content, images in self._fallback_split(content, doc.images):
                    meta = {**(extra_metadata or {}), "header": header, "level": level}
                    chunk = TextChunk(
                        text=f"{header}\n\n{sub_content}",
                        source_doc_id=doc.source_path or "",
                        chunk_index=chunk_idx,
                        images=images,
                        metadata=meta,
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
            else:
                # Use section as-is
                content, images = self._assign_images(content, doc.images)
                meta = {**(extra_metadata or {}), "header": header, "level": level}
                chunk = TextChunk(
                    text=f"{header}\n\n{content}",
                    source_doc_id=doc.source_path or "",
                    chunk_index=chunk_idx,
                    images=images,
                    metadata=meta,
                )
                chunks.append(chunk)
                chunk_idx += 1

        return chunks
