"""Base markdown chunking implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mkdown import TextChunk

from docler.chunkers.base import TextChunker
from docler.chunkers.markdown_chunker.utils import assign_images, split_by_headers
from docler.configs.chunker_configs import MarkdownChunkerConfig


if TYPE_CHECKING:
    from collections.abc import Iterator

    from mkdown import Document, Image


class MarkdownChunker(TextChunker[MarkdownChunkerConfig]):
    """Header-based markdown chunker with fallback to size-based chunks."""

    Config = MarkdownChunkerConfig
    NAME = "markdown"

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

    def _fallback_split(
        self, content: str, images: list[Image]
    ) -> Iterator[tuple[str, list[Image]]]:
        """Split content by size when no headers exist."""
        start = 0
        while start < len(content):
            chunk_content = content[start : start + self.max_chunk_size]
            chunk_content, chunk_images = assign_images(chunk_content, images)
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
        header_sections = list(split_by_headers(doc.content))
        if not header_sections:
            # Fallback to size-based if no headers
            for content, images in self._fallback_split(doc.content, doc.images):
                chunk = TextChunk(
                    content=content,
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
            meta = {**(extra_metadata or {}), "header": header, "level": level}
            if len(content) > self.max_chunk_size:
                for sub_content, images in self._fallback_split(content, doc.images):
                    chunk = TextChunk(
                        content=f"{header}\n\n{sub_content}",
                        source_doc_id=doc.source_path or "",
                        chunk_index=chunk_idx,
                        images=images,
                        metadata=meta,
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
            else:
                content, images = assign_images(content, doc.images)
                chunk = TextChunk(
                    content=f"{header}\n\n{content}",
                    source_doc_id=doc.source_path or "",
                    chunk_index=chunk_idx,
                    images=images,
                    metadata=meta,
                )
                chunks.append(chunk)
                chunk_idx += 1

        return chunks
