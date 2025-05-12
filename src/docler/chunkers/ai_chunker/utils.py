"""AI-based markdown chunking implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mkdown import TextChunk


if TYPE_CHECKING:
    from mkdown import Document

    from docler.chunkers.ai_chunker.models import Chunk


def add_line_numbers(text: str) -> str:
    """Add line numbers to text."""
    lines = text.splitlines()
    return "\n".join(f"{i + 1:3d} | {line}" for i, line in enumerate(lines))


def create_text_chunk(
    doc: Document,
    chunk: Chunk,
    chunk_idx: int,
    extra_metadata: dict[str, Any] | None = None,
) -> TextChunk:
    """Create a TextChunk from chunk definition."""
    lines = doc.content.splitlines()
    chunk_lines = lines[chunk.start_row - 1 : chunk.end_row]
    chunk_text = "\n".join(chunk_lines)
    base = extra_metadata or {}
    metadata = {**base, "keywords": chunk.keywords, "references": chunk.references}
    chunk_images = [i for i in doc.images if i.filename and i.filename in chunk_text]
    return TextChunk(
        content=chunk_text,
        source_doc_id=doc.source_path or "",
        chunk_index=chunk_idx,
        images=chunk_images,
        metadata=metadata,
    )
