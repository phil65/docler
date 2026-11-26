"""Markdown document models and helpers (formerly the separate `mkdown` package)."""

from docler.markdown.document import Document
from docler.markdown.textchunk import TextChunk
from docler.markdown.image import Image
from docler.markdown.utils import (
    CHUNK_BOUNDARY_TYPE,
    DEFAULT_PREFIX,
    PAGE_BREAK_TYPE,
    create_chunk_boundary,
    create_page_break,
    create_image_reference,
    create_metadata_comment,
    get_chunk_boundaries,
    parse_metadata_comments,
    split_markdown_by_chunks,
    split_markdown_by_page,
)


__all__ = [
    "CHUNK_BOUNDARY_TYPE",
    "DEFAULT_PREFIX",
    "PAGE_BREAK_TYPE",
    "Document",
    "Image",
    "TextChunk",
    "create_chunk_boundary",
    "create_image_reference",
    "create_metadata_comment",
    "create_page_break",
    "get_chunk_boundaries",
    "parse_metadata_comments",
    "split_markdown_by_chunks",
    "split_markdown_by_page",
]
