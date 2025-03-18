"""Base classes for text chunking implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from docler.provider import BaseProvider


if TYPE_CHECKING:
    from collections.abc import Iterable

    from docler.models import Document, Image


@dataclass
class TextChunk:
    """Chunk of text with associated metadata and images."""

    text: str
    source_doc_id: str
    chunk_index: int
    page_number: int | None = None
    images: list[Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TextChunker(BaseProvider, ABC):
    """Base class for text chunkers."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    async def split(
        self,
        text: Document,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Split text into chunks."""
        raise NotImplementedError

    async def split_texts(
        self, texts: Iterable[Document | tuple[Document, dict[str, Any]]]
    ) -> list[TextChunk]:
        """Split multiple texts into chunks."""
        result: list[TextChunk] = []
        for item in texts:
            text, metadata = item if isinstance(item, tuple) else (item, None)
            chunks = await self.split(text, metadata)
            result.extend(chunks)
        return result
