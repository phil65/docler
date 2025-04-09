"""Base classes for text chunking implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, overload

from docler.models import ChunkedDocument
from docler.provider import BaseProvider


if TYPE_CHECKING:
    from docler.models import Document, TextChunk


class TextChunker[TConfig](BaseProvider[TConfig], ABC):
    """Base class for text chunkers."""

    NAME: str

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

    @overload
    async def chunk(
        self,
        document_or_documents: Document,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ChunkedDocument: ...

    @overload
    async def chunk(
        self,
        document_or_documents: list[Document],
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[ChunkedDocument]: ...

    async def chunk(
        self,
        document_or_documents: Document | list[Document],
        extra_metadata: dict[str, Any] | None = None,
    ) -> ChunkedDocument | list[ChunkedDocument]:
        """Split document(s) into chunks and return ChunkedDocument(s).

        Args:
            document_or_documents: Document or list of documents to split
            extra_metadata: Additional metadata to include in chunks

        Returns:
            ChunkedDocument or list of ChunkedDocuments containing
            both the original document(s) and their chunks
        """
        if isinstance(document_or_documents, list):
            # Process a list of documents
            results: list[ChunkedDocument] = []
            for document in document_or_documents:
                chunks = await self.split(document, extra_metadata)
                chunked_document = ChunkedDocument.from_document(document, chunks)
                results.append(chunked_document)
            return results
        # Process a single document
        document = document_or_documents
        chunks = await self.split(document, extra_metadata)
        return ChunkedDocument.from_document(document, chunks)
