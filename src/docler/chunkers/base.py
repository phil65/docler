"""Base classes for text chunking implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, overload

from mkdown import Document, create_chunk_boundary

from docler.models import ChunkedDocument
from docler.provider import BaseProvider


if TYPE_CHECKING:
    from mkdown import TextChunk


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

    async def chunk_with_boundaries(
        self,
        document: Document,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Split document into chunks and add chunk boundary markers to content.

        This method creates a new document that contains the same content as the
        original but with chunk boundary markers inserted at the beginning of each chunk.

        Args:
            document: Document to split and mark
            extra_metadata: Additional metadata to include in chunks

        Returns:
            Document with chunk boundary markers inserted
        """
        chunks = await self.split(document, extra_metadata)
        return self.add_chunk_boundaries_to_doc(document, chunks)

    def add_chunk_boundaries_to_doc(
        self,
        doc: Document,
        chunks: list[TextChunk],
    ) -> Document:
        """Add chunk boundary comments to document content.

        This method enriches the document content with chunk boundary
        comments that mark where each chunk begins.

        Args:
            doc: Original document
            chunks: Chunks created from the document

        Returns:
            Document with chunk boundary comments inserted
        """
        # Use line numbers from chunks if available, otherwise use character offsets
        lines = doc.content.splitlines()
        content_with_boundaries = []

        # Sort chunks by start position if available
        sorted_chunks = sorted(
            chunks, key=lambda c: c.metadata.get("start_line", c.chunk_index)
        )

        for i, chunk in enumerate(sorted_chunks):
            metadata = chunk.metadata or {}
            start_line = metadata.get("start_line")
            end_line = metadata.get("end_line")

            # If this is the first chunk or we don't have line numbers,
            # add the boundary at the beginning of the content
            if i == 0 and start_line is None:
                boundary = create_chunk_boundary(
                    chunk_id=chunk.chunk_index,
                    keywords=metadata.get("keywords"),
                    extra_data=metadata,
                )
                content_with_boundaries.append(boundary)
                content_with_boundaries.append(doc.content)
                break

            # Add the boundary before the chunk
            # If we have line numbers, add the boundary before that line
            if start_line is not None and end_line is not None:
                # Adjust line numbers to be 0-based for array indexing
                start_idx = max(0, start_line - 1)

                if i == 0:
                    # Handle first chunk
                    boundary = create_chunk_boundary(
                        chunk_id=chunk.chunk_index,
                        keywords=metadata.get("keywords"),
                        extra_data={
                            k: v
                            for k, v in metadata.items()
                            if k
                            not in {"start_line", "end_line", "keywords", "token_count"}
                        },
                    )

                    # Add content before first chunk
                    content_with_boundaries.extend(lines[:start_idx])
                    # Add boundary
                    content_with_boundaries.append(boundary)
                    # Add content of first chunk
                    if i == len(sorted_chunks) - 1:
                        # If this is the only chunk, add all remaining content
                        content_with_boundaries.extend(lines[start_idx:])
                else:
                    # Handle middle chunks
                    boundary = create_chunk_boundary(
                        chunk_id=chunk.chunk_index,
                        keywords=metadata.get("keywords"),
                        extra_data={
                            k: v
                            for k, v in metadata.items()
                            if k
                            not in {"start_line", "end_line", "keywords", "token_count"}
                        },
                    )

                    # Add boundary
                    content_with_boundaries.append(boundary)

                    # Add content until end of chunk or next chunk
                    if i == len(sorted_chunks) - 1:
                        # If this is the last chunk, add all remaining content
                        content_with_boundaries.extend(lines[start_idx:])
                    else:
                        next_start = sorted_chunks[i + 1].metadata.get(
                            "start_line", len(lines) + 1
                        )
                        next_start_idx = max(0, next_start - 1)
                        content_with_boundaries.extend(lines[start_idx:next_start_idx])

        # If we processed the chunks line by line, join them back
        if content_with_boundaries:
            return Document(
                content="\n".join(content_with_boundaries),
                images=doc.images,
                title=doc.title,
                source_path=doc.source_path,
                mime_type=doc.mime_type,
                metadata=doc.metadata,
            )

        # Fallback to simple boundary placement
        boundaries = []
        for chunk in chunks:
            boundary = create_chunk_boundary(
                chunk_id=chunk.chunk_index,
                keywords=chunk.metadata.get("keywords"),
                extra_data=chunk.metadata,
            )
            boundaries.append(boundary)

        boundary_content = f"\n\n{boundaries[0]}\n\n{doc.content}"
        for boundary in boundaries[1:]:
            # Just append at the end as a reference
            boundary_content += f"\n\n{boundary}\n\n"

        return Document(
            content=boundary_content,
            images=doc.images,
            title=doc.title,
            source_path=doc.source_path,
            mime_type=doc.mime_type,
            metadata=doc.metadata,
        )
