"""Vector store implementation for document and text chunk storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from mkdown import TextChunk


Metric = Literal["cosine", "euclidean", "dot"]


class BaseVectorDB(ABC):
    """Abstract interface for vector databases that handle both storage and retrieval."""

    def __init__(self, vector_store_id: str):
        self.vector_store_id = vector_store_id

    @abstractmethod
    async def add_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[str]:
        """Add text chunks with metadata.

        Args:
            chunks: List of text chunks to add

        Returns:
            List of IDs for the stored chunks
        """

    @abstractmethod
    async def query(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, float]]:
        """Find similar texts for a query.

        Args:
            query: Query text to search for
            k: Number of results to return
            filters: Optional filters to apply to results

        Returns:
            List of (text, score, metadata) tuples
        """

    @abstractmethod
    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID.

        Args:
            chunk_id: ID of chunk to delete

        Returns:
            True if chunk was deleted, False otherwise
        """
