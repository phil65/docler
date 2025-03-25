"""Vector store implementation for document and text chunk storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    import numpy as np

    from docler.models import SearchResult, TextChunk, Vector


Metric = Literal["cosine", "euclidean", "dot"]


class VectorStoreBackend(ABC):
    """Low-level vector store interface for raw vector operations."""

    @abstractmethod
    async def add_vector(
        self,
        vector: np.ndarray,
        metadata: dict[str, Any],
        id_: str | None = None,
    ) -> str:
        """Add single vector to store.

        Args:
            vector: Vector embedding to store
            metadata: Metadata dictionary for the vector
            id_: Optional ID (generated if not provided)

        Returns:
            ID of the stored vector
        """

    @abstractmethod
    async def add_vectors(
        self,
        vectors: list[np.ndarray],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add raw vectors to store.

        Args:
            vectors: List of vector embeddings to store
            metadata: List of metadata dictionaries (one per vector)
            ids: Optional list of IDs (generated if not provided)

        Returns:
            List of IDs for the stored vectors
        """

    @abstractmethod
    async def get_vector(self, chunk_id: str) -> Vector | None:
        """Get a vector and its metadata by ID.

        Args:
            chunk_id: ID of vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None if not
        """

    @abstractmethod
    async def search_vectors(
        self,
        query_vector: np.ndarray,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query_vector: Vector to search for
            k: Number of results to return
            filters: Optional filters to apply to results

        Returns:
            List of search results
        """

    @abstractmethod
    async def delete(self, chunk_id: str) -> bool:
        """Delete a vector by ID.

        Args:
            chunk_id: ID of vector to delete

        Returns:
            True if vector was deleted, False otherwise
        """


class VectorDB(ABC):
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
