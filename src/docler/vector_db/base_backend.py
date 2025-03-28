"""Vector store implementation for document and text chunk storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np

    from docler.models import SearchResult, Vector


class VectorStoreBackend(ABC):
    """Low-level vector store interface for raw vector operations."""

    def __init__(self, *, vector_store_id: str, **kwargs: Any):
        """Initialize vector store backend.

        Args:
            vector_store_id: Unique identifier for this vector store instance
            kwargs: Additional keyword arguments for the backend
        """
        self._vector_store_id = vector_store_id
        super().__init__(**kwargs)

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

    async def add_vector(
        self,
        vector: np.ndarray,
        metadata: dict[str, Any],
        id_: str | None = None,
    ) -> str:
        """Add single vector to store.

        Default implementation calls add_vectors with a single item.
        Override this method if you need special handling for single items.
        """
        ids = await self.add_vectors([vector], [metadata], [id_] if id_ else None)
        return ids[0]

    @abstractmethod
    async def delete(self, chunk_id: str) -> bool:
        """Delete a vector by ID.

        Args:
            chunk_id: ID of vector to delete

        Returns:
            True if vector was deleted, False otherwise
        """


class LocalVectorStoreBackend(VectorStoreBackend):
    """Vector store that operates entirely locally with file-based storage."""

    def __init__(self, *, vector_store_id: str, storage_path: str, **kwargs: Any):
        """Initialize a local vector store.

        Args:
            vector_store_id: Unique identifier for this vector store instance
            storage_path: Path to local storage directory
            kwargs: Additional keyword arguments
        """
        super().__init__(vector_store_id=vector_store_id, **kwargs)
        self._storage_path = storage_path

    @property
    def storage_path(self) -> str:
        """Path to local storage location."""
        return self._storage_path


class RemoteVectorStoreBackend(VectorStoreBackend):
    """Vector store that connects to a remote service."""

    def __init__(  # noqa: D417
        self,
        *,
        vector_store_id: str,
        connection_url: str,
        api_key: str | None = None,
        **kwargs,
    ):
        """Initialize a remote vector store.

        Args:
            vector_store_id: Unique identifier for this vector store instance
            connection_url: URL to connect to the remote service
            api_key: Optional API key for authentication
        """
        super().__init__(vector_store_id=vector_store_id, **kwargs)
        self._connection_url = connection_url
        self._api_key = api_key

    @property
    def connection_url(self) -> str:
        """URL for the remote service."""
        return self._connection_url
