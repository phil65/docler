"""ChromaDB vector store backend implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from docler.vector_db.base import SearchResult, VectorStoreBackend


if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)


class ChromaBackend(VectorStoreBackend):
    """ChromaDB implementation of vector store backend."""

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str | None = None,
        distance_metric: str = "cosine",
    ) -> None:
        """Initialize ChromaDB backend.

        Args:
            collection_name: Name of collection to use
            persist_directory: Directory for persistent storage (memory if None)
            distance_metric: Distance metric to use for similarity search

        Raises:
            ImportError: If chromadb is not installed
        """
        try:
            import chromadb
        except ImportError as e:
            msg = "ChromaDB is not installed. Please install with 'pip install chromadb'"
            raise ImportError(msg) from e

        # Create client based on configuration
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        # Get or create collection
        self._collection = self._client.get_or_create_collection(  # pyright: ignore
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        msg = "ChromaDB initialized - collection: %s, persistent: %s"
        logger.info(msg, collection_name, bool(persist_directory))

    async def add_vectors(
        self,
        vectors: list[np.ndarray],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add vectors to ChromaDB.

        Args:
            vectors: List of vector embeddings
            metadata: List of metadata dictionaries
            ids: Optional list of IDs for vectors

        Returns:
            List of IDs for stored vectors

        Raises:
            ValueError: If vectors and metadata counts don't match
        """
        from uuid import uuid4

        import anyenv

        if len(vectors) != len(metadata):
            msg = "Number of vectors and metadata entries must match"
            raise ValueError(msg)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in vectors]

        # Convert numpy arrays to lists for JSON serialization
        vector_lists = [v.tolist() for v in vectors]

        try:
            # Try to add new vectors
            await anyenv.run_in_thread(
                self._collection.add,
                ids=ids,
                embeddings=vector_lists,
                metadatas=metadata,
            )
        except ValueError:
            # If vectors exist, update them
            await anyenv.run_in_thread(
                self._collection.update,
                ids=ids,
                embeddings=vector_lists,
                metadatas=metadata,
            )

        return ids

    async def search_vectors(
        self,
        query_vector: np.ndarray,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search ChromaDB for similar vectors.

        Args:
            query_vector: Vector to search for
            k: Number of results to return
            filters: Optional metadata filters to apply

        Returns:
            List of search results
        """
        import anyenv

        # Convert query vector to list for JSON serialization
        query_list = [query_vector.tolist()]

        # Execute search
        results = await anyenv.run_in_thread(
            self._collection.query,
            query_embeddings=query_list,
            n_results=k,
            where=filters,
            include=["metadatas", "distances"],
        )

        # Format results
        search_results = []
        if results["ids"] and results["ids"][0]:  # type: ignore
            for i, doc_id in enumerate(results["ids"][0]):  # type: ignore
                # Convert distance to similarity score (1 - distance)
                distance = results["distances"][0][i] if "distances" in results else 0.0  # type: ignore
                similarity = 1.0 - distance

                metadata: dict[str, Any] = (
                    results["metadatas"][0][i] if "metadatas" in results else {}  # type: ignore
                )
                text = metadata.get("text")
                result = SearchResult(
                    chunk_id=str(doc_id),
                    score=similarity,
                    metadata=metadata,
                    text=text,
                )
                search_results.append(result)

        return search_results

    async def delete(self, chunk_id: str) -> bool:
        """Delete vector by ID.

        Args:
            chunk_id: ID of vector to delete

        Returns:
            True if vector was deleted successfully
        """
        import anyenv

        try:
            await anyenv.run_in_thread(self._collection.delete, ids=[chunk_id])
        except Exception:  # noqa: BLE001
            return False
        else:
            return True
