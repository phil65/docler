"""ChromaDB vector store backend implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast
import uuid

from docler.models import SearchResult
from docler.vector_db.base import VectorStoreBackend


if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)


class ChromaBackend(VectorStoreBackend):
    """ChromaDB implementation of vector store backend."""

    NAME = "ChromaDB"
    REQUIRED_PACKAGES: ClassVar = {"chromadb"}

    def __init__(
        self,
        vector_store_id: str = "default",
        persist_directory: str | None = None,
        distance_metric: str = "cosine",
    ):
        """Initialize ChromaDB backend.

        Args:
            vector_store_id: Name of collection to use
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

        # Get or create collection without embedding function
        self._collection = self._client.get_or_create_collection(
            name=vector_store_id,
            metadata={"hnsw:space": distance_metric},
            embedding_function=None,  # We provide embeddings directly
        )

        logger.info(
            "ChromaDB initialized - collection: %s, persistent: %s",
            vector_store_id,
            bool(persist_directory),
        )

    async def add_vector(
        self,
        vector: np.ndarray,
        metadata: dict[str, Any],
        id_: str | None = None,
    ) -> str:
        """Add single vector to ChromaDB."""
        import anyenv

        # Generate ID if not provided
        if id_ is None:
            id_ = str(uuid.uuid4())

        # Convert numpy array to list for JSON serialization
        vector_list = vector.tolist()

        try:
            # Add new vector
            await anyenv.run_in_thread(
                self._collection.add,
                ids=[id_],
                embeddings=[vector_list],
                metadatas=[metadata],
            )
        except ValueError:
            # Update if already exists
            await anyenv.run_in_thread(
                self._collection.update,
                ids=[id_],
                embeddings=[vector_list],
                metadatas=[metadata],
            )

        return id_

    async def add_vectors(
        self,
        vectors: list[np.ndarray],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add multiple vectors to ChromaDB."""
        import anyenv

        if len(vectors) != len(metadata):
            msg = "Number of vectors and metadata entries must match"
            raise ValueError(msg)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

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

    async def get_vector(
        self,
        chunk_id: str,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Get vector and metadata from ChromaDB."""
        import anyenv
        import numpy as np

        result = await anyenv.run_in_thread(
            self._collection.get,
            ids=[chunk_id],
            include=["embeddings", "metadatas"],
        )

        # Check if vector exists and has results
        if not result["ids"] or not result["embeddings"] or not result["metadatas"]:
            return None

        # Convert list to numpy array
        vector = np.array(result["embeddings"][0])
        # Cast metadata to dict[str, Any]
        metadata = cast(dict[str, Any], result["metadatas"][0])

        return vector, metadata

    async def update_vector(
        self,
        chunk_id: str,
        vector: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update vector in ChromaDB."""
        import anyenv

        # Get current vector if it exists
        current = await self.get_vector(chunk_id)
        if current is None:
            return False

        current_vector, current_metadata = current

        # Use new values or keep current ones
        update_vector = vector if vector is not None else current_vector
        update_metadata = metadata if metadata is not None else current_metadata

        # Convert numpy array to list
        vector_list = update_vector.tolist()

        try:
            await anyenv.run_in_thread(
                self._collection.update,
                ids=[chunk_id],
                embeddings=[vector_list],
                metadatas=[update_metadata],
            )
        except Exception:  # noqa: BLE001
            return False
        else:
            return True

    async def delete(self, chunk_id: str) -> bool:
        """Delete vector from ChromaDB."""
        import anyenv

        try:
            await anyenv.run_in_thread(self._collection.delete, ids=[chunk_id])
        except Exception:  # noqa: BLE001
            return False
        else:
            return True

    async def search_vectors(
        self,
        query_vector: np.ndarray,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search ChromaDB for similar vectors."""
        import anyenv

        # Convert query vector to list
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
        search_results: list[Any] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        distances = results.get("distances")
        metadatas = results.get("metadatas")

        assert distances is not None
        assert distances[0]
        assert metadatas is not None
        assert metadatas[0]

        for i, doc_id in enumerate(results["ids"][0]):
            metadata = dict(metadatas[0][i])
            text = metadata.pop("text", None)
            if text is not None:
                text = str(text)

            result = SearchResult(
                chunk_id=str(doc_id),
                score=1.0 - float(distances[0][i]),
                metadata=metadata,
                text=text,
            )
            search_results.append(result)

        return search_results
