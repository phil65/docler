"""RAG-based tool registry for semantic tool discovery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
import uuid

from docler.vector_db.base import Metric, SearchResult, VectorStore


if TYPE_CHECKING:
    import numpy as np

    from docler.configs.vector_db_configs import ChromaConfig


logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation."""

    def __init__(self, config: ChromaConfig):
        """Initialize ChromaDB vector store.

        Args:
            config: ChromaDB configuration
        """
        import chromadb

        # Create client based on configuration
        if config.persist_directory:
            self._client = chromadb.PersistentClient(path=config.persist_directory)
        else:
            self._client = chromadb.Client()

        # Get or create collection
        self._collection = self._client.get_or_create_collection(  # pyright: ignore
            name=config.collection_name, metadata={"hnsw:space": "cosine"}
        )

        logger.info("ChromaDB initialized with %s", config)

    def add(
        self,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Add vector to ChromaDB."""
        # Generate ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # Convert numpy to list for JSON serialization
        vector_list = embedding.tolist()

        try:
            self._collection.add(
                ids=[doc_id],
                embeddings=[vector_list],
                metadatas=[metadata] if metadata else None,
            )
        except ValueError:
            # Update if already exists
            self._collection.update(
                ids=[doc_id],
                embeddings=[vector_list],
                metadatas=[metadata] if metadata else None,
            )

        return doc_id

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        metric: Metric = "cosine",
        search_params: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search ChromaDB for similar vectors."""
        # Execute search
        embeddings = [query_vector.tolist()]
        results = self._collection.query(
            query_embeddings=embeddings,
            n_results=limit,
            where=filters,
            include=["metadatas", "distances"],  # pyright: ignore
        )

        # Format results
        formatted_results: list[SearchResult] = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if "distances" in results else 0.0  # pyright: ignore
                similarity = 1.0 - distance  # Convert distance to similarity
                metadata = results["metadatas"][0][i] if "metadatas" in results else {}  # pyright: ignore
                result = SearchResult(doc_id, similarity, metadata)  # pyright: ignore
                formatted_results.append(result)

        return formatted_results

    def delete(self, doc_id: str) -> bool:
        """Delete vector by ID."""
        try:
            self._collection.delete(ids=[doc_id])
            return True  # noqa: TRY300
        except Exception:  # noqa: BLE001
            return False
