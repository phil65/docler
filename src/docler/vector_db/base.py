"""Vector store implementation for document and text chunk storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from docler.chunkers.base import TextChunk
from docler.models import Image
from docler.provider import BaseProvider


if TYPE_CHECKING:
    from llmling_agent.embeddings import EmbeddingProvider
    import numpy as np


Metric = Literal["cosine", "euclidean", "dot"]


@dataclass
class SearchResult:
    """A single vector search result."""

    chunk_id: str
    score: float  # similarity score between 0-1
    metadata: dict[str, Any]
    text: str | None = None


class VectorStoreBackend(BaseProvider, ABC):
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
    async def get_vector(
        self,
        chunk_id: str,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Get a vector and its metadata by ID.

        Args:
            chunk_id: ID of vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None if not
        """

    @abstractmethod
    async def update_vector(
        self,
        chunk_id: str,
        vector: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing vector.

        Args:
            chunk_id: ID of vector to update
            vector: New vector embedding (unchanged if None)
            metadata: New metadata (unchanged if None)

        Returns:
            True if vector was updated, False if not found
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

    @abstractmethod
    async def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add simple text strings with metadata.

        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs

        Returns:
            List of IDs for the stored texts
        """

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
    async def similar_chunks(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, float]]:
        """Find similar chunks for a query.

        Args:
            query: Query text to search for
            k: Number of results to return
            filters: Optional filters to apply to results

        Returns:
            List of (chunk, score) tuples
        """

    @abstractmethod
    async def similar_texts(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
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


class CompositeVectorDB(VectorDB):
    """Vector database that combines a backend with an embedding model."""

    def __init__(
        self,
        backend: VectorStoreBackend,
        embedding_model: EmbeddingProvider,
    ):
        """Initialize vector store.

        Args:
            backend: Vector store backend
            embedding_model: Embedding model to use
        """
        self._backend = backend
        self._embeddings = embedding_model

    async def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add simple text strings with metadata.

        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs

        Returns:
            List of IDs for the stored texts
        """
        # Handle default arguments
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Convert to embeddings
        embeddings = await self._embeddings.embed_texts(texts)

        # Add text to metadata
        for i, text in enumerate(texts):
            metadatas[i]["text"] = text
        return await self._backend.add_vectors(embeddings, metadatas, ids=ids)

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
        # Convert chunks to embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = await self._embeddings.embed_texts(texts)

        # Prepare metadata
        metadata = []
        for chunk in chunks:
            # Extract image data if present
            image_data = {}
            for i, img in enumerate(chunk.images):
                if isinstance(img.content, bytes):
                    encoded = base64.b64encode(img.content).decode()
                else:
                    encoded = img.content

                image_data[f"image_{i}"] = {
                    "id": img.id,
                    "content": encoded,
                    "mime_type": img.mime_type,
                    "filename": img.filename,
                }

            # Create metadata dict with chunk properties
            chunk_meta = {
                "text": chunk.text,
                "source_doc_id": chunk.source_doc_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "images": image_data,
                **chunk.metadata,
            }
            metadata.append(chunk_meta)

        # Store in backend
        chunk_ids = [f"{chunk.source_doc_id}_{chunk.chunk_index}" for chunk in chunks]
        return await self._backend.add_vectors(embeddings, metadata, ids=chunk_ids)

    async def similar_chunks(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, float]]:
        """Find similar chunks for a query.

        Args:
            query: Query text to search for
            k: Number of results to return
            filters: Optional filters to apply to results

        Returns:
            List of (chunk, score) tuples
        """
        # Convert query to embedding
        embedding = await self._embeddings.embed_query(query)

        # Search backend
        results = await self._backend.search_vectors(embedding, k=k, filters=filters)

        # Convert results back to chunks
        chunks_with_scores = []
        for result in results:
            # Extract images if present
            images = []
            if "images" in result.metadata:
                for img_data in result.metadata["images"].values():
                    if not img_data or not isinstance(img_data, dict):
                        continue

                    # Convert base64 content back to bytes
                    content = img_data["content"]
                    if isinstance(content, str):
                        content = base64.b64decode(content)
                    image = Image(
                        id=img_data["id"],
                        content=content,
                        mime_type=img_data["mime_type"],
                        filename=img_data.get("filename"),
                    )
                    images.append(image)

            # Create chunk from metadata
            excl = ["text", "source_doc_id", "chunk_index", "page_number", "images"]
            metadata = {k: v for k, v in result.metadata.items() if k not in excl}
            chunk = TextChunk(
                text=result.metadata.get("text", ""),
                source_doc_id=result.metadata.get("source_doc_id", ""),
                chunk_index=result.metadata.get("chunk_index", 0),
                page_number=result.metadata.get("page_number"),
                images=images,
                metadata=metadata,
            )
            chunks_with_scores.append((chunk, result.score))
        return chunks_with_scores

    async def similar_texts(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Find similar texts for a query.

        Args:
            query: Query text to search for
            k: Number of results to return
            filters: Optional filters to apply to results

        Returns:
            List of (text, score, metadata) tuples
        """
        embedding = await self._embeddings.embed_query(query)
        results = await self._backend.search_vectors(embedding, k=k, filters=filters)
        return [
            (
                result.metadata.get("text", ""),
                result.score,
                {k: v for k, v in result.metadata.items() if k != "text"},
            )
            for result in results
        ]

    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID.

        Args:
            chunk_id: ID of chunk to delete

        Returns:
            True if chunk was deleted, False otherwise
        """
        return await self._backend.delete(chunk_id)


class IntegratedVectorDB(VectorDB):
    """Base class for vector databases with integrated embedding and storage."""

    @abstractmethod
    async def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add simple text strings with metadata."""

    @abstractmethod
    async def add_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[str]:
        """Add text chunks with metadata."""

    @abstractmethod
    async def similar_chunks(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, float]]:
        """Find similar chunks for a query."""

    @abstractmethod
    async def similar_texts(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Find similar texts for a query."""

    @abstractmethod
    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID."""
