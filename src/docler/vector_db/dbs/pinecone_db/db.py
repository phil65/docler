"""Pinecone vector store backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
import uuid

from docler.log import get_logger
from docler.models import SearchResult, Vector
from docler.vector_db.base import VectorStoreBackend
from docler.vector_db.dbs.pinecone_db.utils import (
    convert_filters,
    prepare_metadata,
    restore_metadata,
)


if TYPE_CHECKING:
    import numpy as np
    from pinecone.control.pinecone_asyncio import _IndexAsyncio as IndexAsyncio


logger = get_logger(__name__)


class PineconeBackend(VectorStoreBackend):
    """Pinecone implementation of vector store backend."""

    NAME: ClassVar[str] = "pinecone"
    REQUIRED_PACKAGES: ClassVar = {"pinecone-client"}

    def __init__(
        self,
        host: str,
        api_key: str,
        dimension: int = 1536,
        namespace: str = "default",
    ):
        """Initialize Pinecone backend.

        Args:
            host: Host URL for the index
            api_key: Pinecone API key
            dimension: Dimension of vectors to store
            namespace: Namespace to use for vectors
        """
        self.api_key = api_key
        self._host = host
        self._index: IndexAsyncio | None = None
        self.dimension = dimension
        self.namespace = namespace
        self.batch_size = 100

    @property
    def vector_store_id(self) -> str:
        """Get the vector store ID.

        Returns:
            Vector store ID
        """
        return self._host

    async def _get_index(self) -> IndexAsyncio:
        """Get the asyncio index client.

        Returns:
            IndexAsyncio instance
        """
        from pinecone import PineconeAsyncio

        if not self._index:
            self._index = PineconeAsyncio(api_key=self.api_key).IndexAsyncio(
                host=self._host
            )
        return self._index

    async def add_vector(
        self,
        vector: np.ndarray,
        metadata: dict[str, Any],
        id_: str | None = None,
    ) -> str:
        """Add single vector to Pinecone.

        Args:
            vector: Vector embedding to store
            metadata: Metadata dictionary for the vector
            id_: Optional ID (generated if not provided)

        Returns:
            ID of the stored vector
        """
        if id_ is None:
            id_ = str(uuid.uuid4())
        vector_list: list[float] = vector.tolist()  # type: ignore
        metadata_copy = prepare_metadata(metadata)

        index = await self._get_index()
        async with index:
            vector_tuple = (id_, vector_list, metadata_copy)
            await index.upsert(vectors=[vector_tuple], namespace=self.namespace)

        return id_

    async def add_vectors(
        self,
        vectors: list[np.ndarray],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add vectors to Pinecone.

        Args:
            vectors: List of vector embeddings
            metadata: List of metadata dictionaries
            ids: Optional list of IDs

        Returns:
            List of IDs for stored vectors
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        vectors_data = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            vector_list: list[float] = vector.tolist()  # type: ignore
            meta_copy = prepare_metadata(meta)
            vectors_data.append((ids[i], vector_list, meta_copy))

        index = await self._get_index()
        async with index:
            for i in range(0, len(vectors_data), self.batch_size):
                batch = vectors_data[i : i + self.batch_size]
                await index.upsert(vectors=batch, namespace=self.namespace)

        return ids

    async def get_vector(self, chunk_id: str) -> Vector | None:
        """Get vector and metadata from Pinecone.

        Args:
            chunk_id: ID of vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None if not
        """
        import numpy as np

        index = await self._get_index()
        async with index:
            result = await index.fetch(ids=[chunk_id], namespace=self.namespace)

        vectors = result.vectors
        if chunk_id not in vectors:
            return None

        vector_data = vectors[chunk_id]
        vector = np.array(vector_data.values)
        metadata = restore_metadata(vector_data.metadata or {})
        return Vector(data=vector, metadata=metadata, id=chunk_id)

    async def delete(self, chunk_id: str) -> bool:
        """Delete vector by ID.

        Args:
            chunk_id: ID of vector to delete

        Returns:
            True if vector was deleted, False if not found
        """
        index = await self._get_index()
        try:
            async with index:
                await index.delete(ids=[chunk_id], namespace=self.namespace)
        except Exception:
            logger.exception("Failed to delete vector %s", chunk_id)
            return False
        else:
            return True

    async def search_vectors(
        self,
        query_vector: np.ndarray,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search Pinecone for similar vectors.

        Args:
            query_vector: Vector to search for
            k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of search results
        """
        vector_list: list[float] = query_vector.tolist()  # type: ignore
        filter_obj = convert_filters(filters) if filters else None

        query_params = {
            "vector": vector_list,
            "top_k": k,
            "namespace": self.namespace,
            "include_metadata": True,
        }

        if filter_obj:
            query_params["filter"] = filter_obj

        index = await self._get_index()
        try:
            async with index:
                results = await index.query(**query_params)
        except Exception:
            logger.exception("Error searching Pinecone")
            return []

        search_results = []
        for match in results.matches:  # pyright: ignore
            raw_metadata = match.metadata or {}
            metadata = restore_metadata(raw_metadata)
            score = match.score or 0.0
            text = metadata.pop("text", None) if isinstance(metadata, dict) else None
            result = SearchResult(
                chunk_id=match.id,
                score=score,
                metadata=metadata,
                text=text,
            )
            search_results.append(result)

        return search_results

    async def close(self):
        """Close the Pinecone connection."""
        self._index = None
