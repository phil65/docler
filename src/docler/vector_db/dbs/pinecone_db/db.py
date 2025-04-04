"""Pinecone vector store backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
import uuid

from docler.log import get_logger
from docler.models import SearchResult, Vector
from docler.utils import get_api_key
from docler.vector_db.base_backend import VectorStoreBackend
from docler.vector_db.dbs.pinecone_db.utils import (
    convert_filters,
    prepare_metadata,
    restore_metadata,
    to_search_result,
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
        api_key: str | None = None,
        dimension: int = 1536,
        namespace: str = "default",
    ):
        """Initialize Pinecone backend."""
        self.api_key = api_key or get_api_key("PINECONE_API_KEY")
        self._host = host
        self._index: IndexAsyncio | None = None
        self.dimension = dimension
        self.namespace = namespace
        self.batch_size = 100

    @property
    def vector_store_id(self) -> str:
        """Get the vector store ID."""
        return self._host

    async def _get_index(self) -> IndexAsyncio:
        """Get the asyncio index client."""
        from pinecone import PineconeAsyncio

        if not self._index:
            client = PineconeAsyncio(api_key=self.api_key)
            self._index = client.IndexAsyncio(host=self._host)
        return self._index

    async def list_vector_ids(
        self,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        index = await self._get_index()
        result = await index.list_paginated(namespace=namespace, limit=limit)
        return [i["id"] for i in result["vectors"]]

    async def add_vectors(
        self,
        vectors: list[np.ndarray],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add vectors to Pinecone."""
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
        """Get vector and metadata from Pinecone."""
        import numpy as np

        async with (index := await self._get_index()):
            result = await index.fetch(ids=[chunk_id], namespace=self.namespace)

        vectors = result.vectors
        if chunk_id not in vectors:
            return None

        vector_data = vectors[chunk_id]
        vector = np.array(vector_data.values)
        metadata = restore_metadata(vector_data.metadata or {})
        return Vector(data=vector, metadata=metadata, id=chunk_id)

    async def delete(self, chunk_id: str) -> bool:
        """Delete vector by ID."""
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
        """Search Pinecone for similar vectors."""
        vector_list: list[float] = query_vector.tolist()  # type: ignore
        filter_obj = convert_filters(filters) if filters else None
        index = await self._get_index()
        try:
            async with index:
                results = await index.query(
                    vector=vector_list,
                    top_k=k,
                    namespace=self.namespace,
                    include_metadata=True,
                    filter=filter_obj,
                )
        except Exception:
            logger.exception("Error searching Pinecone")
            return []
        return [to_search_result(i) for i in results.matches]  # type: ignore

    async def close(self):
        """Close the Pinecone connection."""
        self._index = None


if __name__ == "__main__":

    async def main():
        db = PineconeBackend(host="https://test-y8nq1hj.svc.aped-4627-b74a.pinecone.io")
        data = await db.list_vector_ids()
        print(data)

    import asyncio

    asyncio.run(main())
