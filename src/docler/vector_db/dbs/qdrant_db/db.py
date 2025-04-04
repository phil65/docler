"""Qdrant vector store backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal
import uuid

from docler.log import get_logger
from docler.models import SearchResult, Vector
from docler.process_runner import ProcessRunner
from docler.vector_db.base_backend import VectorStoreBackend
from docler.vector_db.dbs.qdrant_db.utils import (
    get_query,
    to_pointstructs,
    to_search_result,
)


if TYPE_CHECKING:
    import os

    import numpy as np


logger = get_logger(__name__)
Metric = Literal["cosine", "euclidean", "dotproduct", "manhattan"]


class QdrantBackend(VectorStoreBackend):
    """Qdrant implementation of vector store backend."""

    REQUIRED_PACKAGES: ClassVar = {"qdrant-client"}

    def __init__(
        self,
        collection_name: str = "default",
        metric: Metric = "cosine",
        location: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        vector_size: int = 1536,
        prefer_grpc: bool = True,
    ):
        """Initialize Qdrant backend."""
        from qdrant_client import AsyncQdrantClient, QdrantClient
        from qdrant_client.http import models

        client_kwargs: dict[str, Any] = {"prefer_grpc": prefer_grpc}
        if url:
            client_kwargs["url"] = url
            if api_key:
                client_kwargs["api_key"] = api_key
        elif location:
            client_kwargs["location"] = location
        else:
            client_kwargs["location"] = ":memory:"
        self._client = AsyncQdrantClient(**client_kwargs)
        self._collection_name = collection_name

        temp_client = QdrantClient(**client_kwargs)
        collections = temp_client.get_collections().collections
        collection_names = [c.name for c in collections]
        metric_map = dict(
            cosine=models.Distance.COSINE,
            euclidean=models.Distance.EUCLID,
            dotproduct=models.Distance.DOT,
            manhattan=models.Distance.MANHATTAN,
        )
        if self._collection_name not in collection_names:
            cfg = models.VectorParams(size=vector_size, distance=metric_map[metric])
            temp_client.create_collection(self._collection_name, vectors_config=cfg)

    @staticmethod
    def run_server(path: str | os.PathLike[str]) -> ProcessRunner:
        args = [
            "docker",
            "run",
            "-p",
            "6333:6333",
            "-p",
            "6334:6334",
            "-v",
            "$(pwd)/qdrant_storage:/qdrant/storage:z",
            "qdrant/qdrant",
        ]
        return ProcessRunner(args)

    async def add_vectors(
        self,
        vectors: list[np.ndarray],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add vectors to Qdrant."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        points = to_pointstructs(vectors, metadata, ids)
        await self._client.upsert(collection_name=self._collection_name, points=points)
        return ids

    async def get_vector(self, chunk_id: str) -> Vector | None:
        """Get a vector and its metadata by ID."""
        import numpy as np

        points = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=[chunk_id],
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            return None
        point = points[0]
        data = np.array(point.vector)
        return Vector(data=data, metadata=point.payload or {}, id=str(point.id))

    async def list_vector_ids(
        self,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[str | int]:
        records, _ = await self._client.scroll(
            collection_name=self._collection_name,
            limit=limit or 999999,
            with_payload=False,
            with_vectors=False,
        )
        return [record.id for record in records]

    async def delete(self, chunk_id: str) -> bool:
        """Delete vector by ID."""
        from qdrant_client.http import models

        try:
            selector = models.PointIdsList(points=[chunk_id])
            await self._client.delete(self._collection_name, points_selector=selector)
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
        """Search Qdrant for similar vectors."""
        vector_list = query_vector.astype(float).tolist()
        results = await self._client.search(
            collection_name=self._collection_name,
            query_vector=vector_list,  # type: ignore
            limit=k,
            with_payload=True,
            filter=get_query(filters),
        )
        return [to_search_result(i) for i in results]

    async def close(self):
        """Close the Qdrant connection."""
        await self._client.close()


if __name__ == "__main__":
    db = QdrantBackend(url="http://localhost:6333", collection_name="test")
