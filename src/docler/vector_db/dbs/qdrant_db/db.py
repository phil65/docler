"""Qdrant vector store backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal
import uuid

from docler.log import get_logger
from docler.models import SearchResult, Vector
from docler.vector_db.base import VectorStoreBackend
from docler.vector_db.dbs.qdrant_db.utils import get_query


if TYPE_CHECKING:
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
        """Initialize Qdrant backend.

        Args:
            collection_name: Name of collection to use
            metric: Metric to use for similarity search
            location: Path to local storage (memory if None)
            url: URL of Qdrant server (overrides location)
            api_key: API key for Qdrant cloud
            vector_size: Size of vectors to store
            prefer_grpc: Whether to prefer gRPC over HTTP
        """
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

    async def add_vector(
        self,
        vector: np.ndarray,
        metadata: dict[str, Any],
        id_: str | None = None,
    ) -> str:
        """Add single vector to Qdrant.

        Args:
            vector: Vector embedding to store
            metadata: Metadata dictionary
            id_: Optional ID (generated if not provided)

        Returns:
            ID of the stored vector
        """
        from qdrant_client.http import models

        if id_ is None:
            id_ = str(uuid.uuid4())
        vector_list = vector.astype(float).tolist()
        point = models.PointStruct(id=id_, vector=vector_list, payload=metadata)
        _result = await self._client.upsert(self._collection_name, points=[point])
        return id_

    async def add_vectors(
        self,
        vectors: list[np.ndarray],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add vectors to Qdrant.

        Args:
            vectors: List of vector embeddings
            metadata: List of metadata dictionaries
            ids: Optional list of IDs

        Returns:
            List of IDs for stored vectors
        """
        from qdrant_client.http import models

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        points = []
        for i, vector in enumerate(vectors):
            # Convert to float64 then to list to ensure compatibility
            vector_ls = vector.astype(float).tolist()
            struct = models.PointStruct(id=ids[i], vector=vector_ls, payload=metadata[i])
            points.append(struct)

        await self._client.upsert(collection_name=self._collection_name, points=points)
        return ids

    async def get_vector(self, chunk_id: str) -> Vector | None:
        """Get a vector and its metadata by ID.

        Args:
            chunk_id: ID of vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None if not
        """
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

    async def delete(self, chunk_id: str) -> bool:
        """Delete vector by ID.

        Args:
            chunk_id: ID of vector to delete

        Returns:
            True if vector was deleted, False if not found
        """
        from qdrant_client.http import models
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            selector = models.PointIdsList(points=[chunk_id])
            await self._client.delete(self._collection_name, points_selector=selector)
        except UnexpectedResponse:
            return False
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
        """Search Qdrant for similar vectors.

        Args:
            query_vector: Vector to search for
            k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of search results
        """
        vector_list = query_vector.astype(float).tolist()
        results = await self._client.search(
            collection_name=self._collection_name,
            query_vector=vector_list,  # type: ignore
            limit=k,
            with_payload=True,
            filter=get_query(filters),
        )

        search_results = []
        for hit in results:
            data = hit.payload or {}
            text = data.pop("text", None) if data else None
            txt = str(text) if text is not None else None
            id_ = str(hit.id)
            result = SearchResult(chunk_id=id_, score=hit.score, metadata=data, text=txt)
            search_results.append(result)

        return search_results

    async def close(self):
        """Close the Qdrant connection."""
        await self._client.close()


if __name__ == "__main__":
    db = QdrantBackend(url="http://localhost:6333", collection_name="test")
