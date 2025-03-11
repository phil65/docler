"""RAG-based tool registry for semantic tool discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import uuid

from llmling_agent.log import get_logger
from llmling_agent.vector_db import SearchResult

from docler.vector_db.base import VectorStoreBackend


if TYPE_CHECKING:
    import numpy as np


logger = get_logger(__name__)


class QdrantBackend(VectorStoreBackend):
    """Qdrant implementation of vector store backend."""

    def __init__(
        self,
        collection_name: str = "default",
        location: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        vector_size: int = 1536,
        prefer_grpc: bool = True,
    ):
        """Initialize Qdrant backend.

        Args:
            collection_name: Name of collection to use
            location: Path to local storage (memory if None)
            url: URL of Qdrant server (overrides location)
            api_key: API key for Qdrant cloud
            vector_size: Size of vectors to store
            prefer_grpc: Whether to prefer gRPC over HTTP
        """
        import qdrant_client
        from qdrant_client.http import models

        # Create client based on configuration
        client_kwargs: dict[str, Any] = {"prefer_grpc": prefer_grpc}
        if url:
            client_kwargs["url"] = url
            if api_key:
                client_kwargs["api_key"] = api_key
        elif location:
            client_kwargs["location"] = location
        else:
            client_kwargs["location"] = ":memory:"

        self._client = qdrant_client.QdrantClient(**client_kwargs)
        self._collection_name = collection_name

        # Check if collection exists
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        # Create collection if it doesn't exist
        if self._collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )

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
        import anyenv
        from qdrant_client.http import models

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        # Convert numpy arrays to lists and create points
        points = []
        for i, vector in enumerate(vectors):
            # Convert to float64 then to list to ensure compatibility
            vector_list = vector.astype(float).tolist()

            points.append(
                models.PointStruct(id=ids[i], vector=vector_list, payload=metadata[i])
            )

        # Upsert vectors
        await anyenv.run_in_thread(
            self._client.upsert, collection_name=self._collection_name, points=points
        )

        return ids

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
        import anyenv
        from qdrant_client.http import models

        # Convert numpy to list
        vector_list = query_vector.astype(float).tolist()

        # Build filter if needed
        filter_query = None
        if filters:
            conditions = []
            for field_name, value in filters.items():
                if isinstance(value, list):
                    # Handle list values
                    conditions.append(
                        models.FieldCondition(
                            key=field_name,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    # Handle single values
                    conditions.append(
                        models.FieldCondition(
                            key=field_name,
                            match=models.MatchValue(value=value),
                        )
                    )

            if conditions:
                filter_query = models.Filter(must=conditions)

        # Execute search
        results = await anyenv.run_in_thread(
            self._client.search,
            collection_name=self._collection_name,
            query_vector=vector_list,
            limit=k,
            with_payload=True,
            filter=filter_query,
        )

        # Format results
        search_results = []
        for hit in results:
            payload = hit.payload or {}
            result = SearchResult(
                doc_id=str(hit.id),
                score=hit.score,
                metadata=payload,
                text=payload.get("text"),
            )
            search_results.append(result)

        return search_results

    async def delete(self, chunk_id: str) -> bool:
        """Delete vector by ID.

        Args:
            chunk_id: ID of vector to delete

        Returns:
            True if vector was deleted, False otherwise
        """
        import anyenv
        from qdrant_client.http import models
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            # Create selector for the point ID
            selector = models.PointIdsList(points=[chunk_id])

            # Delete the point
            await anyenv.run_in_thread(
                self._client.delete,
                collection_name=self._collection_name,
                points_selector=selector,
            )
        except UnexpectedResponse:
            # If point not found or other error
            return False
        except Exception:  # noqa: BLE001
            # Any other exception
            return False
        else:
            return True
