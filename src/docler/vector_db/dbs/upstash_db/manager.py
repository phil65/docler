"""Upstash Vector Store manager."""

from __future__ import annotations

import logging
import os
from typing import Any

from docler.vector_db.dbs.upstash_db.db import UpstashBackend


logger = logging.getLogger(__name__)


class UpstashVectorManager:
    """Manager for Upstash Vector API with namespace support."""

    def __init__(self, url: str | None = None, token: str | None = None):
        """Initialize the Upstash Vector manager.

        Args:
            url: Upstash Vector REST API URL (falls back to UPSTASH_VECTOR_REST_URL)
            token: Upstash Vector API token (falls back to UPSTASH_VECTOR_REST_TOKEN)

        Raises:
            ValueError: If URL or token is not provided or found in environment
        """
        self.url = url or os.getenv("UPSTASH_VECTOR_REST_URL")
        self.token = token or os.getenv("UPSTASH_VECTOR_REST_TOKEN")

        if not self.url:
            msg = (
                "Upstash Vector URL must be provided via parameter"
                " or UPSTASH_VECTOR_REST_URL env var"
            )
            raise ValueError(msg)

        if not self.token:
            msg = (
                "Upstash Vector token must be provided via parameter"
                " or UPSTASH_VECTOR_REST_TOKEN env var"
            )
            raise ValueError(msg)

        from upstash_vector import AsyncIndex

        self._client = AsyncIndex(url=self.url, token=self.token, allow_telemetry=False)
        self._vector_stores: dict[str, UpstashBackend] = {}

    async def list_namespaces(self) -> list[str]:
        """List all available namespaces in the Upstash Vector instance.

        Returns:
            List of namespace names
        """
        try:
            return await self._client.list_namespaces()
        except Exception:
            logger.exception("Error listing Upstash namespaces")
            return []

    async def create_vector_store(
        self,
        name: str,
        dimensions: int = 1536,
    ) -> UpstashBackend:
        """Create a new vector store using namespace.

        Args:
            name: Name for the new vector store (used as namespace)
            dimensions: Vector dimensions

        Returns:
            Configured vector database instance

        Raises:
            ValueError: If creation fails
        """
        if name in self._vector_stores:
            msg = f"Vector store with name '{name}' already exists"
            raise ValueError(msg)

        try:
            # Create and return a configured database instance
            # Note: Upstash automatically creates namespaces when used
            db = UpstashBackend(
                url=self.url,
                token=self.token,
                collection_name=name,
            )

            self._vector_stores[name] = db
        except Exception as e:
            msg = f"Failed to create vector store: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e
        else:
            return db

    async def get_vector_store(self, name: str) -> UpstashBackend:
        """Get a connection to an existing namespace.

        Args:
            name: Name of the namespace to connect to

        Returns:
            Configured vector database instance

        Raises:
            ValueError: If connection fails
        """
        if name in self._vector_stores:
            return self._vector_stores[name]
        try:
            db = UpstashBackend(url=self.url, token=self.token, collection_name=name)
            self._vector_stores[name] = db
        except Exception as e:
            msg = f"Failed to connect to vector store '{name}': {e}"
            logger.exception(msg)
            raise ValueError(msg) from e
        else:
            return db

    async def delete_vector_store(self, name: str) -> bool:
        """Delete all vectors in a namespace.

        Args:
            name: Name of the namespace to clear

        Returns:
            True if successful, False if failed
        """
        try:
            await self._client.delete_namespace(name)
            if name in self._vector_stores:
                del self._vector_stores[name]
        except Exception:
            logger.exception("Error deleting vectors in namespace %s", name)
            return False
        else:
            return True

    async def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the Upstash Vector index.

        Returns:
            Dictionary containing:
            - vector_count: Total number of vectors across all namespaces
            - pending_vector_count: Vectors waiting to be indexed
            - index_size_bytes: Size of the index in bytes
            - dimension: Vector dimension
            - similarity_function: Similarity function used
            - dense_index: Information about dense index if available
            - sparse_index: Information about sparse index if available
            - namespaces: Per-namespace statistics
        """
        try:
            info = await self._client.info()
            stats = {
                "vector_count": info.vector_count,
                "pending_vector_count": info.pending_vector_count,
                "index_size_bytes": info.index_size,
                "dimension": info.dimension,
                "similarity_function": info.similarity_function,
                "namespaces": {},
            }

            if info.dense_index:
                stats["dense_index"] = {
                    "dimension": info.dense_index.dimension,
                    "similarity_function": info.dense_index.similarity_function,
                    "embedding_model": info.dense_index.embedding_model,
                }

            if info.sparse_index:
                stats["sparse_index"] = {
                    "embedding_model": info.sparse_index.embedding_model,
                }

            for namespace_name, namespace_info in info.namespaces.items():
                ns_name = namespace_name or "default"
                stats["namespaces"][ns_name] = {  # type: ignore
                    "vector_count": namespace_info.vector_count,
                    "pending_vector_count": namespace_info.pending_vector_count,
                }

        except Exception:
            logger.exception("Error fetching Upstash vector statistics")
            return {
                "vector_count": 0,
                "pending_vector_count": 0,
                "index_size_bytes": 0,
                "dimension": 0,
                "similarity_function": "unknown",
                "namespaces": {},
            }
        else:
            return stats

    async def search(
        self,
        namespace: str,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search a namespace directly.

        Args:
            namespace: Name of the namespace to search
            query: Query text
            k: Maximum number of results
            filters: Optional filters to apply

        Returns:
            List of search results
        """
        try:
            vector_db = await self.get_vector_store(namespace)
            results = await vector_db.search_text(query, k, filters)
            formatted_results = []
            for result in results:
                result_obj = {
                    "id": result.chunk_id,
                    "text": result.text or "",
                    "score": result.score,
                    **result.metadata,
                }
                formatted_results.append(result_obj)

        except Exception:
            logger.exception("Error searching namespace")
            return []
        else:
            return formatted_results

    async def close(self) -> None:
        """Close all vector store connections."""
        # Clear tracked vector stores
        self._vector_stores.clear()


if __name__ == "__main__":
    import anyenv

    async def main():
        manager = UpstashVectorManager()

        namespaces = await manager.list_namespaces()
        print(f"Available namespaces: {namespaces}")
        db = await manager.create_vector_store("my_namespace")
        await db.add_texts(["Hello", "World"])
        stats = await manager.get_statistics()
        print(f"Vector statistics: {stats}")
        results = await manager.search("my_namespace", "Hello", k=1)
        print(f"Search results: {results}")

    anyenv.run_sync(main())
