"""Pinecone Vector Store manager."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal, cast

import anyio

from docler.vector_db.base import VectorDB
from docler.vector_db.base_manager import VectorManagerBase
from docler.vector_db.dbs.pinecone_db.db import PineconeBackend


if TYPE_CHECKING:
    from pinecone import IndexModel


Metric = Literal["cosine", "euclidean", "dotproduct"]

logger = logging.getLogger(__name__)


class PineconeVectorManager(VectorManagerBase):
    """Manager for Pinecone Vector Stores."""

    def __init__(self, api_key: str | None = None):
        """Initialize the Pinecone Vector Store manager.

        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            msg = "Pinecone API key must be provided via param or PINECONE_API_KEY envvar"
            raise ValueError(msg)

        try:
            from pinecone import Pinecone
        except ImportError as e:
            msg = 'Pinecone is not installed. Please install "pinecone[asyncio]"'
            raise ImportError(msg) from e
        self._client = Pinecone(api_key=self.api_key)
        self._vector_stores: dict[str, PineconeBackend] = {}

    @property
    def name(self) -> str:
        """Name of this vector database provider."""
        return "pinecone"

    async def list_vector_stores(self) -> list[dict[str, Any]]:
        """List all available vector stores for this provider."""
        try:
            indexes = self._client.list_indexes()
            return [
                {
                    "name": idx["name"],
                    "dimension": idx.get("dimension"),
                    "metric": idx.get("metric"),
                    "status": idx.get("status", {}).get("state"),
                    "ready": idx.get("status", {}).get("ready", False),
                    "host": idx.get("host"),
                }
                for idx in indexes.get("indexes", [])
            ]
        except Exception:
            logger.exception("Error listing Pinecone indexes")
            return []

    # Keep the list_indexes method as a convenience alias
    async def list_indexes(self) -> list[dict[str, Any]]:
        """List all indexes available in the Pinecone account."""
        return await self.list_vector_stores()

    async def create_vector_store(  # type: ignore
        self,
        name: str,
        dimension: int = 1536,
        metric: Metric = "cosine",
        cloud: str = "aws",
        region: str = "us-west-2",
    ) -> VectorDB:
        """Create a new vector store.

        Args:
            name: Name for the new index
            dimension: Dimension of vectors to store
            metric: Distance metric for similarity search
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region

        Returns:
            Configured vector database instance

        Raises:
            ValueError: If creation fails
        """
        try:
            from pinecone import ServerlessSpec

            indexes = self._client.list_indexes()
            index_names = [idx["name"] for idx in indexes.get("indexes", [])]
            if name not in index_names:
                self._client.create_index(
                    name=name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
                index_info = None
                for _ in range(30):  # Wait up to 30 seconds
                    try:
                        index_info = self._client.describe_index(name)
                        if index_info["status"]["ready"]:
                            break
                    except Exception:  # noqa: BLE001
                        pass
                    await anyio.sleep(1)

                if not index_info or not index_info["status"]["ready"]:
                    msg = f"Index {name} creation timed out or failed"
                    raise ValueError(msg)  # noqa: TRY301
            else:
                index_info = self._client.describe_index(name)

            db = await self._get_backend(index_info)

            self._vector_stores[name] = db
        except Exception as e:
            msg = f"Failed to create vector store: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e
        else:
            return cast(VectorDB, db)  # type: ignore[override]  # type: ignore[override]

    async def get_vector_store(self, name: str, **kwargs) -> VectorDB:
        """Get a connection to an existing vector store.

        Args:
            name: Name of the existing index
            kwargs: Additional keyword arguments for the vector store

        Returns:
            Configured vector database instance

        Raises:
            ValueError: If store doesn't exist or connection fails
        """
        if name in self._vector_stores:
            return cast(VectorDB, self._vector_stores[name])

        try:
            # Check if index exists
            indexes = self._client.list_indexes()
            index_names = [idx["name"] for idx in indexes.get("indexes", [])]

            if name not in index_names:
                msg = f"Index {name} does not exist"
                raise ValueError(msg)  # noqa: TRY301

            index_info = self._client.describe_index(name)
            db = await self._get_backend(index_info)
            self._vector_stores[name] = db
        except Exception as e:
            msg = f"Failed to connect to vector store {name}: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e
        else:
            return cast(VectorDB, db)

    async def delete_vector_store(self, name: str) -> bool:
        """Delete a vector store.

        Args:
            name: Name of the index to delete

        Returns:
            True if successful, False if failed
        """
        try:
            indexes = self._client.list_indexes()
            index_names = [idx["name"] for idx in indexes.get("indexes", [])]
            if name not in index_names:
                return False

            # Check if deletion protection is enabled
            index_info = self._client.describe_index(name)
            if index_info.get("deletion_protection", "") == "enabled":
                # Disable deletion protection
                self._client.configure_index(name=name, deletion_protection="disabled")

            self._client.delete_index(name)
            if name in self._vector_stores:
                await self._vector_stores[name].close()
                del self._vector_stores[name]

        except Exception:
            logger.exception("Error deleting vector store %s", name)
            return False
        else:
            return True

    async def _get_backend(
        self, index_info: dict[str, Any] | IndexModel
    ) -> PineconeBackend:
        """Create a backend instance for an index.

        Args:
            index_info: Index information from describe_index

        Returns:
            Configured vector database backend
        """
        host = index_info.get("host")
        if not host:
            msg = "Index host not found in index information"
            raise ValueError(msg)

        return PineconeBackend(
            pinecone_client=self._client,
            host=host,
            dimension=index_info.get("dimension", 1536),
            namespace="default",
        )

    async def close(self) -> None:
        """Close all vector store connections."""
        for db in self._vector_stores.values():
            await db.close()
        self._vector_stores.clear()


if __name__ == "__main__":
    import anyenv

    async def main():
        manager = PineconeVectorManager()
        indexes = await manager.list_indexes()
        print(indexes)
        await manager.close()

    anyenv.run_sync(main())
