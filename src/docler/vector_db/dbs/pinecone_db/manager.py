"""Pinecone Vector Store manager with asyncio support."""

from __future__ import annotations

from typing import ClassVar, Literal, cast

from docler.configs.vector_db_configs import PineconeCloud, PineconeConfig, PineconeRegion
from docler.log import get_logger
from docler.models import VectorStoreInfo
from docler.utils import get_api_key
from docler.vector_db.base import VectorDB
from docler.vector_db.base_manager import VectorManagerBase
from docler.vector_db.dbs.pinecone_db.db import PineconeBackend


Metric = Literal["cosine", "euclidean", "dotproduct"]

logger = get_logger(__name__)


class PineconeVectorManager(VectorManagerBase[PineconeConfig]):
    """Manager for Pinecone Vector Stores with asyncio support."""

    Config = PineconeConfig
    NAME = "pinecone"
    REQUIRED_PACKAGES: ClassVar = {"pinecone-client"}

    def __init__(self, api_key: str | None = None):
        """Initialize the Pinecone Vector Store manager."""
        self.api_key = api_key or get_api_key("PINECONE_API_KEY")
        self._vector_stores: dict[str, PineconeBackend] = {}

    @classmethod
    def from_config(cls, config: PineconeConfig) -> PineconeVectorManager:
        """Create instance from configuration."""
        key = config.api_key.get_secret_value() if config.api_key else None
        return cls(api_key=key)

    def to_config(self) -> PineconeConfig:
        """Extract configuration from instance."""
        from pydantic import SecretStr

        return PineconeConfig(api_key=SecretStr(self.api_key) if self.api_key else None)

    @property
    def name(self) -> str:
        """Name of this vector database provider."""
        return self.NAME

    async def list_vector_stores(self) -> list[VectorStoreInfo]:
        """List all available vector stores for this provider."""
        from pinecone import PineconeAsyncio

        async with PineconeAsyncio(api_key=self.api_key) as client:
            indexes = await client.list_indexes()
            return [
                VectorStoreInfo(
                    db_id=idx.host,
                    name=idx.name,
                    metadata=dict(
                        dimension=idx.dimension,
                        metric=idx.metric,
                        status=idx.status.state if idx.status else None,
                        ready=idx.status.ready if idx.status else False,
                        vector_type=idx.vector_type,
                        tags=idx.tags,
                    ),
                )
                for idx in indexes
            ]

    async def create_vector_store(
        self,
        name: str,
        dimension: int = 1536,
        metric: Metric = "cosine",
        cloud: PineconeCloud = "aws",
        region: PineconeRegion = "us-east-1",
        namespace: str = "default",
        **kwargs,
    ) -> VectorDB:
        """Create a new vector store.

        Args:
            name: Name for the new index
            dimension: Dimension of vectors to store
            metric: Distance metric for similarity search
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
            namespace: Namespace for the index
            **kwargs: Additional parameters

        Returns:
            Configured vector database instance

        Raises:
            ValueError: If creation fails
        """
        from pinecone import PineconeAsyncio

        try:
            from pinecone import ServerlessSpec

            if await self.has_vector_store(name):
                msg = f"Index {name!r} already exists"
                raise ValueError(msg)  # noqa: TRY301
            async with PineconeAsyncio(api_key=self.api_key) as client:
                spec = ServerlessSpec(cloud=cloud.lower(), region=region)
                await client.create_index(name, spec, dimension=dimension, metric=metric)
                index_info = await client.describe_index(name)
                db = PineconeBackend(
                    api_key=self.api_key,
                    host=index_info.host,
                    dimension=dimension,
                    namespace=namespace,
                )
                self._vector_stores[name] = db
                return cast(VectorDB, db)  # type: ignore[override]

        except Exception as e:
            msg = f"Failed to create vector store: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e

    async def get_vector_store(self, name: str, **kwargs) -> VectorDB:
        """Get a connection to an existing vector store."""
        from pinecone import PineconeAsyncio

        if name in self._vector_stores:
            return cast(VectorDB, self._vector_stores[name])

        try:
            if not await self.has_vector_store(name):
                msg = f"Index {name} does not exist"
                raise ValueError(msg)  # noqa: TRY301
            async with PineconeAsyncio(api_key=self.api_key) as client:
                index_info = await client.describe_index(name)
            db = PineconeBackend(
                api_key=self.api_key,
                host=index_info.host,
                dimension=index_info.dimension,
                namespace=kwargs.get("namespace", "default"),
            )
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
        from pinecone import PineconeAsyncio

        try:
            if not await self.has_vector_store(name):
                return False
            async with PineconeAsyncio(api_key=self.api_key) as client:
                index_info = await client.describe_index(name)
                if index_info.deletion_protection == "enabled":
                    await client.configure_index(name, deletion_protection="disabled")
                await client.delete_index(name)

            if name in self._vector_stores:
                await self._vector_stores[name].close()
                del self._vector_stores[name]
        except Exception:
            logger.exception("Error deleting vector store %s", name)
            return False
        else:
            return True

    async def close(self) -> None:
        """Close all vector store connections."""
        for db in self._vector_stores.values():
            await db.close()
        self._vector_stores.clear()


if __name__ == "__main__":
    import anyenv

    async def main():
        manager = PineconeVectorManager()
        store = await manager.create_vector_store("test-store")
        print(store)
        indexes = await manager.list_vector_stores()
        print(indexes)
        await manager.close()

    anyenv.run_sync(main())
