"""Pinecone Vector Store manager with asyncio support."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, cast

from pydantic import SecretStr

from docler.configs.vector_db_configs import PineconeCloud, PineconeConfig, PineconeRegion
from docler.utils import get_api_key
from docler.vector_db.base import BaseVectorDB
from docler.vector_db.base_manager import VectorManagerBase
from docler.vector_db.dbs.pinecone_db.db import PineconeBackend
from docler.vector_db.dbs.pinecone_db.utils import to_vector_store_info


if TYPE_CHECKING:
    from docler.models import VectorStoreInfo


Metric = Literal["cosine", "euclidean", "dotproduct"]


class PineconeVectorManager(VectorManagerBase[PineconeConfig]):
    """Manager for Pinecone Vector Stores with asyncio support."""

    Config = PineconeConfig
    NAME = "pinecone"
    REQUIRED_PACKAGES: ClassVar = {"pinecone-client"}

    def __init__(self, api_key: str | None = None):
        """Initialize the Pinecone Vector Store manager."""
        super().__init__()
        self.api_key = api_key or get_api_key("PINECONE_API_KEY")

    @classmethod
    def from_config(cls, config: PineconeConfig) -> PineconeVectorManager:
        """Create instance from configuration."""
        key = config.api_key.get_secret_value() if config.api_key else None
        return cls(api_key=key)

    def to_config(self) -> PineconeConfig:
        """Extract configuration from instance."""
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
            return [to_vector_store_info(idx) for idx in indexes]

    async def create_vector_store(
        self,
        name: str,
        dimension: int = 1536,
        metric: Metric = "cosine",
        cloud: PineconeCloud = "aws",
        region: PineconeRegion = "us-east-1",
        namespace: str = "default",
        **kwargs,
    ) -> BaseVectorDB:
        """Create a new vector store."""
        from pinecone import PineconeAsyncio, ServerlessSpec

        if await self.has_vector_store(name):
            msg = f"Index {name!r} already exists"
            raise ValueError(msg)
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
            return cast(BaseVectorDB, db)  # type: ignore[override]

    async def get_vector_store(self, name: str, **kwargs) -> BaseVectorDB:
        """Get a connection to an existing vector store."""
        from pinecone import PineconeAsyncio

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

        except Exception as e:
            msg = f"Failed to connect to vector store {name}: {e}"
            self.logger.exception(msg)
            raise ValueError(msg) from e
        else:
            return cast(BaseVectorDB, db)

    async def delete_vector_store(self, name: str) -> bool:
        """Delete a vector store."""
        from pinecone import PineconeAsyncio

        try:
            if not await self.has_vector_store(name):
                return False
            async with PineconeAsyncio(api_key=self.api_key) as client:
                index_info = await client.describe_index(name)
                if index_info.deletion_protection == "enabled":
                    await client.configure_index(name, deletion_protection="disabled")
                await client.delete_index(name)

        except Exception:
            self.logger.exception("Error deleting vector store %s", name)
            return False
        else:
            return True

    async def close(self):
        """Close all vector store connections."""


if __name__ == "__main__":
    import anyenv

    async def main():
        manager = PineconeVectorManager()
        indexes = await manager.list_vector_stores()
        print(indexes)
        await manager.close()

    anyenv.run_sync(main())
