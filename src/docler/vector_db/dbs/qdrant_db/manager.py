"""Qdrant Vector Store manager with asyncio support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import HttpUrl, SecretStr

from docler.configs.vector_db_configs import QdrantConfig
from docler.process_runner import ProcessRunner
from docler.vector_db.base import BaseVectorDB
from docler.vector_db.base_manager import VectorManagerBase
from docler.vector_db.dbs.qdrant_db.db import QdrantBackend
from docler.vector_db.dbs.qdrant_db.utils import get_distance


if TYPE_CHECKING:
    from docler.common_types import StrPath
    from docler.models import VectorStoreInfo

Metric = Literal["cosine", "euclidean", "dotproduct", "manhattan"]
SERVER_ARGS = [
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


class QdrantVectorManager(VectorManagerBase[QdrantConfig]):
    """Manager for Qdrant Vector Stores with asyncio support."""

    Config = QdrantConfig
    NAME = "qdrant"
    REQUIRED_PACKAGES: ClassVar = {"qdrant-client"}

    def __init__(
        self,
        *,
        location: str | None = None,
        url: HttpUrl | str | None = None,
        api_key: str | None = None,
        prefer_grpc: bool = True,
    ):
        """Initialize the Qdrant Vector Store manager.

        Args:
            location: Path to local Qdrant storage
            url: URL for Qdrant server
            api_key: API key for Qdrant cloud
            prefer_grpc: Whether to prefer gRPC over HTTP
        """
        from qdrant_client import AsyncQdrantClient

        super().__init__()
        self.location = location
        self.url = str(url) if url else None
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc

        client_kwargs: dict[str, Any] = {}
        if self.url:
            client_kwargs["url"] = self.url
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
        elif self.location:
            client_kwargs["location"] = self.location
        else:
            client_kwargs["location"] = ":memory:"

        self._client = AsyncQdrantClient(prefer_grpc=self.prefer_grpc, **client_kwargs)

    @classmethod
    def from_config(cls, config: QdrantConfig) -> QdrantVectorManager:
        """Create instance from configuration."""
        api_key = config.api_key.get_secret_value() if config.api_key else None
        return cls(
            location=config.location,
            url=str(config.url),
            api_key=api_key,
            prefer_grpc=config.prefer_grpc,
        )

    def to_config(self) -> QdrantConfig:
        """Extract configuration from instance."""
        return QdrantConfig(
            location=self.location,
            url=HttpUrl(self.url) if self.url else None,
            api_key=SecretStr(self.api_key) if self.api_key else None,
            prefer_grpc=self.prefer_grpc,
        )

    @staticmethod
    def run_server(path: StrPath, **kwargs) -> ProcessRunner:
        """Run a Qdrant server using Docker.

        Args:
            path: Path for persistent storage
            **kwargs: Additional arguments for ProcessRunner

        Returns:
            A ProcessRunner instance
        """
        import os
        import platform

        # Handle Windows path conversion
        if platform.system() == "Windows":
            # Convert Windows path to Docker-compatible path
            if os.path.isabs(path):  # noqa: PTH117
                volume_path = str(path).replace("\\", "/")
                # If it's a drive letter, convert C:\ to /c/
                if ":" in volume_path:
                    drive, rest = volume_path.split(":", 1)
                    volume_path = f"/{drive.lower()}{rest}"
            else:
                # For relative paths
                volume_path = "${PWD}/" + str(path).replace("\\", "/")

            cmd = f"docker run -p 6333:6333 -p 6334:6334 -v {volume_path}:/qdrant/storage qdrant/qdrant"  # noqa: E501
        else:
            # Unix-style command
            cmd = f"docker run -p 6333:6333 -p 6334:6334 -v {path}:/qdrant/storage:z qdrant/qdrant"  # noqa: E501

        return ProcessRunner(cmd, wait_tcp=[("localhost", 6333)], **kwargs)

    @property
    def name(self) -> str:
        """Name of this vector database provider."""
        return self.NAME

    async def list_vector_stores(self) -> list[VectorStoreInfo]:
        """List all available vector stores (collections) for this provider."""
        from docler.models import VectorStoreInfo

        try:
            collections = await self._client.get_collections()
            result = []

            for c in collections.collections:
                metadata = {"name": c.name}
                info = VectorStoreInfo(db_id=c.name, name=c.name, metadata=metadata)
                result.append(info)
        except Exception:
            self.logger.exception("Error listing Qdrant collections")
            return []
        else:
            return result

    async def create_vector_store(
        self,
        name: str,
        vector_size: int = 1536,
        metric: Metric = "cosine",
        **kwargs,
    ) -> BaseVectorDB:
        """Create a new vector store (collection)."""
        from qdrant_client.http import models

        try:
            collections_list = await self._client.get_collections()
            collection_names = [c.name for c in collections_list.collections]

            if name not in collection_names:
                distance = get_distance(metric)
                params = models.VectorParams(size=vector_size, distance=distance)
                await self._client.create_collection(name, vectors_config=params)
            db = QdrantBackend(
                collection_name=name,
                location=self.location,
                url=self.url,
                api_key=self.api_key,
                prefer_grpc=self.prefer_grpc,
                vector_size=vector_size,
                metric=metric,
            )
            return cast(BaseVectorDB, db)

        except Exception as e:
            msg = f"Failed to create Qdrant collection: {e}"
            self.logger.exception(msg)
            raise ValueError(msg) from e

    async def get_vector_store(
        self,
        name: str,
        **kwargs,
    ) -> BaseVectorDB:
        """Get a connection to an existing collection."""
        from qdrant_client.http.models import VectorParams

        collection_info = await self._client.get_collection(name)
        vector_size = kwargs.get("vector_size", 1536)
        metric: Metric = "cosine"
        match collection_info.config.params.vectors:
            # case dict() as dct:
            #     vector_size = dct["xyz"].size
            case VectorParams() as params:
                vector_size = params.size
                distance_str = str(params.distance).lower()
                if distance_str == "cosine":
                    metric = "cosine"
                elif distance_str in ("euclid", "euclidean"):
                    metric = "euclidean"
                elif distance_str in ("dot", "dotproduct"):
                    metric = "dotproduct"
                elif distance_str == "manhattan":
                    metric = "manhattan"
            case _:
                pass
        db = QdrantBackend(
            collection_name=name,
            location=self.location,
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=self.prefer_grpc,
            vector_size=vector_size,
            metric=metric,
        )

        return cast(BaseVectorDB, db)

    async def delete_vector_store(self, name: str) -> bool:
        """Delete a vector store (collection)."""
        try:
            await self._client.delete_collection(collection_name=name)
        except Exception:
            self.logger.exception("Failed to delete collection %s", name)
            return False
        else:
            return True

    async def close(self):
        """Close all vector store connections."""
        await self._client.close()


if __name__ == "__main__":
    import asyncio

    async def main():
        # async with QdrantVectorManager.run_server("test"):
        manager = QdrantVectorManager(url="http://localhost:6333")
        store = await manager.create_vector_store("test-store")
        print(f"Created vector store: {store.vector_store_id}")
        stores = await manager.list_vector_stores()
        print(f"Available stores: {[s.name for s in stores]}")
        store = await manager.get_vector_store("test-store")
        success = await manager.delete_vector_store("test-store")
        print(f"Deleted store: {success}")
        await manager.close()

    asyncio.run(main())
