"""ChromaDB vector store manager implementation."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import TYPE_CHECKING, ClassVar, cast

from docler.configs.vector_db_configs import ChromaConfig
from docler.models import VectorStoreInfo
from docler.process_runner import ProcessRunner
from docler.vector_db.base import BaseVectorDB
from docler.vector_db.base_manager import VectorManagerBase
from docler.vector_db.dbs.chroma_db.db import ChromaBackend


if TYPE_CHECKING:
    import chromadb

    from docler.common_types import StrPath


class ChromaVectorManager(VectorManagerBase[ChromaConfig]):
    """Manager for ChromaDB vector stores with fully async implementation."""

    Config = ChromaConfig
    NAME = "chroma"
    REQUIRED_PACKAGES: ClassVar = {"chromadb"}

    def __init__(
        self,
        *,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int = 8000,
        ssl: bool = False,
        headers: dict[str, str] | None = None,
    ):
        """Initialize the ChromaDB vector store manager.

        Args:
            persist_directory: Directory for persistent storage
            host: Hostname for remote ChromaDB server
            port: Port for remote ChromaDB server
            ssl: Whether to use SSL for server connection
            headers: Optional headers for server connection
        """
        super().__init__()
        self.persist_directory = persist_directory
        self.host = host or "localhost"
        self.port = port
        self.ssl = ssl
        self.headers = headers
        self._list_client: chromadb.AsyncClientAPI | None = None

    @staticmethod
    def run_server(path: StrPath, **kwargs) -> ProcessRunner:
        cmd = f"chroma run {path}"
        # if self.port:
        #     cmd += f" --port {self.port}"
        return ProcessRunner(cmd, **kwargs)

    @property
    def name(self) -> str:
        """Name of this vector database provider."""
        return self.NAME

    @classmethod
    def from_config(cls, config: ChromaConfig) -> ChromaVectorManager:
        """Create instance from configuration."""
        return cls(persist_directory=config.persist_directory)

    def to_config(self) -> ChromaConfig:
        """Extract configuration from instance."""
        return ChromaConfig(persist_directory=self.persist_directory)

    async def _get_list_client(self) -> chromadb.AsyncClientAPI:
        """Get a client for listing collections."""
        import chromadb

        if not self._list_client:
            self._list_client = await chromadb.AsyncHttpClient(
                host=self.host,
                port=self.port,
                ssl=self.ssl,
                headers=self.headers,
            )
        return self._list_client

    async def list_vector_stores(self) -> list[VectorStoreInfo]:
        """List all available vector stores (collections) for this provider."""
        try:
            client = await self._get_list_client()
            collections = await client.list_collections()
            return [VectorStoreInfo(db_id=c.name, name=c.name) for c in collections]
        except Exception:
            self.logger.exception("Error listing ChromaDB collections")
            return []

    async def create_vector_store(self, name: str, **kwargs) -> BaseVectorDB:
        """Create a new vector store (collection)."""
        try:
            db = ChromaBackend(
                vector_store_id=name,
                persist_directory=self.persist_directory,
                **kwargs,
            )
            return cast(BaseVectorDB, db)

        except Exception as e:
            msg = f"Failed to create ChromaDB collection: {e}"
            self.logger.exception(msg)
            raise ValueError(msg) from e

    async def get_vector_store(
        self,
        name: str,
        **kwargs,
    ) -> BaseVectorDB:
        """Get a connection to an existing collection."""
        try:
            client = await self._get_list_client()
            collection_names = await client.list_collections()
            if name not in collection_names:
                msg = f"Collection {name!r} not found in ChromaDB"
                raise ValueError(msg)  # noqa: TRY301
            db = ChromaBackend(
                vector_store_id=name,
                persist_directory=self.persist_directory,
                **kwargs,
            )

            return cast(BaseVectorDB, db)

        except Exception as e:
            msg = f"Failed to connect to ChromaDB collection {name}: {e}"
            self.logger.exception(msg)
            raise ValueError(msg) from e

    async def delete_vector_store(self, name: str) -> bool:
        """Delete a vector store (collection)."""
        try:
            client = await self._get_list_client()
            await client.delete_collection(name=name)
            if self.persist_directory:
                collection_path = Path(self.persist_directory) / name
                if collection_path.exists():
                    shutil.rmtree(collection_path, ignore_errors=True)

        except Exception:
            self.logger.exception("Failed to delete collection %s", name)
            return False
        else:
            return True

    async def close(self):
        """Close all vector store connections."""
        if self._list_client:
            try:
                await self._list_client.reset()
            except Exception as e:  # noqa: BLE001
                self.logger.warning("Error closing list client: %s", e)

            self._list_client = None


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)

    async def main():
        async with ChromaVectorManager.run_server(
            "test",
            wait_output=[".*Uvicorn running.*"],
            wait_timeout=10,
        ):
            manager = ChromaVectorManager()
            await manager.create_vector_store("test")
            await manager.list_vector_stores()
            await manager.delete_vector_store("test")
            await manager.close()

    asyncio.run(main())
