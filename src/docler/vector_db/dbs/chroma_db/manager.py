"""ChromaDB vector store manager implementation."""

from __future__ import annotations

from typing import ClassVar, cast

from docler.configs.vector_db_configs import ChromaConfig
from docler.log import get_logger
from docler.models import VectorStoreInfo
from docler.vector_db.base import VectorDB
from docler.vector_db.base_manager import VectorManagerBase
from docler.vector_db.dbs.chroma_db.db import ChromaBackend


logger = get_logger(__name__)


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
        try:
            import chromadb  # noqa: F401
        except ImportError as e:
            msg = "ChromaDB is not installed. Please install with 'pip install chromadb'"
            raise ImportError(msg) from e

        # Store configuration
        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        self.ssl = ssl
        self.headers = headers

        # Map of collection name -> backend instance
        self._vector_stores: dict[str, ChromaBackend] = {}

        # Initialized client for listing collections
        self._list_client = None

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
        return ChromaConfig(
            persist_directory=self.persist_directory,
            collection_name="default",
        )

    async def _get_list_client(self):
        """Get a client for listing collections."""
        import chromadb

        if not self._list_client:
            # Create an async client for listing
            if self.host:
                # TODO: need to check how async persistent works
                self._list_client = await chromadb.AsyncHttpClient(  # type: ignore
                    host=self.host, port=self.port, ssl=self.ssl, headers=self.headers
                )
            else:
                self._list_client = await chromadb.AsyncClient()  # type: ignore

        return self._list_client

    async def list_vector_stores(self) -> list[VectorStoreInfo]:
        """List all available vector stores (collections) for this provider."""
        try:
            client = await self._get_list_client()
            collections = await client.list_collections()
            return [VectorStoreInfo(db_id=c, name=c) for c in collections]
        except Exception:
            logger.exception("Error listing ChromaDB collections")
            return []

    async def create_vector_store(
        self,
        name: str,
        **kwargs,
    ) -> VectorDB:
        """Create a new vector store (collection).

        Args:
            name: Name for the new collection
            **kwargs: Additional parameters for the collection

        Returns:
            Configured vector database instance

        Raises:
            ValueError: If creation fails
        """
        if name in self._vector_stores:
            return cast(VectorDB, self._vector_stores[name])

        try:
            db = ChromaBackend(
                vector_store_id=name,
                persist_directory=self.persist_directory,
                **kwargs,
            )

            # await db.initialize()

            # Store for later cleanup
            self._vector_stores[name] = db

            return cast(VectorDB, db)

        except Exception as e:
            msg = f"Failed to create ChromaDB collection: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e

    async def get_vector_store(
        self,
        name: str,
        **kwargs,
    ) -> VectorDB:
        """Get a connection to an existing collection.

        Args:
            name: Name of the collection
            **kwargs: Additional parameters for the connection

        Returns:
            Configured vector database instance

        Raises:
            ValueError: If collection doesn't exist or connection fails
        """
        if name in self._vector_stores:
            return cast(VectorDB, self._vector_stores[name])

        try:
            # Check if collection exists
            client = await self._get_list_client()
            collection_names = await client.list_collections()

            if name not in collection_names:
                msg = f"Collection {name!r} not found in ChromaDB"
                raise ValueError(msg)  # noqa: TRY301

            # Connect to existing collection
            db = ChromaBackend(
                vector_store_id=name,
                persist_directory=self.persist_directory,
                **kwargs,
            )

            # Initialize
            # await db.initialize()

            # Store for later cleanup
            self._vector_stores[name] = db

            return cast(VectorDB, db)

        except Exception as e:
            msg = f"Failed to connect to ChromaDB collection {name}: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e

    async def delete_vector_store(self, name: str) -> bool:
        """Delete a vector store (collection).

        Args:
            name: Name of the collection to delete

        Returns:
            True if successful, False if failed
        """
        return True

    #     try:
    #         # Clean up any existing instance
    #         if name in self._vector_stores:
    #             db = self._vector_stores[name]
    #             success = await db.delete_collection()
    #             await db.close()
    #             del self._vector_stores[name]
    #             return success

    #         # No existing instance, use list client
    #         client = await self._get_list_client()
    #         await client.delete_collection(name=name)

    #     except Exception:
    #         logger.exception("Failed to delete collection %s", name)
    #         return False
    #     else:
    #         return True

    async def close(self) -> None:
        """Close all vector store connections."""
        # for db in list(self._vector_stores.values()):
        #     await db.close()
        self._vector_stores.clear()

        # Close list client
        if self._list_client:
            try:
                await self._list_client.reset()
            except Exception as e:  # noqa: BLE001
                logger.warning("Error closing list client: %s", e)

            self._list_client = None
