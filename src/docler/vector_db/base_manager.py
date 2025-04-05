"""Base class for vector database managers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from docler.provider import BaseProvider


if TYPE_CHECKING:
    from docler.configs.vector_db_configs import BaseVectorStoreConfig
    from docler.models import VectorStoreInfo
    from docler.vector_db.base import BaseVectorDB


class VectorManagerBase[TConfig](BaseProvider[TConfig], ABC):
    """Abstract base class for vector database managers."""

    Config: ClassVar[type[BaseVectorStoreConfig]]

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this vector database provider."""

    @abstractmethod
    async def list_vector_stores(self) -> list[VectorStoreInfo]:
        """List all available vector stores for this provider."""

    @abstractmethod
    async def create_vector_store(self, name: str, **kwargs) -> BaseVectorDB:
        """Create a new vector store."""

    @abstractmethod
    async def get_vector_store(self, name: str, **kwargs) -> BaseVectorDB:
        """Get a connection to an existing vector store."""

    @abstractmethod
    async def delete_vector_store(self, name: str) -> bool:
        """Delete a vector store."""

    @abstractmethod
    async def close(self):
        """Close all vector store connections."""

    async def delete_all_vector_stores(self) -> bool:
        """Delete all vector stores."""
        stores = await self.list_vector_stores()
        for store in stores:
            await self.delete_vector_store(store.db_id)
        return True

    async def has_vector_store(self, name: str) -> bool:
        """Check if a vector store exists."""
        indexes = await self.list_vector_stores()
        index_names = [idx.name for idx in indexes]
        return name in index_names
