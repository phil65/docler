"""Base class for vector database managers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from docler.provider import BaseProvider


if TYPE_CHECKING:
    from docler.models import VectorStoreInfo
    from docler.vector_db.base import VectorDB


class VectorManagerBase[TConfig](BaseProvider[TConfig], ABC):
    """Abstract base class for vector database managers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this vector database provider."""

    @abstractmethod
    async def list_vector_stores(self) -> list[VectorStoreInfo]:
        """List all available vector stores for this provider."""

    @abstractmethod
    async def create_vector_store(self, name: str, **kwargs) -> VectorDB:
        """Create a new vector store."""

    @abstractmethod
    async def get_vector_store(self, name: str, **kwargs) -> VectorDB:
        """Get a connection to an existing vector store."""

    @abstractmethod
    async def delete_vector_store(self, name: str) -> bool:
        """Delete a vector store."""

    @abstractmethod
    async def close(self) -> None:
        """Close all vector store connections."""
