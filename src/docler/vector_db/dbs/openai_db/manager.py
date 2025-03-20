"""OpenAI Vector Store manager."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal

from docler.configs.vector_db_configs import OpenAIVectorConfig
from docler.vector_db.base_manager import VectorManagerBase
from docler.vector_db.dbs.openai_db.db import ChunkingStrategy, OpenAIVectorDB


if TYPE_CHECKING:
    from docler.vector_db.base import VectorDB


logger = logging.getLogger(__name__)


class OpenAIVectorManager(VectorManagerBase[OpenAIVectorConfig]):
    """Manager for OpenAI Vector Stores API."""

    NAME = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
    ):
        """Initialize the OpenAI Vector Store manager.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI organization ID (defaults to OPENAI_ORG_ID env var)
        """
        from openai import AsyncOpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            msg = "OpenAI API key must be provided via parameter or OPENAI_API_KEY envvar"
            raise ValueError(msg)

        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self._client = AsyncOpenAI(api_key=self.api_key, organization=self.organization)
        self._vector_stores: dict[str, OpenAIVectorDB] = {}

    @property
    def name(self) -> str:
        return self.NAME

    async def list_vector_stores(self) -> list[dict[str, Any]]:
        """List all vector stores available in the OpenAI account.

        Returns:
            List of vector store metadata objects
        """
        try:
            response = await self._client.vector_stores.list()
            return [
                {"id": vs.id, "name": vs.name, "created_at": vs.created_at}
                for vs in response.data
            ]
        except Exception:
            logger.exception("Error listing vector stores")
            return []

    async def create_vector_store(
        self,
        name: str,
        *,
        chunking_strategy: ChunkingStrategy = "auto",
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **_kwargs: Any,
    ) -> VectorDB:
        """Create a new vector store.

        Args:
            name: Name for the new vector store
            chunking_strategy: Strategy for chunking text
            max_chunk_size: Maximum chunk size in tokens (for static strategy)
            chunk_overlap: Overlap between chunks in tokens (for static strategy)

        Returns:
            Configured vector database instance

        Raises:
            ValueError: If creation fails
        """
        try:
            # Create the vector store
            response = await self._client.vector_stores.create(name=name)
            vector_store_id = response.id

            # Create and return a configured database instance
            db = OpenAIVectorDB(
                vector_store_id=vector_store_id,
                client=self._client,
                chunking_strategy=chunking_strategy,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Store for tracking
            self._vector_stores[name] = db
        except Exception as e:
            msg = f"Failed to create vector store: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e
        else:
            return db

    async def get_vector_store(
        self,
        name: str,
        *,
        chunking_strategy: ChunkingStrategy = "auto",
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **_kwargs: Any,
    ) -> VectorDB:
        """Get a connection to an existing vector store.

        Args:
            name: ID of the existing vector store
            chunking_strategy: Strategy for chunking text
            max_chunk_size: Maximum chunk size in tokens (for static strategy)
            chunk_overlap: Overlap between chunks in tokens (for static strategy)

        Returns:
            Configured vector database instance

        Raises:
            ValueError: If store doesn't exist or connection fails
        """
        if name in self._vector_stores:
            return self._vector_stores[name]
        try:
            await self._client.vector_stores.retrieve(vector_store_id=name)
            db = OpenAIVectorDB(
                vector_store_id=name,
                client=self._client,
                chunking_strategy=chunking_strategy,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self._vector_stores[name] = db
        except Exception as e:
            msg = f"Failed to connect to vector store {name}: {e}"
            logger.exception(msg)
            raise ValueError(msg) from e
        else:
            return db

    async def delete_vector_store(self, vector_store_id: str) -> bool:
        """Delete a vector store.

        Args:
            vector_store_id: ID of the vector store to delete

        Returns:
            True if successful, False if failed
        """
        try:
            await self._client.vector_stores.delete(vector_store_id=vector_store_id)
            if vector_store_id in self._vector_stores:
                del self._vector_stores[vector_store_id]
        except Exception:
            logger.exception("Error deleting vector store %s", vector_store_id)
            return False
        else:
            return True

    async def add_file_to_store(
        self,
        vector_store_id: str,
        file_path: str,
        attributes: dict[str, Any] | None = None,
        chunking_strategy: Literal["auto", "static"] = "auto",
    ) -> str | None:
        """Add a file to a vector store.

        Args:
            vector_store_id: ID of the vector store
            file_path: Path to the file to add
            attributes: Optional metadata to attach to the file
            chunking_strategy: Strategy for chunking the file

        Returns:
            File ID if successful, None if failed
        """
        import upathtools

        try:
            data = await upathtools.read_path(file_path, mode="rb")
            upload_response = await self._client.files.create(
                file=(file_path, data),
                purpose="user_data",
            )
            file_id = upload_response.id
            if chunking_strategy == "auto":
                chunking_config: dict[str, Any] = {"type": "auto"}
            else:
                chunking_config = {
                    "type": "static",
                    "static": {
                        "max_chunk_size_tokens": 1000,
                        "chunk_overlap_tokens": 200,
                    },
                }

            await self._client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_id,
                attributes=attributes or {},
                chunking_strategy=chunking_config,  # type: ignore
            )
        except Exception:
            logger.exception("Error adding file to vector store")
            return None
        else:
            return file_id

    async def search(
        self,
        vector_store_id: str,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search a vector store directly.

        Args:
            vector_store_id: ID of the vector store to search
            query: Query text
            k: Maximum number of results
            filters: Optional filters to apply

        Returns:
            List of search results
        """
        try:
            vector_db = await self.get_vector_store(vector_store_id)
            results = await vector_db.similar_texts(query, k, filters)
            formatted_results = []
            for text, score, metadata in results:
                result_obj = {"text": text, "score": score, **metadata}
                formatted_results.append(result_obj)
        except Exception:
            logger.exception("Error searching vector store")
            return []
        else:
            return formatted_results

    def _convert_filters(self, filters: dict[str, Any]) -> Any:  # type: ignore
        """Convert standard filters to OpenAI's filter format.

        Args:
            filters: Dictionary of filters

        Returns:
            OpenAI-compatible filter object
        """
        filter_conditions: list[dict[str, Any]] = []

        for key, value in filters.items():
            if isinstance(value, list):
                or_conditions = [
                    {"key": key, "type": "eq", "value": item}
                    for item in value
                    if isinstance(item, str | int | float | bool)
                ]
                if or_conditions:
                    filter_conditions.append({"type": "or", "filters": or_conditions})
            elif isinstance(value, str | int | float | bool):
                # Handle single value (equality)
                filter_ = {"key": key, "type": "eq", "value": value}
                filter_conditions.append(filter_)

        # Combine with AND if multiple conditions
        if len(filter_conditions) > 1:
            return {"type": "and", "filters": filter_conditions}
        if filter_conditions:
            # Just return the single filter
            return filter_conditions[0]
        # No filters
        return {}

    async def close(self) -> None:
        """Close all vector store connections."""
        # Close all tracked vector stores
        for db in self._vector_stores.values():
            await db.close()

        self._vector_stores.clear()


if __name__ == "__main__":
    import asyncio

    async def main():
        manager = OpenAIVectorManager()
        dbs = await manager.list_vector_stores()
        print(dbs)

    asyncio.run(main())
