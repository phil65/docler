"""Upstash vector store backend implementation."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from upstash_vector.core.index_operations import MetadataUpdateMode

from docler.vector_db.base import SearchResult, VectorStoreBackend


if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)


class UpstashBackend(VectorStoreBackend):
    """Upstash Vector implementation of vector store backend."""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        collection_name: str = "default",
    ) -> None:
        """Initialize Upstash Vector backend.

        Args:
            url: Upstash Vector REST API URL. Falls back to UPSTASH_ENDPOINT env var.
            token: Upstash Vector API token. Falls back to UPSTASH_API_KEY env var.
            collection_name: Name to use as namespace for vectors.

        Raises:
            ImportError: If upstash_vector is not installed
            ValueError: If URL or token is not provided
        """
        try:
            from upstash_vector import Index
        except ImportError as e:
            msg = "Upstash Vector is not installed."
            raise ImportError(msg) from e

        # Get configuration from params or env
        self.url = url or os.getenv("UPSTASH_ENDPOINT")
        self.token = token or os.getenv("UPSTASH_API_KEY")

        if not self.url:
            msg = "Upstash Vector URL must be provided via 'url' parameter or UPSTASH_ENDPOINT env var"  # noqa: E501
            raise ValueError(msg)

        if not self.token:
            msg = "Upstash Vector token must be provided via 'token' parameter or UPSTASH_API_KEY env var"  # noqa: E501
            raise ValueError(msg)

        self.namespace = collection_name
        self._client = Index(url=self.url, token=self.token)

        logger.info("Upstash Vector initialized with namespace: %s", self.namespace)

    async def add_vector(
        self,
        vector: np.ndarray,
        metadata: dict[str, Any],
        id_: str | None = None,
    ) -> str:
        """Add single vector to Upstash.

        Args:
            vector: Vector embedding to store
            metadata: Metadata dictionary for the vector
            id_: Optional ID (generated if not provided)

        Returns:
            ID of the stored vector
        """
        import uuid

        import anyenv
        from upstash_vector import Vector

        # Generate ID if not provided
        if id_ is None:
            id_ = str(uuid.uuid4())

        # Extract text from metadata if present
        text = metadata.pop("text", None) if metadata else None

        # Convert numpy array to list for JSON serialization
        vector_list = vector.tolist()

        # Create Upstash Vector object
        upstash_vector = Vector(
            id=id_,
            vector=vector_list,  # Upstash expects list format
            metadata=metadata,
            data=text,  # Store text in data field
        )

        # Upload vector
        await anyenv.run_in_thread(
            self._client.upsert,
            vectors=[upstash_vector],
            namespace=self.namespace,
        )

        return id_

    async def add_vectors(
        self,
        vectors: list[np.ndarray],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add multiple vectors to Upstash.

        Args:
            vectors: List of vector embeddings to store
            metadata: List of metadata dictionaries (one per vector)
            ids: Optional list of IDs (generated if not provided)

        Returns:
            List of IDs for the stored vectors
        """
        import uuid

        import anyenv
        from upstash_vector import Vector

        if len(vectors) != len(metadata):
            msg = "Number of vectors and metadata entries must match"
            raise ValueError(msg)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        # Create list of Upstash Vector objects
        upstash_vectors = []
        for id_, vector, meta in zip(ids, vectors, metadata):
            # Extract text from metadata if present
            text = meta.pop("text", None) if meta else None

            # Convert numpy array to list
            vector_list = vector.tolist()

            # Create Upstash Vector object
            upstash_vector = Vector(
                id=id_,
                vector=vector_list,
                metadata=meta,
                data=text,
            )
            upstash_vectors.append(upstash_vector)

        # Upload vectors (in batches if needed)
        batch_size = 100  # Adjust based on Upstash's limits
        for i in range(0, len(upstash_vectors), batch_size):
            batch = upstash_vectors[i : i + batch_size]
            await anyenv.run_in_thread(
                self._client.upsert,
                vectors=batch,
                namespace=self.namespace,
            )

        return ids

    async def get_vector(
        self,
        chunk_id: str,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Get vector and metadata from Upstash.

        Args:
            chunk_id: ID of vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None if not
        """
        import anyenv
        import numpy as np

        # Fetch vector from Upstash
        results = await anyenv.run_in_thread(
            self._client.fetch,
            ids=[chunk_id],
            include_vectors=True,
            include_metadata=True,
            include_data=True,
            namespace=self.namespace,
        )

        # Check if vector exists
        if not results or results[0] is None:
            return None

        result = results[0]

        # Convert vector to numpy array
        vector = np.array(result.vector)

        # Prepare metadata, including text if available
        metadata = result.metadata or {}
        if result.data:
            metadata["text"] = result.data

        return vector, metadata

    async def update_vector(
        self,
        chunk_id: str,
        vector: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update vector in Upstash.

        Args:
            chunk_id: ID of vector to update
            vector: New vector embedding (unchanged if None)
            metadata: New metadata (unchanged if None)

        Returns:
            True if vector was updated, False if not found
        """
        import anyenv

        # Get current vector if we need partial update
        if vector is None or metadata is None:
            current = await self.get_vector(chunk_id)
            if current is None:
                return False

            current_vector, current_metadata = current

            if vector is None:
                vector = current_vector
            if metadata is None:
                metadata = current_metadata

        # Extract text from metadata if present
        text = metadata.pop("text", None) if metadata else None

        # Convert numpy array to list
        vector_list = vector.tolist()

        try:
            # Update vector in Upstash
            result = await anyenv.run_in_thread(
                self._client.update,
                id=chunk_id,
                vector=vector_list,
                metadata=metadata,
                data=text,
                namespace=self.namespace,
                metadata_update_mode=MetadataUpdateMode.OVERWRITE,
            )
        except Exception:
            logger.exception("Failed to update vector %s", chunk_id)
            return False
        else:
            return result

    async def delete(self, chunk_id: str) -> bool:
        """Delete vector from Upstash.

        Args:
            chunk_id: ID of vector to delete

        Returns:
            True if vector was deleted, False if not found
        """
        import anyenv

        try:
            result = await anyenv.run_in_thread(
                self._client.delete,
                ids=[chunk_id],
                namespace=self.namespace,
            )
            # Check if any vectors were actually deleted
        except Exception:
            logger.exception("Failed to delete vector %s", chunk_id)
            return False
        else:
            return result.deleted > 0

    async def search_vectors(
        self,
        query_vector: np.ndarray,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search Upstash for similar vectors.

        Args:
            query_vector: Vector to search for
            k: Number of results to return
            filters: Optional filters to apply to results

        Returns:
            List of search results
        """
        import anyenv

        # Convert numpy array to list
        vector_list = query_vector.tolist()

        # Prepare filter string if filters are provided
        filter_str = ""
        if filters:
            # Convert filters dict to Upstash filter expression
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    # Handle list values
                    values_str = ", ".join([f'"{v}"' for v in value])
                    conditions.append(f"{key} IN [{values_str}]")
                else:
                    # Handle single value
                    conditions.append(f'{key} == "{value}"')

            filter_str = " AND ".join(conditions)

        # Execute search
        results = await anyenv.run_in_thread(
            self._client.query,
            vector=vector_list,
            top_k=k,
            include_vectors=False,
            include_metadata=True,
            include_data=True,
            filter=filter_str,
            namespace=self.namespace,
        )

        # Format results
        search_results = []
        for result in results:
            # Prepare metadata
            metadata = result.metadata or {}

            # Add text data if present
            text = result.data
            res = SearchResult(
                chunk_id=result.id,
                score=result.score,
                metadata=metadata,
                text=text,
            )
            search_results.append(res)

        return search_results
