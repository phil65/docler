"""Pinecone vector store backend implementation."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, ClassVar
import uuid

from docler.vector_db.base import SearchResult, VectorStoreBackend


if TYPE_CHECKING:
    import numpy as np
    from pinecone import Pinecone


logger = logging.getLogger(__name__)


class PineconeBackend(VectorStoreBackend):
    """Pinecone implementation of vector store backend."""

    NAME: ClassVar[str] = "pinecone"
    REQUIRED_PACKAGES: ClassVar = {"pinecone-client"}

    def __init__(
        self,
        pinecone_client: Pinecone,
        host: str,
        dimension: int = 1536,
        namespace: str = "default",
    ):
        """Initialize Pinecone backend.

        Args:
            pinecone_client: Pinecone client
            host: Host URL for the index
            dimension: Dimension of vectors to store
            namespace: Namespace to use for vectors
        """
        self._pinecone = pinecone_client
        self._host = host
        self._index = None
        self.dimension = dimension
        self.namespace = namespace
        self._initialized = False
        self.batch_size = 100

    async def _ensure_initialized(self):
        """Ensure the index client is initialized."""
        if not self._initialized:
            self._index = self._pinecone.IndexAsyncio(host=self._host)
            self._initialized = True
        return self._index

    async def add_vector(
        self,
        vector: np.ndarray,
        metadata: dict[str, Any],
        id_: str | None = None,
    ) -> str:
        """Add single vector to Pinecone.

        Args:
            vector: Vector embedding to store
            metadata: Metadata dictionary for the vector
            id_: Optional ID (generated if not provided)

        Returns:
            ID of the stored vector
        """
        if id_ is None:
            id_ = str(uuid.uuid4())
        vector_list: list[float] = vector.tolist()  # pyright: ignore
        metadata_copy = self._prepare_metadata(metadata)
        index = await self._ensure_initialized()
        assert index
        async with index:
            vector_tuple = (id_, vector_list, metadata_copy)
            await index.upsert(vectors=[vector_tuple], namespace=self.namespace)

        return id_

    async def add_vectors(
        self,
        vectors: list[np.ndarray],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add vectors to Pinecone.

        Args:
            vectors: List of vector embeddings
            metadata: List of metadata dictionaries
            ids: Optional list of IDs

        Returns:
            List of IDs for stored vectors
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        vectors_data = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            vector_list: list[float] = vector.tolist()  # pyright: ignore
            meta_copy = self._prepare_metadata(meta)
            vectors_data.append((ids[i], vector_list, meta_copy))
        index = await self._ensure_initialized()
        assert index
        async with index:
            for i in range(0, len(vectors_data), self.batch_size):
                batch = vectors_data[i : i + self.batch_size]
                await index.upsert(vectors=batch, namespace=self.namespace)

        return ids

    async def get_vector(
        self,
        chunk_id: str,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Get vector and metadata from Pinecone.

        Args:
            chunk_id: ID of vector to retrieve

        Returns:
            Tuple of (vector, metadata) if found, None if not
        """
        import numpy as np

        index = await self._ensure_initialized()
        assert index
        async with index:
            result = await index.fetch(ids=[chunk_id], namespace=self.namespace)
        vectors = result.vectors
        if chunk_id not in vectors:
            return None
        vector_data = vectors[chunk_id]
        vector = np.array(vector_data["values"])
        metadata = self._restore_metadata(vector_data.metadata or {})

        return vector, metadata

    async def update_vector(
        self,
        chunk_id: str,
        vector: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update vector in Pinecone.

        Args:
            chunk_id: ID of vector to update
            vector: New vector embedding (unchanged if None)
            metadata: New metadata (unchanged if None)

        Returns:
            True if vector was updated, False if not found
        """
        if vector is None or metadata is None:
            current = await self.get_vector(chunk_id)
            if current is None:
                return False
            current_vector, current_metadata = current
            vector = vector if vector is not None else current_vector
            metadata = metadata if metadata is not None else current_metadata

        vector_list: list[float] = vector.tolist()  # pyright: ignore
        prepared_metadata = self._prepare_metadata(metadata)
        index = await self._ensure_initialized()
        assert index
        try:
            async with index:
                vectors = [(chunk_id, vector_list, prepared_metadata)]
                await index.upsert(vectors=vectors, namespace=self.namespace)
        except Exception:
            logger.exception("Failed to update vector %s", chunk_id)
            return False
        else:
            return True

    async def delete(self, chunk_id: str) -> bool:
        """Delete vector by ID.

        Args:
            chunk_id: ID of vector to delete

        Returns:
            True if vector was deleted, False if not found
        """
        index = await self._ensure_initialized()
        assert index
        try:
            async with index:
                await index.delete(ids=[chunk_id], namespace=self.namespace)
        except Exception:
            logger.exception("Failed to delete vector %s", chunk_id)
            return False
        else:
            return True

    async def search_vectors(
        self,
        query_vector: np.ndarray,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search Pinecone for similar vectors.

        Args:
            query_vector: Vector to search for
            k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of search results
        """
        vector_list: list[float] = query_vector.tolist()  # pyright: ignore
        filter_obj = self._convert_filters(filters) if filters else None
        query_params = {
            "vector": vector_list,
            "top_k": k,
            "namespace": self.namespace,
            "include_metadata": True,
        }

        if filter_obj:
            query_params["filter"] = filter_obj
        index = await self._ensure_initialized()
        assert index
        try:
            async with index:
                results = await index.query(**query_params)
        except Exception:
            logger.exception("Error searching Pinecone")
            return []

        search_results = []
        for match in results.get("matches", []):
            raw_metadata = match.get("metadata", {})
            metadata = self._restore_metadata(raw_metadata)
            score = match.get("score", 0.0)
            text = metadata.pop("text", None) if isinstance(metadata, dict) else None
            result = SearchResult(
                chunk_id=match["id"],
                score=score,
                metadata=metadata,
                text=text,
            )
            search_results.append(result)

        return search_results

    def _convert_filters(self, filters: dict[str, Any]) -> dict:
        """Convert standard filters to Pinecone filter format.

        Args:
            filters: Dictionary of filters

        Returns:
            Pinecone-compatible filter object
        """
        if not filters:
            return {}

        pinecone_filter = {}
        for key, value in filters.items():
            pinecone_filter[key] = {"$in": value} if isinstance(value, list) else value
        return pinecone_filter

    def _prepare_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Prepare metadata for Pinecone storage.

        Pinecone has metadata limitations, so we encode complex objects.

        Args:
            metadata: Original metadata dictionary

        Returns:
            Pinecone-compatible metadata
        """
        import anyenv

        prepared = {}
        for key, value in metadata.items():
            if isinstance(value, str | int | float | bool):
                prepared[key] = value
            elif isinstance(value, list | dict):
                # Convert complex types to JSON strings
                try:
                    # First try to store it directly if simple enough
                    if isinstance(value, list) and all(
                        isinstance(x, str | int | float | bool) for x in value
                    ):
                        prepared[key] = value  # type: ignore
                    else:
                        prepared[f"{key}_json"] = anyenv.dump_json(value)
                except (TypeError, ValueError):
                    dumped = anyenv.dump_json(str(value)).encode()
                    prepared[f"{key}_b64"] = base64.b64encode(dumped).decode()
            elif value is not None:
                prepared[key] = str(value)

        # Ensure text field is preserved
        if "text" in metadata:
            prepared["text"] = str(metadata["text"])

        return prepared

    def _restore_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Restore encoded metadata fields.

        Args:
            metadata: Metadata from Pinecone

        Returns:
            Restored metadata with decoded fields
        """
        import json

        restored = metadata.copy()

        # Process encoded fields
        for key in list(restored.keys()):
            # Restore JSON encoded fields
            if key.endswith("_json") and key[:-5] not in restored:
                try:
                    base_key = key[:-5]
                    restored[base_key] = json.loads(restored[key])
                    del restored[key]
                except Exception:  # noqa: BLE001
                    pass

            # Restore base64 encoded fields
            elif key.endswith("_b64") and key[:-4] not in restored:
                try:
                    base_key = key[:-4]
                    json_str = base64.b64decode(restored[key]).decode()
                    restored[base_key] = json.loads(json_str)
                    del restored[key]
                except Exception:  # noqa: BLE001
                    pass

        return restored

    async def close(self):
        """Close the Pinecone connection."""
        # Close the index client
        if self._initialized and self._index:
            try:
                # With the updated context manager approach,
                # the client handles closing automatically
                self._initialized = False
                self._index = None
            except Exception:
                logger.exception("Error closing Pinecone index connection")
