"""ChromaDB vector store backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
import uuid

from docler.log import get_logger
from docler.models import SearchResult, Vector
from docler.vector_db.base_backend import VectorStoreBackend
from docler.vector_db.dbs.chroma_db.utils import to_search_results


if TYPE_CHECKING:
    import numpy as np


logger = get_logger(__name__)
Metric = Literal["cosine", "l2", "ip"]


class ChromaBackend(VectorStoreBackend):
    """ChromaDB implementation of vector store backend."""

    NAME = "ChromaDB"
    REQUIRED_PACKAGES: ClassVar = {"chromadb"}

    def __init__(
        self,
        vector_store_id: str = "default",
        persist_directory: str | None = None,
        distance_metric: Metric = "cosine",
    ):
        """Initialize ChromaDB backend.

        Args:
            vector_store_id: Name of collection to use
            persist_directory: Directory for persistent storage
                               (connects to ChromaDB server if None)
            distance_metric: Distance metric to use for similarity search

        Raises:
            ImportError: If chromadb is not installed
        """
        import chromadb

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)  # type: ignore
        else:
            self._client = chromadb.Client()  # type: ignore
        self._collection = self._client.get_or_create_collection(
            name=vector_store_id,
            metadata={"hnsw:space": distance_metric},
            embedding_function=None,
        )
        self.vector_store_id = vector_store_id
        msg = "ChromaDB initialized - collection: %s, persistent: %s"
        logger.info(msg, vector_store_id, bool(persist_directory))

    async def add_vectors(
        self,
        vectors: list[np.ndarray],
        metadata: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add multiple vectors to ChromaDB."""
        import anyenv

        if len(vectors) != len(metadata):
            msg = "Number of vectors and metadata entries must match"
            raise ValueError(msg)

        ids_ = [str(uuid.uuid4()) for _ in vectors] if ids is None else ids
        vector_lists: list[float] = [v.tolist() for v in vectors]  # type: ignore
        await anyenv.run_in_thread(
            self._collection.add,  # type: ignore
            ids_,
            vector_lists,
            metadata,
        )
        return ids_

    async def get_vector(self, chunk_id: str) -> Vector | None:
        """Get vector and metadata from ChromaDB."""
        import anyenv
        import numpy as np

        result = await anyenv.run_in_thread(
            self._collection.get,
            ids=[chunk_id],
            include=["embeddings", "metadatas"],
        )

        if not result["ids"] or not result["embeddings"] or not result["metadatas"]:
            return None

        vector = np.array(result["embeddings"][0])
        metadata = cast(dict[str, Any], result["metadatas"][0])
        return Vector(data=vector, metadata=metadata, id=chunk_id)

    async def list_vector_ids(
        self,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        data = self._collection.get(include=[], limit=limit)
        return data["ids"]

    async def delete(self, chunk_id: str) -> bool:
        """Delete vector from ChromaDB."""
        import anyenv

        try:
            await anyenv.run_in_thread(self._collection.delete, ids=[chunk_id])
        except Exception:  # noqa: BLE001
            return False
        else:
            return True

    async def search_vectors(
        self,
        query_vector: np.ndarray,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search ChromaDB for similar vectors."""
        import anyenv

        query_list = [query_vector.tolist()]
        results = await anyenv.run_in_thread(
            self._collection.query,
            query_embeddings=query_list,
            n_results=k,
            where=filters,
            include=["metadatas", "distances"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        return to_search_results(results)


if __name__ == "__main__":
    import asyncio

    import numpy as np

    from docler.vector_db.dbs.chroma_db import ChromaVectorManager

    async def main():
        async with ChromaVectorManager.run_server("chroma"):
            db = ChromaBackend(persist_directory="./chroma")
            query_vector = np.array([0.1, 0.2, 0.3])
            await db.add_vector(query_vector, metadata={"source": "user"})
            search_results = await db.search_vectors(query_vector)
            print(search_results)

    asyncio.run(main())
