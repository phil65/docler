"""Module providing different embedding model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import numpy as np


class EmbeddingProvider(ABC):
    """Base class for streaming embedding providers."""

    @abstractmethod
    def embed_stream(
        self,
        texts: AsyncIterator[str],
        batch_size: int = 8,
    ) -> AsyncIterator[np.ndarray]:
        """Stream embeddings one at a time."""

    async def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Convert multiple texts to embeddings.

        Args:
            texts: List of texts to convert to embeddings

        Returns:
            List of embedding vectors
        """
        results: list[np.ndarray] = []

        # Create an async iterator from the list
        async def text_iterator():
            for text in texts:
                yield text

        # Use embed_stream internally
        async for embedding in self.embed_stream(text_iterator()):
            results.append(embedding)  # noqa: PERF401

        return results

    async def embed_query(self, query: str) -> np.ndarray:
        """Convert a single query to an embedding.

        Args:
            query: Query text to convert

        Returns:
            Embedding vector for the query
        """
        embeddings = await self.embed_texts([query])
        return embeddings[0]
