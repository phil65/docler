"""Vector store implementation for document and text chunk storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import numpy as np


class EmbeddingModel(ABC):
    """Abstract interface for embedding models."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Convert texts to embeddings.

        Args:
            texts: List of texts to convert to embeddings

        Returns:
            List of embedding vectors
        """

    @abstractmethod
    async def embed_query(self, query: str) -> np.ndarray:
        """Convert query to embedding.

        Args:
            query: Query text to convert

        Returns:
            Embedding vector for the query
        """
