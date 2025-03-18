"""Module providing LiteLLM-based embedding generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from docler.embeddings.base import EmbeddingProvider


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class LiteLLMEmbeddings(EmbeddingProvider):
    """Embeddings provider using LiteLLM, supporting various model providers."""

    NAME = "LiteLLM"
    REQUIRED_PACKAGES: ClassVar[list[str]] = ["litellm"]

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        dimensions: int | None = None,
        **litellm_kwargs: Any,
    ):
        """Initialize the LiteLLM embeddings provider.

        Args:
            model: The model identifier (e.g., "text-embedding-ada-002",
                  "mistral/mistral-embed", "gemini/text-embedding-004")
            api_key: Optional API key for the provider
            dimensions: Optional number of dimensions for the embeddings
            **litellm_kwargs: Additional arguments passed to litellm.embedding()
        """
        self.model = model
        self.api_key = api_key
        self.dimensions = dimensions
        self.litellm_kwargs = litellm_kwargs

        # Import here to allow the class to be imported even without litellm installed
        import litellm

        self._litellm = litellm

    async def embed_stream(
        self,
        texts: AsyncIterator[str],
        batch_size: int = 8,
    ) -> AsyncIterator[np.ndarray]:
        """Stream embeddings one at a time.

        Args:
            texts: Iterator of text strings to embed
            batch_size: Number of texts to process in each batch

        Yields:
            numpy.ndarray: Embedding vector for each input text
        """
        batch: list[str] = []

        async for text in texts:
            batch.append(text)
            if len(batch) >= batch_size:
                embeddings = await self._get_embeddings(batch)
                for embedding in embeddings:
                    yield embedding
                batch = []

        if batch:
            embeddings = await self._get_embeddings(batch)
            for embedding in embeddings:
                yield embedding

    async def _get_embeddings(
        self,
        texts: list[str],
    ) -> list[np.ndarray]:
        """Get embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of numpy arrays containing embedding vectors
        """
        kwargs = self.litellm_kwargs.copy()

        # Add API key if provided
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # Add dimensions if specified
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        # Call litellm's embedding function
        response = await self._litellm.aembedding(
            model=self.model,
            input=texts,
            **kwargs,
        )

        # Extract and convert embeddings to numpy arrays
        return [
            np.array(item["embedding"], dtype=np.float32) for item in response["data"]
        ]
