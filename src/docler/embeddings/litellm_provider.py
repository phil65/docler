"""Module providing LiteLLM-based embedding generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from docler.configs.embedding_configs import LiteLLMEmbeddingConfig
from docler.embeddings.base import EmbeddingProvider


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class LiteLLMEmbeddings(EmbeddingProvider[LiteLLMEmbeddingConfig]):
    """Embeddings provider using LiteLLM, supporting various model providers."""

    Config = LiteLLMEmbeddingConfig

    NAME = "LiteLLM"
    REQUIRED_PACKAGES: ClassVar = {"litellm"}

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        dimensions: int | None = None,
        input_type: str | None = None,
        base_url: str | None = None,
        **litellm_kwargs: str | float | bool,
    ):
        """Initialize the LiteLLM embeddings provider.

        Args:
            model: The model identifier (e.g., "openai/text-embedding-3-small",
                  "mistral/mistral-embed", "gemini/text-embedding-004")
            api_key: Optional API key for the provider
            dimensions: Optional number of dimensions for the embeddings
            input_type: Optional input type for the embeddings
            base_url: Optional base URL for the provider
            **litellm_kwargs: Additional arguments passed to litellm.embedding()
        """
        import litellm

        self.model = model
        self.api_key = api_key
        self.dimensions = dimensions
        self.input_type = input_type
        self.base_url = base_url
        self.litellm_kwargs = litellm_kwargs
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

    async def _get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Get embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of numpy arrays containing embedding vectors
        """
        kwargs = self.litellm_kwargs.copy()
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.input_type:
            kwargs["input_type"] = self.input_type

        response = await self._litellm.aembedding(model=self.model, input=texts, **kwargs)
        return [np.array(i["embedding"], dtype=np.float32) for i in response["data"]]
