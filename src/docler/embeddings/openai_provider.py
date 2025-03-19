"""Module providing streaming embedding generation."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from docler.embeddings.base import EmbeddingProvider


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import openai


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings with fallback to httpx if openai package not available."""

    NAME = "OpenAI"
    REQUIRED_PACKAGES: ClassVar = {"openai", "httpx"}
    dimensions: ClassVar[int] = 1536

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
    ):
        self.api_key = api_key
        self.model = model
        self._client = self._init_client()

    def _init_client(self) -> openai.AsyncClient | None:
        """Try to initialize OpenAI client, return None if package not available."""
        if importlib.util.find_spec("openai"):
            import openai

            return openai.AsyncClient(api_key=self.api_key)
        return None

    async def embed_stream(
        self,
        texts: AsyncIterator[str],
        batch_size: int = 8,
    ) -> AsyncIterator[np.ndarray]:
        """Embeddings iterator."""
        batch: list[str] = []

        async for text in texts:
            batch.append(text)
            if len(batch) >= batch_size:
                embeddings = (
                    await self._get_embeddings_official(batch)
                    if self._client
                    else await self._get_embeddings_httpx(batch)
                )
                for embedding in embeddings:
                    yield np.array(embedding, dtype=np.float32)
                batch = []

        if batch:
            embeddings = (
                await self._get_embeddings_official(batch)
                if self._client
                else await self._get_embeddings_httpx(batch)
            )
            for embedding in embeddings:
                yield np.array(embedding, dtype=np.float32)

    async def _get_embeddings_official(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Get embeddings using official OpenAI client."""
        assert self._client
        response = await self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def _get_embeddings_httpx(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Get embeddings using httpx."""
        import httpx

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "input": texts,
            "model": self.model,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()

        return [item["embedding"] for item in result["data"]]
