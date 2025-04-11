"""Module providing streaming embedding generation."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, ClassVar

from docler.configs.embedding_configs import OpenAIEmbeddingConfig
from docler.embeddings.base import EmbeddingProvider
from docler.utils import get_api_key


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import numpy as np


class OpenAIEmbeddings(EmbeddingProvider[OpenAIEmbeddingConfig]):
    """OpenAI embeddings with fallback to httpx if openai package not available."""

    Config = OpenAIEmbeddingConfig

    NAME = "OpenAI"
    REQUIRED_PACKAGES: ClassVar = {"openai", "httpx"}
    dimensions: ClassVar[int] = 1536

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or get_api_key("OPENAI_API_KEY")
        self.model = model
        self.use_openai = find_spec("openai")

    async def embed_stream(
        self,
        texts: AsyncIterator[str],
        batch_size: int = 8,
    ) -> AsyncIterator[np.ndarray]:
        """Embeddings iterator."""
        import numpy as np

        batch: list[str] = []

        async for text in texts:
            batch.append(text)
            if len(batch) >= batch_size:
                embeddings = (
                    await self._get_embeddings_official(batch)
                    if self.use_openai
                    else await self._get_embeddings_rest(batch)
                )
                for embedding in embeddings:
                    yield np.array(embedding, dtype=np.float32)
                batch = []

        if batch:
            embeddings = (
                await self._get_embeddings_official(batch)
                if self.use_openai
                else await self._get_embeddings_rest(batch)
            )
            for embedding in embeddings:
                yield np.array(embedding, dtype=np.float32)

    async def _get_embeddings_official(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Get embeddings using official OpenAI client."""
        import openai

        client = openai.AsyncClient(api_key=self.api_key)
        response = await client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    async def _get_embeddings_rest(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Get embeddings using httpx."""
        import anyenv

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"input": texts, "model": self.model}
        result = await anyenv.post_json(
            url,
            json_data=data,
            headers=headers,
            return_type=dict,
        )
        return [item["embedding"] for item in result["data"]]
