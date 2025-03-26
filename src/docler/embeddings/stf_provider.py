"""Module providing StreamingTransformer embedding generation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from docler.configs.embedding_configs import SentenceTransformerEmbeddingConfig
from docler.embeddings.base import EmbeddingProvider


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import numpy as np


class SentenceTransformerEmbeddings(
    EmbeddingProvider[SentenceTransformerEmbeddingConfig]
):
    """Local embeddings using sentence-transformers."""

    Config = SentenceTransformerEmbeddingConfig

    NAME = "SentenceTransformers"
    REQUIRED_PACKAGES: ClassVar = {"sentence-transformers"}

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)  # type: ignore
        self.dimensions = self.model.get_sentence_embedding_dimension()  # pyright: ignore

    async def embed_stream(
        self,
        texts: AsyncIterator[str],
        batch_size: int = 8,
    ) -> AsyncIterator[np.ndarray]:
        batch: list[str] = []

        async for text in texts:
            batch.append(text)
            if len(batch) >= batch_size:
                # Run CPU-intensive encoding in thread pool
                embeddings = await asyncio.to_thread(self.model.encode, batch)
                for embedding in embeddings:  # Convert from numpy
                    yield embedding
                batch = []

        if batch:
            embeddings = await asyncio.to_thread(self.model.encode, batch)
            for embedding in embeddings:
                yield embedding


if __name__ == "__main__":
    import asyncio

    async def main():
        provider = SentenceTransformerEmbeddings()
        for embedding in await provider.embed_texts(["Hello world"]):
            print(embedding)

    asyncio.run(main())
