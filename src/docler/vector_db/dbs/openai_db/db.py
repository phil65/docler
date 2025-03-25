"""OpenAI Vector Store implementation."""

from __future__ import annotations

import io
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from docler.models import TextChunk
from docler.utils import get_api_key
from docler.vector_db.base import VectorDB


if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types import FileChunkingStrategyParam


logger = logging.getLogger(__name__)

BATCH_SIZE = 10
ChunkingStrategy = Literal["auto", "static"]


def to_chunking_config(
    strategy: str,
    max_chunk_size: int,
    chunk_overlap: int,
) -> FileChunkingStrategyParam:
    """Create chunking configuration for OpenAI."""
    if strategy == "auto":
        return {"type": "auto"}
    if strategy == "fixed":
        return {
            "type": "static",
            "static": {
                "max_chunk_size_tokens": max_chunk_size,
                "chunk_overlap_tokens": chunk_overlap,
            },
        }
    # if strategy == "semantic":
    #     return {"type": "semantic"}
    msg = f"Unsupported chunking strategy: {strategy}"
    raise ValueError(msg)


class OpenAIVectorDB(VectorDB):
    """Vector database using OpenAI's Vector Stores API."""

    NAME: ClassVar[str] = "openai"
    REQUIRED_PACKAGES: ClassVar = {"openai"}

    def __init__(
        self,
        vector_store_id: str,
        *,
        client: AsyncOpenAI | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        chunking_strategy: ChunkingStrategy = "auto",
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize connection to an existing OpenAI Vector Store.

        Args:
            vector_store_id: ID of the existing vector store
            client: Preconfigured AsyncOpenAI client (optional)
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            organization: OpenAI organization ID (falls back to OPENAI_ORG_ID env var)
            chunking_strategy: Strategy for chunking text
            max_chunk_size: Maximum chunk size in tokens (fixed strategy)
            chunk_overlap: Overlap between chunks in tokens (fixed strategy)
        """
        from openai import AsyncOpenAI

        super().__init__(vector_store_id)
        if client is not None:
            self._client = client
        else:
            api_key = api_key or get_api_key("OPENAI_API_KEY")
            organization = organization or os.getenv("OPENAI_ORG_ID")
            self._client = AsyncOpenAI(api_key=api_key, organization=organization)
        self.chunking_config = to_chunking_config(
            chunking_strategy, max_chunk_size, chunk_overlap
        )

    async def add_chunks(self, chunks: list[TextChunk]) -> list[str]:
        """Add text chunks with metadata.

        Args:
            chunks: List of text chunks to add

        Returns:
            List of IDs for the stored chunks
        """
        file_ids = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]

            # Process each chunk
            batch_ids = []
            for chunk in batch:
                # Prepare metadata
                metadata: dict[str, str | float | bool] = {
                    "source_doc_id": chunk.source_doc_id,
                    "chunk_index": chunk.chunk_index,
                }
                if chunk.page_number is not None:
                    metadata["page_number"] = chunk.page_number
                for key, value in chunk.metadata.items():
                    if isinstance(value, str | bool | int | float):
                        metadata[key] = value  # noqa: PERF403
                file_obj = (
                    f"{chunk.source_doc_id}_{chunk.chunk_index}",
                    io.BytesIO(chunk.text.encode("utf-8")),
                )
                r = await self._client.files.create(file=file_obj, purpose="user_data")
                file_id = r.id
                _file = await self._client.vector_stores.files.create(
                    vector_store_id=self.vector_store_id,
                    file_id=file_id,
                    attributes=metadata,
                    chunking_strategy=self.chunking_config,
                )

                batch_ids.append(file_id)
            # Add batch IDs to overall results
            file_ids.extend(batch_ids)

        return file_ids

    async def query(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, float]]:
        """Find similar chunks for a query.

        Args:
            query: Query text to search for
            k: Number of results to return
            filters: Optional filters to apply to results

        Returns:
            List of (chunk, score) tuples
        """
        from docler.vector_db.dbs.openai_db.utils import convert_filters

        filter_obj = convert_filters(filters) if filters else None
        extra = {"filters": filter_obj} if filter_obj else {}
        try:
            response = await self._client.vector_stores.search(
                vector_store_id=self.vector_store_id,
                query=query,
                max_num_results=k,
                **extra,  # type: ignore
            )
        except Exception:
            logger.exception("Error searching OpenAI vector store")
            return []
        chunks_with_scores = []
        for result in response.data:
            content_text = ""
            for content_item in result.content:
                if content_item.type == "text":
                    content_text += content_item.text + "\n"
            metadata = {}
            if result.attributes:
                metadata = dict(result.attributes)
            source_doc_id = metadata.pop("source_doc_id", str(result.file_id))
            chunk_index = metadata.pop("chunk_index", 0)
            page_number = metadata.pop("page_number", None)
            assert isinstance(source_doc_id, str)
            assert isinstance(chunk_index, int)
            assert isinstance(page_number, int | None)
            chunk = TextChunk(
                text=content_text.strip(),
                source_doc_id=source_doc_id,
                chunk_index=chunk_index,
                page_number=page_number,
                metadata=metadata,
                images=[],
            )

            chunks_with_scores.append((chunk, float(result.score)))

        return chunks_with_scores

    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID.

        In OpenAI's Vector Store API, we delete files, not individual chunks.

        Args:
            chunk_id: ID of chunk to delete

        Returns:
            True if chunk was deleted, False otherwise
        """
        try:
            await self._client.vector_stores.files.delete(
                vector_store_id=self.vector_store_id,
                file_id=chunk_id,
            )
        except Exception:
            logger.exception("Error deleting file from OpenAI vector store")
            return False
        else:
            return True

    async def close(self) -> None:
        """Close resources."""
        return
