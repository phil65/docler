"""Token-aware text chunking implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from mkdown import TextChunk

from docler.chunkers.base import TextChunker
from docler.configs.chunker_configs import TokenAwareChunkerConfig
from docler.log import get_logger


if TYPE_CHECKING:
    from mkdown import Document


logger = get_logger(__name__)


def count_tokens(text: str, model_name: str) -> int:
    """Count tokens in text using tokonomics.

    Args:
        text: Text to count tokens for
        model_name: Name of the model to use for tokenization calculation

    Returns:
        Number of tokens in the text
    """
    from tokonomics import count_tokens

    # Extract model name from full model identifier (e.g., "openai/gpt-4" -> "gpt-4")
    model_name = model_name.split("/")[-1] if "/" in model_name else model_name
    model_name = model_name.split(":")[-1] if ":" in model_name else model_name
    return count_tokens(text, model=model_name)


class TokenAwareChunker(TextChunker[TokenAwareChunkerConfig]):
    """Chunker that splits text based on token counts rather than character lengths."""

    REQUIRED_PACKAGES: ClassVar = {"tokonomics"}
    NAME = "token-aware"
    Config = TokenAwareChunkerConfig

    def __init__(
        self,
        *,
        model: str,
        max_tokens_per_chunk: int = 4000,
        chunk_overlap_lines: int = 20,
    ):
        """Initialize token-aware chunker.

        Args:
            model: Model ID to use for tokenization calculation
            max_tokens_per_chunk: Maximum tokens per chunk
            chunk_overlap_lines: Number of lines to overlap between chunks
        """
        super().__init__()
        self.model = model
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.chunk_overlap_lines = chunk_overlap_lines

    async def split(
        self,
        doc: Document,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Split document into chunks based on token count.

        Args:
            doc: Document to split
            extra_metadata: Additional metadata to include in chunks

        Returns:
            List of text chunks
        """
        lines = doc.content.splitlines()
        chunks: list[TextChunk] = []
        chunk_index = 0
        start_idx = 0
        meta = extra_metadata or {}

        # Process until we've gone through all lines
        while start_idx < len(lines):
            # Start with a minimum chunk size of 100 lines or remaining lines
            end_idx = min(start_idx + 100, len(lines))
            current_chunk = "\n".join(lines[start_idx:end_idx])
            token_count = count_tokens(current_chunk, self.model)

            # Keep adding lines until we reach the token limit or end of document
            while end_idx < len(
                lines
            ) and token_count < self.max_tokens_per_chunk - count_tokens(
                lines[end_idx], self.model
            ):
                end_idx += 1
                current_chunk = "\n".join(lines[start_idx:end_idx])
                token_count = count_tokens(current_chunk, self.model)

            # Find images relevant to this chunk
            chunk_images = [
                img
                for img in doc.images
                if img.filename and img.filename in current_chunk
            ]

            # Create the chunk with appropriate metadata
            chunk_metadata = {
                **meta,
                "token_count": token_count,
                "line_range": (start_idx + 1, end_idx),  # 1-based line numbers
                "start_line": start_idx + 1,
                "end_line": end_idx,
            }

            chunk = TextChunk(
                content=current_chunk,
                source_doc_id=doc.source_path or "",
                chunk_index=chunk_index,
                images=chunk_images,
                metadata=chunk_metadata,
            )

            chunks.append(chunk)
            chunk_index += 1

            # Move to next chunk with overlap
            start_idx = end_idx - self.chunk_overlap_lines

            # Avoid getting stuck on the same content
            if start_idx <= chunks[-1].metadata["start_line"] - 1:
                start_idx = chunks[-1].metadata["start_line"] + 50

            # Stop if we've processed all lines
            if start_idx >= len(lines):
                break

        return chunks


if __name__ == "__main__":
    import asyncio

    from mkdown import Document

    async def main():
        # Example usage
        doc = Document(
            source_path="example.txt",
            content="This is a test document.\nIt has multiple lines.\n" * 100,
        )
        chunker = TokenAwareChunker(model="gpt-4")
        chunks = await chunker.split(doc)
        print(f"Split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(
                f"Chunk {i}: {chunk.metadata['token_count']} tokens, "
                f"Lines {chunk.metadata['start_line']}-{chunk.metadata['end_line']}"
            )

    asyncio.run(main())
