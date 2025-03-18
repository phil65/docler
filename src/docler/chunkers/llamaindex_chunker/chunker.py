"""LlamaIndex-based text chunking implementation."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from docler.chunkers.base import TextChunk, TextChunker


if TYPE_CHECKING:
    from docler.models import Document


logger = logging.getLogger(__name__)

ChunkerType = Literal["sentence", "token", "fixed", "markdown"]


class LlamaIndexChunker(TextChunker):
    """Text chunker using LlamaIndex chunkers.

    This is a wrapper around LlamaIndex's chunking functionality,
    allowing dynamic import to avoid a fixed dependency.
    """

    REQUIRED_PACKAGES: ClassVar[list[str]] = ["llama-index"]

    def __init__(
        self,
        *,
        chunker_type: ChunkerType = "markdown",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        include_metadata: bool = True,
        include_prev_next_rel: bool = False,
    ):
        """Initialize the LlamaIndex chunker.

        Args:
            chunker_type: Which LlamaIndex chunker to use
            chunk_size: Target size of chunks in appropriate units
            chunk_overlap: Overlap between chunks
            include_metadata: Whether to include document metadata in chunks
            include_prev_next_rel: Whether to track relationships between chunks
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunker_type = chunker_type
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel

    def _get_llama_chunker(self):
        """Dynamically import and create a LlamaIndex chunker.

        Returns:
            The appropriate LlamaIndex chunker instance based on configuration

        Raises:
            ImportError: If llama-index isn't installed
        """
        try:
            if self.chunker_type == "sentence":
                from llama_index.core.node_parser import SentenceSplitter

                return SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    include_metadata=self.include_metadata,
                    include_prev_next_rel=self.include_prev_next_rel,
                )
            if self.chunker_type == "token":
                from llama_index.core.node_parser import TokenTextSplitter

                return TokenTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    include_metadata=self.include_metadata,
                    include_prev_next_rel=self.include_prev_next_rel,
                )
            if self.chunker_type == "fixed":
                from llama_index.core.node_parser import SentenceWindowNodeParser

                return SentenceWindowNodeParser.from_defaults(
                    window_size=self.chunk_size,
                    window_metadata_key="window",
                    original_text_metadata_key="original_text",
                )
            # markdown as default
            from llama_index.core.node_parser import MarkdownNodeParser

            return MarkdownNodeParser(
                include_metadata=self.include_metadata,
                include_prev_next_rel=self.include_prev_next_rel,
            )
        except ImportError:
            msg = (
                "LlamaIndex is not installed. "
                "Please install it with `pip install llama-index`"
            )
            raise ImportError(msg) from None

    def _find_images_for_chunk(self, doc: Document, chunk_text: str) -> list:
        """Find images that are referenced in the chunk.

        Args:
            doc: Original document
            chunk_text: Text of the current chunk

        Returns:
            List of images referenced in the chunk
        """
        return [img for img in doc.images if img.filename and img.filename in chunk_text]

    async def split(
        self,
        doc: Document,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Split document into chunks using LlamaIndex.

        Args:
            doc: Document to split
            extra_metadata: Additional metadata to include in chunks

        Returns:
            List of text chunks

        Raises:
            ImportError: If llama-index isn't installed
        """
        try:
            # Convert our document to a LlamaIndex document
            llama_index = importlib.import_module("llama_index.core")
            llama_doc = llama_index.Document(
                text=doc.content,
                metadata={
                    "source": doc.source_path or "",
                    "title": doc.title or "",
                    **(extra_metadata or {}),
                },
            )

            # Get the appropriate chunker
            chunker = self._get_llama_chunker()

            # Parse the document into nodes
            nodes = chunker.get_nodes_from_documents([llama_doc])

            # Convert nodes to TextChunks
            chunks = []
            for i, node in enumerate(nodes):
                # Extract metadata from node
                metadata = {**node.metadata}
                if hasattr(node, "relationships"):
                    metadata["relationships"] = node.relationships

                # Find images referenced in this chunk
                chunk_images = self._find_images_for_chunk(doc, node.get_content())

                # Create TextChunk
                chunk = TextChunk(
                    text=node.get_content(),
                    source_doc_id=doc.source_path or "",
                    chunk_index=i,
                    images=chunk_images,
                    metadata={
                        **(extra_metadata or {}),
                        **metadata,
                    },
                )
                chunks.append(chunk)

            return chunks  # noqa: TRY300

        except ImportError:
            msg = (
                "LlamaIndex is not installed. "
                "Please install it with `pip install llama-index`"
            )
            raise ImportError(msg) from None


if __name__ == "__main__":
    import asyncio

    from docler.models import Document

    async def main():
        # Example usage
        doc = Document(
            source_path="example.txt",
            content="# This is an example\n\nSample content here.",
        )
        chunker = LlamaIndexChunker(chunker_type="markdown")
        chunks = await chunker.split(doc)
        for chunk in chunks:
            print(f"Chunk {chunk.chunk_index}:\n{chunk.text}\n")

    asyncio.run(main())
