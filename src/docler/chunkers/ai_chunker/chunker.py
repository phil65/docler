"""AI-based markdown chunking implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from docler.chunkers.ai_chunker.models import Chunk, Chunks
from docler.chunkers.base import TextChunk, TextChunker
from docler.common_types import DEFAULT_CHUNKER_MODEL


if TYPE_CHECKING:
    from docler.models import Document


SYS_PROMPT = """
You are an expert at dividing text into meaningful chunks
while preserving context and relationships.

The task is to act like a chunker in an RAG pipeline.

Analyze the text and split it into coherent chunks.

All indexes are 1-based. Be accurate with line numbers.
Extract key terms and concepts as keywords
If any block is related to another block, you can add that info.
"""

CHUNKING_PROMPT = """
Here's the text with line numbers:

<text>
{numbered_text}
</text>
"""


class AIChunker(TextChunker):
    """LLM-based document chunker."""

    REQUIRED_PACKAGES: ClassVar = {"llmling-agent"}

    def __init__(
        self,
        model: str | None = None,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
        system_prompt: str | None = None,
    ):
        """Initialize the AI chunker.

        Args:
            model: LLM model to use
            provider: LLM provider to use
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
            system_prompt: System prompt to use
        """
        self.model = model or DEFAULT_CHUNKER_MODEL
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.system_prompt = system_prompt or SYS_PROMPT

    def _add_line_numbers(self, text: str) -> str:
        """Add line numbers to text."""
        lines = text.splitlines()
        return "\n".join(f"{i + 1:3d} | {line}" for i, line in enumerate(lines))

    async def _get_chunks(self, text: str) -> Chunks:
        """Get chunk definitions from LLM."""
        from llmling_agent import Agent

        numbered_text = self._add_line_numbers(text)
        # agent: llmling_agent.StructuredAgent[None, Chunks] = llmling_agent.Agent(
        #     model=self.model,
        #     system_prompt=self.system_prompt,
        # ).to_structured(Chunks)
        # prompt = CHUNKING_PROMPT.format(numbered_text=numbered_text)
        # response = await agent.run(prompt)
        agent: Agent[None] = Agent(model=self.model, system_prompt=self.system_prompt)
        prompt = CHUNKING_PROMPT.format(numbered_text=numbered_text)
        chunks = await agent.talk.extract_multiple(
            text,
            Chunk,
            prompt=prompt,
            mode="structured",  # tool_calls
        )
        return Chunks(chunks=chunks)

    def _create_text_chunk(
        self,
        doc: Document,
        chunk: Chunk,
        chunk_idx: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> TextChunk:
        """Create a TextChunk from chunk definition."""
        lines = doc.content.splitlines()
        chunk_lines = lines[chunk.start_row - 1 : chunk.end_row]
        chunk_text = "\n".join(chunk_lines)
        base = extra_metadata or {}
        metadata = {**base, "keywords": chunk.keywords, "references": chunk.references}
        chunk_images = [i for i in doc.images if i.filename and i.filename in chunk_text]
        return TextChunk(
            text=chunk_text,
            source_doc_id=doc.source_path or "",
            chunk_index=chunk_idx,
            images=chunk_images,
            metadata=metadata,
        )

    async def split(
        self,
        doc: Document,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Split document into chunks using LLM analysis."""
        chunks = await self._get_chunks(doc.content)
        return [
            self._create_text_chunk(doc, chunk, i, extra_metadata)
            for i, chunk in enumerate(chunks.chunks)
        ]


if __name__ == "__main__":
    import asyncio

    from docler.models import Document

    async def main():
        # Example usage
        doc = Document(source_path="example.txt", content=SYS_PROMPT)
        chunker = AIChunker()
        chunks = await chunker.split(doc)
        print(chunks)

    asyncio.run(main())
