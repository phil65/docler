"""AI-based markdown chunking implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from docler.chunkers.ai_chunker.models import Chunk, Chunks
from docler.chunkers.ai_chunker.utils import add_line_numbers, create_text_chunk
from docler.chunkers.base import TextChunker
from docler.common_types import DEFAULT_CHUNKER_MODEL
from docler.models import TextChunk


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
        user_prompt: str | None = None,
        system_prompt: str | None = None,
    ):
        """Initialize the AI chunker.

        Args:
            model: LLM model to use
            provider: LLM provider to use
            user_prompt: User prompt to use
            system_prompt: System prompt to use
        """
        self.model = model or DEFAULT_CHUNKER_MODEL
        self.user_prompt = user_prompt or CHUNKING_PROMPT
        self.system_prompt = system_prompt or SYS_PROMPT

    async def _get_chunks(self, text: str) -> Chunks:
        """Get chunk definitions from LLM."""
        from llmling_agent import Agent

        numbered_text = add_line_numbers(text)
        # agent: llmling_agent.StructuredAgent[None, Chunks] = llmling_agent.Agent(
        #     model=self.model,
        #     system_prompt=self.system_prompt,
        # ).to_structured(Chunks)
        # prompt = CHUNKING_PROMPT.format(numbered_text=numbered_text)
        # response = await agent.run(prompt)
        agent: Agent[None] = Agent(model=self.model, system_prompt=self.system_prompt)
        prompt = self.user_prompt.format(numbered_text=numbered_text)
        chunks = await agent.talk.extract_multiple(
            text,
            Chunk,
            prompt=prompt,
            mode="structured",  # tool_calls
        )
        return Chunks(chunks=chunks)

    async def split(
        self,
        doc: Document,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Split document into chunks using LLM analysis."""
        chunks = await self._get_chunks(doc.content)
        return [
            create_text_chunk(doc, chunk, i, extra_metadata)
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
