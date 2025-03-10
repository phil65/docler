"""AI-based markdown chunking implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from docler.chunkers.base import TextChunk


if TYPE_CHECKING:
    from collections.abc import Sequence

    from docler.models import Document


class Chunk(BaseModel):
    """A chunk of text with semantic metadata."""

    start_row: int
    end_row: int
    keywords: list[str] = Field(description="Key terms and concepts in this chunk")
    references: list[int] = Field(
        description="Line numbers that this chunk references or depends on"
    )


class Chunks(BaseModel):
    """Collection of chunks with their metadata."""

    chunks: list[Chunk]


CHUNKING_PROMPT = """You are an expert at dividing text into meaningful chunks while preserving context and relationships.

Analyze this text and split it into coherent chunks. For each chunk:
1. Define its start and end line numbers (1-based)
2. Extract key terms and concepts as keywords
3. Note any line numbers it references or depends on

Rules:
- Each chunk should be semantically complete
- Identify cross-references between different parts of the text
- Keywords should be specific and relevant
- Line numbers must be accurate

Here's the text with line numbers:

{numbered_text}

Return only valid JSON matching this schema:
{schema}
"""  # noqa: E501


class AIChunker:
    """LLM-based document chunker."""

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
    ) -> None:
        """Initialize the AI chunker.

        Args:
            model: LLM model to use
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk
        """
        self.model = model
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _add_line_numbers(self, text: str) -> str:
        """Add line numbers to text."""
        lines = text.splitlines()
        return "\n".join(f"{i + 1:3d} | {line}" for i, line in enumerate(lines))

    async def _get_chunks(self, text: str) -> Chunks:
        """Get chunk definitions from LLM."""
        numbered_text = self._add_line_numbers(text)

        # Get JSON schema for expected response
        schema = Chunks.model_json_schema()
        import litellm

        # Prepare prompt
        prompt = CHUNKING_PROMPT.format(
            numbered_text=numbered_text,
            schema=schema,
        )

        # Get response from LLM
        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a text analysis expert.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        # Parse response into our model
        result = response.choices[0].message.content
        return Chunks.model_validate_json(result)

    def _create_text_chunk(
        self,
        doc: Document,
        chunk: Chunk,
        chunk_idx: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> TextChunk:
        """Create a TextChunk from chunk definition."""
        # Get lines for this chunk
        lines = doc.content.splitlines()
        chunk_lines = lines[chunk.start_row - 1 : chunk.end_row]
        chunk_text = "\n".join(chunk_lines)

        # Build metadata
        metadata = {
            **(extra_metadata or {}),
            "keywords": chunk.keywords,
            "references": chunk.references,
        }

        # Find images referenced in these lines
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
    ) -> Sequence[TextChunk]:
        """Split document into chunks using LLM analysis."""
        # Get chunk definitions from LLM
        chunks = await self._get_chunks(doc.content)

        # Convert to TextChunks
        return [
            self._create_text_chunk(doc, chunk, i, extra_metadata)
            for i, chunk in enumerate(chunks.chunks)
        ]


if __name__ == "__main__":
    import asyncio

    from docler.models import Document

    async def main():
        # Example usage
        doc = Document(source_path="example.txt", content=CHUNKING_PROMPT)
        chunker = AIChunker()
        chunks = await chunker.split(doc)
        print(chunks)

    asyncio.run(main())
