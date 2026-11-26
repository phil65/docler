"""Data models for document representation."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from docler.markdown.image import Image


@dataclass
class TextChunk:
    """Chunk of text with associated metadata and images."""

    content: str
    source_doc_id: str
    chunk_index: int
    page_number: int | None = None
    images: list[Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_numbered_text(self, start_line: int | None = None) -> str:
        """Convert chunk text to numbered format.

        Args:
            start_line: The starting line number (1-based)
                        Defaults to metadata value if available

        Returns:
            Text with line numbers prefixed
        """
        if start_line is None:
            start_line = self.metadata.get("start_line", 1)

        lines = self.content.splitlines()
        return "\n".join(f"{start_line + i:5d} | {line}" for i, line in enumerate(lines))  # pyright: ignore
