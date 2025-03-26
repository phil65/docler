from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from docler.common_types import DEFAULT_PROOF_READER_MODEL
from docler.configs.processor_configs import (
    DEFAULT_PROOF_READER_PROMPT_TEMPLATE,
    DEFAULT_PROOF_READER_SYSTEM_PROMPT,
    LLMProofReaderConfig,
)
from docler.diffs import generate_all_diffs
from docler.models import Document
from docler.processors.base import DocumentProcessor


class LineCorrection(BaseModel):
    """A correction to apply to a specific line."""

    line_number: int
    """The line number to correct (1-based)."""

    corrected: str
    """The corrected text."""


def apply_corrections(
    text: str, corrections: list[LineCorrection]
) -> tuple[str, set[int]]:
    """Apply corrections to the original text.

    Args:
        text: Original text to apply corrections to
        corrections: List of line corrections

    Returns:
        Tuple containing (corrected text, set of corrected line indices)
    """
    lines = text.splitlines()
    corrections.sort(key=lambda c: c.line_number, reverse=True)
    corrected_lines = set()

    for correction in corrections:
        line_idx = correction.line_number - 1
        if 0 <= line_idx < len(lines) and line_idx not in corrected_lines:
            lines[line_idx] = correction.corrected
            corrected_lines.add(line_idx)

    return "\n".join(lines), corrected_lines


def add_line_numbers(text: str) -> str:
    """Add line numbers to text."""
    lines = text.splitlines()
    return "\n".join(f"{i + 1:5d} | {line}" for i, line in enumerate(lines))


class LLMProofReader(DocumentProcessor[LLMProofReaderConfig]):
    """LLM-based proof-reader that improves OCR output using line-based corrections."""

    Config = LLMProofReaderConfig
    REQUIRED_PACKAGES: ClassVar = {"llmling-agent"}

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        prompt_template: str | None = None,
        max_chunk_tokens: int = 10000,
        chunk_overlap_lines: int = 20,
        include_diffs: bool = True,
        add_metadata_only: bool = False,
    ):
        """Initialize LLM document proof-reader.

        Args:
            model: LLM model to use
            system_prompt: Custom system prompt
            prompt_template: Custom prompt template
            max_chunk_tokens: Maximum tokens per chunk
            chunk_overlap_lines: Overlap between chunks in lines
            include_diffs: Whether to include diffs in metadata
            add_metadata_only: If True, only add metadata without modifying content
        """
        self.model = model or DEFAULT_PROOF_READER_MODEL
        self.system_prompt = system_prompt or DEFAULT_PROOF_READER_SYSTEM_PROMPT
        self.prompt_template = prompt_template or DEFAULT_PROOF_READER_PROMPT_TEMPLATE
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_lines = chunk_overlap_lines
        self.include_diffs = include_diffs
        self.add_metadata_only = add_metadata_only

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tokonomics."""
        from tokonomics import count_tokens

        return count_tokens(text, model=self.model.split(":")[-1])

    def _split_into_chunks(self, text: str) -> list[tuple[int, str]]:
        """Split text into chunks with line numbers based on token count.

        Returns:
            List of (start_line, numbered_text) tuples
        """
        lines = text.splitlines()
        chunks = []

        start_idx = 0
        while start_idx < len(lines):
            # Start with a minimum chunk size
            end_idx = min(start_idx + 100, len(lines))

            # Add lines until we reach the token limit or end of document
            current_chunk = "\n".join(
                f"{start_idx + i + 1:5d} | {line}"
                for i, line in enumerate(lines[start_idx:end_idx])
            )
            token_count = self._count_tokens(current_chunk)

            # If we have room for more lines, keep adding them
            while end_idx < len(
                lines
            ) and token_count < self.max_chunk_tokens - self._count_tokens(
                lines[end_idx]
            ):
                end_idx += 1
                current_chunk = "\n".join(
                    f"{start_idx + i + 1:5d} | {line}"
                    for i, line in enumerate(lines[start_idx:end_idx])
                )
                token_count = self._count_tokens(current_chunk)

            # Add this chunk to our list
            chunks.append((start_idx + 1, current_chunk))  # 1-based line numbers

            # Move to next chunk with overlap
            start_idx = end_idx - self.chunk_overlap_lines
            if start_idx <= chunks[-1][0] - 1:
                start_idx = chunks[-1][0] + 50

            # Stop if we've processed all lines
            if start_idx >= len(lines):
                break

        return chunks

    async def process(self, doc: Document) -> Document:
        """Process document using line-based corrections.

        If add_metadata_only is True, adds metadata about corrections but doesn't
        modify the document content.
        """
        from llmling_agent import Agent

        agent = Agent[None](model=self.model, system_prompt=self.system_prompt)
        all_corrections = []

        # If document is small enough, process it all at once
        numbered_text = add_line_numbers(doc.content)
        if self._count_tokens(numbered_text) <= self.max_chunk_tokens:
            user_prompt = self.prompt_template.format(chunk_text=numbered_text)

            corrections = await agent.talk.extract_multiple(
                text=numbered_text,
                as_type=LineCorrection,
                prompt=user_prompt,
                mode="structured",
            )
            all_corrections = corrections
        else:
            # Process larger documents in chunks
            chunks = self._split_into_chunks(doc.content)

            for _, chunk_text in chunks:
                user_prompt = self.prompt_template.format(chunk_text=chunk_text)
                chunk_corrections = await agent.talk.extract_multiple(
                    text=chunk_text,
                    as_type=LineCorrection,
                    prompt=user_prompt,
                    mode="structured",
                )
                all_corrections.extend(chunk_corrections)

        new_content, corrected_lines = apply_corrections(doc.content, all_corrections)
        metadata = doc.metadata.copy() if doc.metadata else {}
        proof_reading = {
            "model": self.model,
            "corrections_count": len(corrected_lines),
            "corrected_lines": sorted(corrected_lines),
            "metadata_only": self.add_metadata_only,
            "corrections": [
                {"line_number": c.line_number, "corrected": c.corrected}
                for c in all_corrections
            ],
        }

        if self.include_diffs:
            diff_metadata = generate_all_diffs(doc.content, new_content)
            # Add diffs to the proof_reading sub-dictionary
            proof_reading.update(diff_metadata)

        metadata["proof_reading"] = proof_reading
        final_content = new_content if not self.add_metadata_only else doc.content
        return Document(
            content=final_content,
            images=doc.images,
            title=doc.title,
            author=doc.author,
            created=doc.created,
            modified=doc.modified,
            source_path=doc.source_path,
            mime_type=doc.mime_type,
            page_count=doc.page_count,
            metadata=metadata,
        )


if __name__ == "__main__":
    import anyenv

    async def main():
        # Create a test document with OCR errors
        test_content = """\
OCR Test Document
This 1s a test document with some common OCR errors.
VVords are sometimes m1staken for other characters.
Spaces occasionally getremoved between words.
The letter 'l' is often confused with the number '1'.
Line endings may be incorrectlybroken.
Punctuation,marks can be misplaced , or incorrect.
Special @characters# might not be recognized properly.
numbers like 5678 can be misread as S67B."""

        doc = Document(
            content=test_content,
            title="Test OCR Document",
            source_path="test_document.txt",
            mime_type="text/plain",
        )
        proofreader = LLMProofReader(
            model=DEFAULT_PROOF_READER_MODEL,
            max_chunk_tokens=4000,
            include_diffs=True,
        )

        print("Original content:")
        print("-" * 50)
        print(test_content)
        print("-" * 50)
        print("Processing with LLM proof reader...")
        corrected_doc = await proofreader.process(doc)
        print("\nCorrected content:")
        print("-" * 50)
        print(corrected_doc.content)
        print("-" * 50)

        # Display proof reading metadata
        if "proof_reading" in corrected_doc.metadata:
            proof_reading = corrected_doc.metadata["proof_reading"]

            print("\nProof reading metadata:")
            for key, value in proof_reading.items():
                if key == "unified_diff":
                    print(f"\n{key}:")
                    print("-" * 50)
                    print(value)
                    print("-" * 50)
                elif key not in ("html_diff", "semantic_diff"):
                    print(f"{key}: {value}")

            print(f"\nCorrected {proof_reading.get('corrections_count', 0)} lines")

        return corrected_doc

    corrected = anyenv.run_sync(main())
    print(f"\nProofreading complete! Processed document: {corrected.title}")
