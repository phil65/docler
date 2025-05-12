from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mkdown import Document
from schemez import Schema

from docler.common_types import DEFAULT_PROOF_READER_MODEL
from docler.configs.chunker_configs import (
    AiChunkerConfig,
    ChunkerConfig,
    LlamaIndexChunkerConfig,
    MarkdownChunkerConfig,
    TokenAwareChunkerConfig,
)
from docler.configs.processor_configs import (
    DEFAULT_PROOF_READER_PROMPT_TEMPLATE,
    DEFAULT_PROOF_READER_SYSTEM_PROMPT,
    LLMProofReaderConfig,
)
from docler.diffs import generate_all_diffs
from docler.processors.base import DocumentProcessor
from docler.utils import add_line_numbers


if TYPE_CHECKING:
    from docler.configs.file_db_configs import ChunkerShorthand


class LineCorrection(Schema):
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


def resolve_chunker_config(
    config: ChunkerConfig | ChunkerShorthand | None, model: str
) -> ChunkerConfig | None:
    """Resolve chunker configuration from shorthand or create default.

    Args:
        config: Shorthand string, configuration object, or None
        model: Model to use for default token-aware chunker

    Returns:
        Fully resolved configuration object
    """
    if isinstance(config, str):
        match config:
            case "markdown":
                return MarkdownChunkerConfig()
            case "llamaindex":
                return LlamaIndexChunkerConfig()
            case "ai":
                return AiChunkerConfig()
            case "token_aware":
                return TokenAwareChunkerConfig(model=model)

    # Return the config if it's already a config object
    return config


class LLMProofReader(DocumentProcessor[LLMProofReaderConfig]):
    """LLM-based proof-reader that improves OCR output using line-based corrections."""

    Config = LLMProofReaderConfig
    REQUIRED_PACKAGES: ClassVar = {"llmling-agent"}
    NAME = "proof_reading"

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        prompt_template: str | None = None,
        chunker: ChunkerConfig | ChunkerShorthand | None = None,
        include_diffs: bool = True,
        add_metadata_only: bool = False,
    ):
        """Initialize LLM document proof-reader.

        Args:
            model: LLM model to use
            system_prompt: Custom system prompt
            prompt_template: Custom prompt template
            chunker: Custom chunker configuration or shorthand
                     (if None, uses TokenAwareChunker with this model)
            include_diffs: Whether to include diffs in metadata
            add_metadata_only: If True, only add metadata without modifying content
        """
        self.model = model or DEFAULT_PROOF_READER_MODEL
        self.system_prompt = system_prompt or DEFAULT_PROOF_READER_SYSTEM_PROMPT
        self.prompt_template = prompt_template or DEFAULT_PROOF_READER_PROMPT_TEMPLATE
        self.include_diffs = include_diffs
        self.add_metadata_only = add_metadata_only

        # Resolve chunker config
        self.chunker_config = resolve_chunker_config(chunker, self.model)

    async def process(self, doc: Document) -> Document:
        """Process document using line-based corrections."""
        from llmling_agent import Agent

        agent = Agent[None](model=self.model, system_prompt=self.system_prompt)

        corrections = []
        if self.chunker_config:
            # Process with chunking
            chunker = self.chunker_config.get_provider()
            temp_doc = Document(content=doc.content, source_path=doc.source_path)
            chunks = await chunker.split(temp_doc)

            for chunk in chunks:
                numbered_text = chunk.to_numbered_text()
                user_prompt = self.prompt_template.format(chunk_text=numbered_text)

                chunk_corrections = await agent.talk.extract_multiple(
                    text=numbered_text,
                    as_type=LineCorrection,
                    prompt=user_prompt,
                    mode="structured",
                )
                corrections.extend(chunk_corrections)
        else:
            # Process the entire document at once
            numbered_text = add_line_numbers(doc.content)
            user_prompt = self.prompt_template.format(chunk_text=numbered_text)

            corrections = await agent.talk.extract_multiple(
                text=numbered_text,
                as_type=LineCorrection,
                prompt=user_prompt,
                mode="structured",
            )

        new_content, corrected_lines = apply_corrections(doc.content, corrections)
        metadata = doc.metadata.copy() if doc.metadata else {}
        proof_reading = {
            "model": self.model,
            "corrections_count": len(corrected_lines),
            "corrected_lines": sorted(corrected_lines),
            "metadata_only": self.add_metadata_only,
            "corrections": [
                {"line_number": c.line_number, "corrected": c.corrected}
                for c in corrections
            ],
        }
        if self.include_diffs:
            diff_metadata = generate_all_diffs(doc.content, new_content)
            proof_reading.update(diff_metadata)

        metadata[self.NAME] = proof_reading
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
