"""Configuration models for document processors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field
from schemez import ModelIdentifier  # noqa: TC002

from docler.common_types import DEFAULT_PROOF_READER_MODEL
from docler.configs.chunker_configs import ChunkerConfig  # noqa: TC001
from docler.provider import ProviderConfig


if TYPE_CHECKING:
    from docler.processors.base import DocumentProcessor


# Default prompts for LLM proof reader
DEFAULT_PROOF_READER_SYSTEM_PROMPT = """\
You are a professional OCR proof-reader. Your task is to correct OCR errors
in the provided text, focusing especially on fixing misrecognized characters,
merged/split words, and formatting issues. Generate corrections only for lines
that need fixing.
"""

DEFAULT_PROOF_READER_PROMPT_TEMPLATE = """\
Proofread the following text and provide corrections for OCR errors.
For each line that needs correction, provide:

LINE_NUMBER: corrected text

Only include lines that need correction. Do not include lines that are correct.
Here is the text with line numbers:

{chunk_text}
"""


class BaseProcessorConfig(ProviderConfig):
    """Base configuration for document processors."""


class LLMProofReaderConfig(BaseProcessorConfig):
    """Configuration for LLM-based proof reader that improves OCR output."""

    type: Literal["llm_proof_reader"] = Field(default="llm_proof_reader", init=False)
    """Type discriminator for LLM proof reader."""

    model: ModelIdentifier = DEFAULT_PROOF_READER_MODEL
    """LLM model to use for proof reading."""

    system_prompt: str = DEFAULT_PROOF_READER_SYSTEM_PROMPT
    """System prompt for the proof reading task."""

    prompt_template: str = DEFAULT_PROOF_READER_PROMPT_TEMPLATE
    """Template for the proof reading prompt."""

    chunker: ChunkerConfig | None = None
    """Optional chunker configuration. If None, processes entire document at once."""

    include_diffs: bool = True
    """Whether to include diffs in metadata."""

    add_metadata_only: bool = False
    """If True, only add metadata without modifying content."""

    def get_provider(self) -> DocumentProcessor:
        """Get the processor instance."""
        from docler.processors.ai_processor import LLMProofReader

        return LLMProofReader(**self.get_config_fields())


# Union type for processor configs
ProcessorConfig = Annotated[
    LLMProofReaderConfig,
    Field(discriminator="type"),
]
