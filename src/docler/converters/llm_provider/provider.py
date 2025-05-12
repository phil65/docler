"""Document converter using LiteLLM providers that support PDF input."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mkdown import Document
import upath

from docler.common_types import DEFAULT_CONVERTER_MODEL
from docler.configs.converter_configs import (
    LLM_SYSTEM_PROMPT,
    LLM_USER_PROMPT,
    LLMConverterConfig,
)
from docler.converters.base import DocumentConverter
from docler.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.models.content import BaseContent
    from schemez import MimeType

    from docler.common_types import StrPath, SupportedLanguage


logger = get_logger(__name__)


class LLMConverter(DocumentConverter[LLMConverterConfig]):
    """Document converter using LLM providers that support PDF input."""

    Config = LLMConverterConfig

    NAME = "llm"
    REQUIRED_PACKAGES: ClassVar = {"llmling-agent"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {"application/pdf"}

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        page_range: str | None = None,
        model: str = DEFAULT_CONVERTER_MODEL,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ):
        """Initialize the LiteLLM converter.

        Args:
            languages: List of supported languages (used in prompting)
            page_range: Page range(s) to extract, like "1-5,7-10" (1-based)
            model: LLM model to use for conversion
            system_prompt: Optional system prompt to guide conversion
            user_prompt: Custom prompt for the conversion task

        Raises:
            ValueError: If model doesn't support PDF input
        """
        super().__init__(languages=languages, page_range=page_range)
        self.model = model  # .replace(":", "/")
        self.system_prompt = system_prompt or LLM_SYSTEM_PROMPT
        self.user_prompt = user_prompt or LLM_USER_PROMPT

    def _convert_path_sync(self, file_path: StrPath, mime_type: MimeType) -> Document:
        """Convert a PDF file using the configured LLM.

        Args:
            file_path: Path to the PDF file
            mime_type: MIME type (must be PDF)

        Returns:
            Converted document
        """
        from llmling_agent import Agent, ImageBase64Content, PDFBase64Content

        path = upath.UPath(file_path)
        file_content = path.read_bytes()
        if path.suffix == ".pdf":
            content: BaseContent = PDFBase64Content.from_bytes(file_content)
        else:
            content = ImageBase64Content.from_bytes(file_content)
        agent = Agent[None](model=self.model, system_prompt=self.system_prompt)
        extra = (
            f" Extract only the following pages: {self.page_range}"
            if self.page_range
            else ""
        )
        response = agent.run_sync(self.user_prompt + extra, content)
        return Document(
            content=response.content,
            title=path.stem,
            source_path=str(path),
            mime_type=mime_type,
        )


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.INFO)

    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = LLMConverter(languages=["en", "de"])
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
