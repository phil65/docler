"""Document converter using LiteLLM providers that support PDF input."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from docler.common_types import DEFAULT_CONVERTER_MODEL
from docler.converters.base import ConverterResult, DocumentConverter
from docler.log import get_logger
from docler_config.converter_configs import (
    LLM_SYSTEM_PROMPT,
    LLM_USER_PROMPT,
    LLMConverterConfig,
)


if TYPE_CHECKING:
    from io import BytesIO

    from llmling_agent.models.content import BaseContent
    from schemez import MimeType

    from docler.common_types import PageRangeString, SupportedLanguage


logger = get_logger(__name__)


class LLMConverter(DocumentConverter[LLMConverterConfig]):
    """Document converter using LLM providers that support PDF input."""

    Config = LLMConverterConfig

    NAME = "llm"
    REQUIRED_PACKAGES: ClassVar = {"agentpool"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {"application/pdf"}

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        page_range: PageRangeString | None = None,
        model: str = DEFAULT_CONVERTER_MODEL,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> None:
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
        self.model = model
        self.system_prompt = system_prompt or LLM_SYSTEM_PROMPT
        self.user_prompt = user_prompt or LLM_USER_PROMPT

    def _convert_sync(self, data: BytesIO, mime_type: MimeType) -> ConverterResult:
        """Convert a PDF file using the configured LLM.

        Args:
            data: File content as BytesIO.
            mime_type: MIME type (must be PDF).

        Returns:
            Intermediate conversion result.
        """
        from agentpool import Agent
        from pydantic_ai import BinaryContent, BinaryImage

        file_content = data.read()
        if mime_type == "application/pdf":
            content: BaseContent = BinaryContent(file_content, media_type=mime_type)
        else:
            content = BinaryImage(file_content, media_type=mime_type)
        agent = Agent(model=self.model, system_prompt=self.system_prompt)
        extra = f" Extract only the following pages: {self.page_range}" if self.page_range else ""
        response = agent.run.sync(self.user_prompt + extra, content)  # type: ignore[attr-defined]
        return ConverterResult(content=response.content)


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.INFO)

    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = LLMConverter(languages=["en", "de"])
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
