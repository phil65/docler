"""Document converter using LiteLLM providers that support PDF input."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from docler.converters.base import DocumentConverter
from docler.models import Document


if TYPE_CHECKING:
    from docler.common_types import StrPath, SupportedLanguage


logger = logging.getLogger(__name__)


USER_PROMPT = """
Convert this PDF document to markdown format
Preserve the original formatting and structure where possible.
Include any important tables or lists.
Describe any images you see in brackets.
{txt}
"""


class LLMConverter(DocumentConverter):
    """Document converter using LLM providers that support PDF input."""

    NAME = "llm"
    REQUIRED_PACKAGES: ClassVar = {"llmling-agent"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {"application/pdf"}

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        model: str = "gemini/gemini-2.0-flash",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ):
        """Initialize the LiteLLM converter.

        Args:
            languages: List of supported languages (used in prompting)
            model: LLM model to use for conversion
            system_prompt: Optional system prompt to guide conversion
            user_prompt: Custom prompt for the conversion task
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response

        Raises:
            ValueError: If model doesn't support PDF input
        """
        super().__init__(languages=languages)

        self.model = model  # .replace(":", "/")
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        txt = ""
        if languages:
            lang_str = ", ".join(languages)
            txt = f"The document may contain text in these languages: {lang_str}."

        self.user_prompt = user_prompt or USER_PROMPT.format(txt)

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a PDF file using the configured LLM.

        Args:
            file_path: Path to the PDF file
            mime_type: MIME type (must be PDF)

        Returns:
            Converted document
        """
        import base64

        import llmling_agent
        import upath

        path = upath.UPath(file_path)

        # Read and encode PDF
        pdf_bytes = path.read_bytes()
        pdf_b64 = base64.b64encode(pdf_bytes).decode()
        content = llmling_agent.ImageBase64Content(
            data=pdf_b64,
            mime_type="application/pdf",
        )
        agent = llmling_agent.Agent[None](
            model=self.model,
            system_prompt=self.system_prompt,
            provider="litellm",  # (pydantic-ai does not work with pdfs yet)
        )
        response = agent.run_sync(self.user_prompt, content)
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

    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"
    converter = LLMConverter(
        languages=["en", "de"],
        user_prompt="Convert this PDF to markdown, focusing on technical details.",
    )
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
