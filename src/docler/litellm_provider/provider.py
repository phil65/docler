"""Document converter using LiteLLM providers that support PDF input."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from litellm import completion

from docler.base import DocumentConverter
from docler.models import Document


if TYPE_CHECKING:
    from docler.common_types import StrPath
    from docler.lang_code import SupportedLanguage


logger = logging.getLogger(__name__)


class LiteLLMConverter(DocumentConverter):
    """Document converter using LLM providers that support PDF input."""

    NAME = "litellm"
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
    ) -> None:
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

        # Build prompt
        prompt_parts = []
        prompt_parts.append("Convert this PDF document to markdown format.")
        if languages:
            lang_str = ", ".join(languages)
            prompt_parts.append(
                f"The document may contain text in these languages: {lang_str}."
            )
        prompt_parts.extend([
            "Preserve the original formatting and structure where possible.",
            "Include any important tables or lists.",
            "Describe any images you see in brackets.",
        ])
        self.user_prompt = user_prompt or " ".join(prompt_parts)

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a PDF file using the configured LLM.

        Args:
            file_path: Path to the PDF file
            mime_type: MIME type (must be PDF)

        Returns:
            Converted document
        """
        import upath

        path = upath.UPath(file_path)

        # Read and encode PDF
        pdf_bytes = path.read_bytes()
        pdf_b64 = base64.b64encode(pdf_bytes).decode()
        pdf_data = f"data:application/pdf;base64,{pdf_b64}"

        # Prepare messages
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": self.user_prompt},
                {"type": "image_url", "image_url": pdf_data},
            ],
        })

        # Get response from LLM
        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return Document(
            content=response.choices[0].message.content,  # type: ignore
            title=path.stem,
            source_path=str(path),
            mime_type=mime_type,
        )


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.INFO)

    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"
    converter = LiteLLMConverter(
        languages=["en", "de"],
        user_prompt="Convert this PDF to markdown, focusing on technical details.",
    )
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
