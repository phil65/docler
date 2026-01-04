"""Document converter using MarkItDown."""

from __future__ import annotations

from io import BytesIO
import re
from typing import TYPE_CHECKING, ClassVar

from mkdown import create_page_break

from docler.converters.base import ConverterResult, DocumentConverter
from docler.log import get_logger
from docler.pdf_utils import extract_pdf_pages
from docler_config.converter_configs import MarkItDownConfig


if TYPE_CHECKING:
    from re import Match

    from schemez import MimeType

    from docler.common_types import PageRangeString, SupportedLanguage


logger = get_logger(__name__)


class MarkItDownConverter(DocumentConverter[MarkItDownConfig]):
    """Document converter using MarkItDown."""

    Config = MarkItDownConfig

    NAME = "markitdown"
    REQUIRED_PACKAGES: ClassVar = {"markitdown"}
    SUPPORTED_MIME_TYPES: ClassVar[set[str]] = {
        # PDFs
        "application/pdf",
        # Office documents
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.oasis.opendocument.text",
        "application/rtf",
        # Ebooks and markup
        "application/epub+zip",
        "text/html",
        "text/markdown",
        "text/plain",
        "text/x-rst",
        "text/org",
        # Images for OCR
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/bmp",
        "image/webp",
        "image/gif",
    }

    SUPPORTED_PROTOCOLS: ClassVar[set[str]] = {"", "file", "http", "https"}

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        page_range: PageRangeString | None = None,
    ) -> None:
        """Initialize the MarkItDown converter.

        Args:
            languages: List of supported languages.
            page_range: Page range(s) to extract, like "1-5,7-10" (1-based)
        """
        from markitdown import MarkItDown

        super().__init__(languages=languages, page_range=page_range)
        self.converter = MarkItDown()

    def _convert_sync(self, data: BytesIO, mime_type: MimeType) -> ConverterResult:
        """Convert a file using MarkItDown.

        Args:
            data: File content as BytesIO.
            mime_type: MIME type of the file.

        Returns:
            Intermediate conversion result.

        Raises:
            ValueError: If conversion fails.
        """
        try:
            if mime_type == "application/pdf" and self.page_range:
                # For PDFs with page range, extract specific pages
                pdf_data = data.read()
                extracted_bytes = extract_pdf_pages(pdf_data, self.page_range)
                buffer = BytesIO(extracted_bytes)
                result = self.converter.convert(buffer)
            else:
                # For other files or full PDFs, use regular conversion
                result = self.converter.convert(data, keep_data_uris=True)

            def replace_slide_marker(match: Match[str]) -> str:
                slide_num = match.group(1) if match.groups() else "?"
                try:
                    page_num = int(slide_num)
                except ValueError:
                    page_num = 1
                return create_page_break(next_page=page_num)

            slide_pattern = r"<!-- Slide number:\s*(\d+)\s*-->"
            markdown = re.sub(slide_pattern, replace_slide_marker, result.text_content)

            return ConverterResult(content=markdown, title=result.title)

        except Exception as e:
            msg = f"Failed to convert file: {e}"
            self.logger.exception(msg)
            raise ValueError(msg) from e


if __name__ == "__main__":
    import anyenv

    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = MarkItDownConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
