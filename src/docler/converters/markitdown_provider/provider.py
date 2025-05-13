"""Document converter using MarkItDown."""

from __future__ import annotations

from io import BytesIO
import re
from re import Match
from typing import TYPE_CHECKING, ClassVar

from mkdown import Document, create_page_break
import upath

from docler.configs.converter_configs import MarkItDownConfig
from docler.converters.base import DocumentConverter
from docler.log import get_logger
from docler.pdf_utils import extract_pdf_pages


if TYPE_CHECKING:
    from schemez import MimeType

    from docler.common_types import StrPath, SupportedLanguage


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

    SUPPORTED_PROTOCOLS: ClassVar[set[str]] = {
        "",
        "file",
        "http",
        "https",
    }

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        page_range: str | None = None,
    ):
        """Initialize the MarkItDown converter.

        Args:
            languages: List of supported languages.
            page_range: Page range(s) to extract, like "1-5,7-10" (1-based)
        """
        from markitdown import MarkItDown

        super().__init__(languages=languages, page_range=page_range)
        self.converter = MarkItDown()

    def _convert_path_sync(self, file_path: StrPath, mime_type: MimeType) -> Document:
        """Convert a file using MarkItDown.

        Args:
            file_path: Path to the file to process.
            mime_type: MIME type of the file.

        Returns:
            Converted document.

        Raises:
            ValueError: If conversion fails.
        """
        path = upath.UPath(file_path)
        try:
            if mime_type == "application/pdf" and self.page_range:
                # For PDFs with page range:
                # 1. Extract specific pages with pdfminer
                # 2. Convert extracted text with markitdown
                pdf_data = path.read_bytes()
                extracted_bytes = extract_pdf_pages(pdf_data, self.page_range)
                buffer = BytesIO(extracted_bytes)
                result = self.converter.convert(buffer)
            else:
                # For other files or full PDFs, use regular conversion
                result = self.converter.convert(str(path), keep_data_uris=True)

            # Rest of the processing stays the same...
            def replace_slide_marker(match: Match[str]) -> str:
                slide_num = match.group(1) if match.groups() else "?"
                try:
                    page_num = int(slide_num)
                except ValueError:
                    page_num = 1
                return create_page_break(next_page=page_num)

            slide_pattern = r"<!-- Slide number:\s*(\d+)\s*-->"
            markdown = re.sub(slide_pattern, replace_slide_marker, result.text_content)

            return Document(
                content=markdown,
                title=result.title or path.stem,
                source_path=str(path),
                mime_type=mime_type,
            )

        except Exception as e:
            msg = f"Failed to convert file {file_path}"
            self.logger.exception(msg)
            raise ValueError(msg) from e


if __name__ == "__main__":
    import anyenv

    pdf_path = "src/docler/resources/pdf_sample.pdf"
    converter = MarkItDownConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result.content)
