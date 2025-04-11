"""Document converter using MarkItDown."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import upath

from docler.configs.converter_configs import MarkItDownConfig
from docler.converters.base import DocumentConverter
from docler.log import get_logger
from docler.models import Document


if TYPE_CHECKING:
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

    def __init__(self, languages: list[SupportedLanguage] | None = None):
        """Initialize the MarkItDown converter."""
        from markitdown import MarkItDown

        super().__init__(languages=languages)
        self.converter = MarkItDown()

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
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
            result = self.converter.convert(str(path), keep_data_uris=True)
            return Document(
                content=result.text_content,
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
    print(result)
