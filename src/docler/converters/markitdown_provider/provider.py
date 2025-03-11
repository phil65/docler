"""Document converter using MarkItDown."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from docler.base import DocumentConverter
from docler.models import Document


if TYPE_CHECKING:
    from docler.common_types import StrPath
    from docler.lang_code import SupportedLanguage


logger = logging.getLogger(__name__)


class MarkItDownConverter(DocumentConverter):
    """Document converter using MarkItDown."""

    NAME = "markitdown"
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
        # MarkItDown can handle these protocols directly
        "",
        "file",
        "http",
        "https",
    }

    def __init__(self, languages: list[SupportedLanguage] | None = None) -> None:
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
        import upath

        path = upath.UPath(file_path)

        try:
            # Convert using MarkItDown
            result = self.converter.convert(str(path))

            return Document(
                content=result.text_content,
                title=path.stem,
                source_path=str(path),
                mime_type=mime_type,
            )

        except Exception as e:
            msg = f"Failed to convert file {file_path}"
            logger.exception(msg)
            raise ValueError(msg) from e


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.INFO)

    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"
    converter = MarkItDownConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
