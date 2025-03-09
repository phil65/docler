"""Document converter using Kreuzberg's extraction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from docler.base import DocumentConverter
from docler.lang_code import TESSERACT_CODES, SupportedLanguage
from docler.models import Document


if TYPE_CHECKING:
    from datetime import datetime

    from docler.common_types import StrPath


logger = logging.getLogger(__name__)


class KreuzbergConverter(DocumentConverter):
    """Document converter using Kreuzberg's extraction."""

    NAME = "kreuzberg"
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

    def __init__(
        self,
        languages: list[SupportedLanguage] | None = None,
        *,
        force_ocr: bool = False,
        max_processes: int | None = None,
    ) -> None:
        """Initialize the Kreuzberg converter.

        Args:
            languages: Language codes for OCR.
            force_ocr: Whether to force OCR even on digital documents.
            max_processes: Maximum number of parallel processes.
        """
        super().__init__(languages=languages)
        self.force_ocr = force_ocr
        if languages:
            self.language = TESSERACT_CODES.get(languages[0])
        else:
            self.language = "eng"
        self.max_processes = max_processes

    def _convert_path_sync(self, file_path: StrPath, mime_type: str) -> Document:
        """Convert a file using Kreuzberg.

        Args:
            file_path: Path to the file to process.
            mime_type: MIME type of the file.

        Returns:
            Converted document.
        """
        import anyenv
        from kreuzberg import extract_file
        from kreuzberg._constants import DEFAULT_MAX_PROCESSES
        import upath

        local_file = upath.UPath(file_path)

        # Extract content using Kreuzberg
        result = anyenv.run_sync(
            extract_file(
                str(local_file),
                force_ocr=self.force_ocr,
                language=self.language or "eng",
                max_processes=self.max_processes or DEFAULT_MAX_PROCESSES,
            )
        )

        # Convert metadata
        metadata = result.metadata

        # Parse date if present
        created: datetime | None = None
        if date_str := metadata.get("date"):
            try:
                from dateutil import parser

                created = parser.parse(date_str)
            except Exception:  # noqa: BLE001
                pass
        authors = metadata.get("authors")
        author = authors[0] if authors else None
        return Document(
            content=result.content,
            title=metadata.get("title"),
            author=metadata.get("creator") or author,
            created=created,
            source_path=str(local_file),
            mime_type=result.mime_type or mime_type,
        )


if __name__ == "__main__":
    import logging

    import anyenv

    logging.basicConfig(level=logging.DEBUG)

    pdf_path = "C:/Users/phili/Downloads/CustomCodeMigration_EndToEnd.pdf"
    converter = KreuzbergConverter()
    result = anyenv.run_sync(converter.convert_file(pdf_path))
    print(result)
