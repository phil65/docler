"""Document converter using PyMuPDF."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fitz  # PyMuPDF

from docler.log import get_logger
from docler.models import PageDimensions, PageMetadata


if TYPE_CHECKING:
    from docler.common_types import PageRangeString


logger = get_logger(__name__)


def parse_page_range(page_range: PageRangeString, shift: int = 0) -> set[int]:
    """Convert a page range string to a set of page numbers.

    Args:
        page_range: String like "1-5,7,9-11" or None.
        shift: Amount to shift page numbers by (e.g., -1 to convert 1-based to 0-based)

    Returns:
        Set of page numbers (shifted by specified amount)

    Raises:
        ValueError: If the page range format is invalid.
    """
    if shift:
        page_range = shift_page_range(page_range, shift)

    pages: set[int] = set()
    try:
        for part in page_range.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                pages.update(range(start, end + 1))
            else:
                pages.add(int(part))
    except ValueError as e:
        msg = f"Invalid page range format: {page_range}. Expected format: '1-5,7,9-11'"
        raise ValueError(msg) from e
    else:
        return pages


def shift_page_range(page_range: PageRangeString, shift: int = 0) -> PageRangeString:
    """Shift page numbers in a page range string by the specified amount.

    Args:
        page_range: Page range string like "1-5,7,9-11"
        shift: Amount to shift page numbers by (e.g., -1 to convert 1-based to 0-based)

    Returns:
        Shifted page range string

    Raises:
        ValueError: If the page range format is invalid
    """
    parts = []
    try:
        for part in page_range.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                if start + shift < 0 or end + shift < 0:
                    msg = f"Invalid shift {shift} for page range {page_range}"
                    raise ValueError(msg)  # noqa: TRY301
                parts.append(f"{start + shift}-{end + shift}")
            else:
                page = int(part)
                if page + shift < 0:
                    msg = f"Invalid shift {shift} for page {page}"
                    raise ValueError(msg)  # noqa: TRY301
                parts.append(str(page + shift))
    except ValueError as e:
        if "Invalid shift" in str(e):
            raise
        msg = f"Invalid page range format: {page_range}. Expected format: '1-5,7,9-11'"
        raise ValueError(msg) from e

    return ",".join(parts)


def extract_pdf_pages(data: bytes, page_range: PageRangeString | None) -> bytes:
    """Extract specific pages from a PDF file and return as new PDF.

    Args:
        data: Source PDF file content as bytes
        page_range: String like "1-5,7,9-11" or None for all pages. 1-based.

    Returns:
        New PDF containing only specified pages as bytes

    Raises:
        ValueError: If page range is invalid or PDF data cannot be processed
    """
    try:
        # Open the source PDF from bytes
        source_doc = fitz.open(stream=data, filetype="pdf")

        # Determine which pages to extract
        pages = (
            parse_page_range(page_range, shift=-1)
            if page_range
            else range(len(source_doc))
        )

        # Create new PDF with selected pages
        output_doc = fitz.open()  # Create empty PDF
        for i in pages:
            if 0 <= i < len(source_doc):
                output_doc.insert_pdf(source_doc, from_page=i, to_page=i)

        # Get PDF as bytes
        pdf_bytes = output_doc.tobytes()

        # Clean up
        source_doc.close()
        output_doc.close()
    except Exception as e:
        msg = f"Failed to extract pages from PDF: {e}"
        raise ValueError(msg) from e
    else:
        return pdf_bytes


def get_pdf_info(data: bytes) -> PageMetadata:
    """Get PDF metadata including page count, dimensions, and file info.

    Args:
        data: PDF file content as bytes

    Returns:
        PageMetadata model containing PDF information

    Raises:
        ValueError: If PDF data cannot be processed
    """
    try:
        doc = fitz.open(stream=data, filetype="pdf")

        # Basic info
        page_count = len(doc)
        file_size = len(data)
        is_encrypted = doc.needs_pass

        # Page dimensions (in points)
        page_dimensions = []
        for page_num in range(page_count):
            page = doc[page_num]
            rect = page.rect
            page_dimensions.append(PageDimensions(width=rect.width, height=rect.height))

        # Document metadata
        metadata = doc.metadata
        title = metadata.get("title", "") if metadata else ""
        author = metadata.get("author", "") if metadata else ""

        doc.close()

        return PageMetadata(
            page_count=page_count,
            file_size=file_size,
            is_encrypted=is_encrypted,
            page_dimensions=page_dimensions,
            title=title,
            author=author,
        )
    except Exception as e:
        msg = f"Failed to get PDF info: {e}"
        raise ValueError(msg) from e
